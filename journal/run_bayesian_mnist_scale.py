"""Run the separate fair MNIST positive-variance validation control.

The main campaign is read only. Original validation records are captured once;
512 added candidates train without test evaluation. Complete 32-candidate
selection precedes the fixed-seed confirmation and both full-risk sweeps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .build_bayesian_mnist_scale import (build_screen, build_selected, canonical_validation_rows,
    encoded, freeze_original, original_mnist_entries, read_entries, write_once)
from .build_campaign import make_entry
from .campaign import completed_trial, save_json, validate_entries
from .run_experiment import normalize_config


SOURCE_FILES = ("journal/run_experiment.py", "journal/theorem1.py",
                "journal/cifar_models.py", "journal/cifar_search_models.py",
                "journal/campaign.py", "journal/build_bayesian_campaign.py", "journal/build_campaign.py")
CONTROL_FILES = ("journal/build_bayesian_mnist_scale.py", "journal/run_bayesian_mnist_scale.py",
                 "journal/mnist_scale_control.slurm")
STAGES = ("pilot", "screen", "confirm", "m_sweep", "p_train_sweep")
COUNTS = dict(pilot=16, screen=512, confirm=80, m_sweep=264, p_train_sweep=168)


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verify_sources(main_revision, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
    main = git("rev-parse", "--verify", "--end-of-options", main_revision + "^{commit}")
    execution = git("rev-parse", "HEAD^{commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use a clean committed checkout for the separate MNIST control")
    files = {}
    for filename in SOURCE_FILES:
        source = git("rev-parse", main + ":" + filename)
        committed = git("rev-parse", execution + ":" + filename)
        working = git("hash-object", "--", filename)
        if source != committed or source != working:
            raise ValueError(f"MNIST control source differs from running main source: {filename}")
        files[filename] = dict(main_blob=source, control_blob=committed, working_blob=working)
    controls = {}
    for filename in CONTROL_FILES:
        committed = git("rev-parse", execution + ":" + filename)
        if committed != git("hash-object", "--", filename):
            raise ValueError(f"MNIST coordinator source differs from its committed file: {filename}")
        controls[filename] = committed
    return dict(main_revision=main, execution_revision=execution, files=files, control_files=controls)


def capture_original(main_root, output, main_revision):
    """Read only training records; a frozen owned capture is reused on resume."""
    main_root, output = Path(main_root), Path(output)
    owned = output / "inputs"
    protocol_path = owned / "capture_protocol.json"
    capture = owned / "source_capture"
    manifest, training = capture / "screen.jsonl", capture / "training_results.json"
    frozen = owned / "original"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["main_root"] != str(main_root.resolve()) or protocol["main_revision"] != main_revision:
            raise ValueError("Frozen MNIST source belongs to another main campaign")
        for path in (manifest, training):
            if fingerprint(path) != protocol["capture_fingerprints"][str(path.resolve())]:
                raise ValueError("Frozen original MNIST source capture changed")
    else:
        state = json.loads((main_root / "pipeline_status.json").read_text())
        if state.get("git_sha") != main_revision:
            raise ValueError("Main campaign status records a different source revision")
        main_manifest, main_training = main_root / "manifests" / "screen.jsonl", main_root / "screen" / "training_results.json"
        if any(path.is_symlink() or not path.is_file() for path in (main_manifest, main_training)):
            raise ValueError("Require the original main manifest and complete MNIST training records")
        manifest_raw = manifest.read_bytes() if manifest.exists() else main_manifest.read_bytes()
        training_raw = training.read_bytes() if training.exists() else main_training.read_bytes()
        recorded = state.get("stages", {}).get("screen", {}).get("manifest_sha256")
        if hashlib.sha256(manifest_raw).hexdigest() != recorded or fingerprint(main_manifest)["sha256"] != recorded:
            raise ValueError("Main screening manifest differs from its immutable recorded hash")
        entries = original_mnist_entries([json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()])
        ids = {entry["task_id"] for entry in entries}
        records = json.loads(training_raw)
        canonical_validation_rows(entries, [row for row in records if row["task_id"] in ids], 16)
        # Atomic main writes can advance other datasets after this read. The
        # captured bytes identify one complete snapshot and are never refreshed.
        write_once(manifest, manifest_raw.decode())
        write_once(training, training_raw.decode())
        protocol = dict(main_root=str(main_root.resolve()), main_revision=main_revision,
            main_manifest=str(main_manifest.resolve()), main_training_results=str(main_training.resolve()),
            original_mnist_tasks=512, official_test_files_opened=False,
            capture_fingerprints={str(path.resolve()): fingerprint(path) for path in (manifest, training)})
        write_once(protocol_path, encoded(protocol))
    freeze_original(manifest, training, frozen)
    return frozen / "original_manifest.jsonl", frozen / "original_training_results.json", protocol


def pilot_entries(extra):
    first = {}
    for source in extra:
        first.setdefault((source["group"], source["method_label"]), source)
    if len(first) != 16:
        raise ValueError("Full-shape MNIST pilot requires every law/method/regime")
    output = []
    for _, source in sorted(first.items()):
        config = {**source["config"], "epochs": 1, "posterior_std": .001, "rho_lr_scale": 0.}
        item = {**source, **make_entry(config, 76200, source["group"], source["method_label"],
                                      "bayes_mnist_scale_pilot", "resource_pilot")}
        item.update(family=source["family"], experiment="resource_pilot", scientific_result=False)
        output.append(item)
    validate_entries(output)
    return output


def write_pilot(extra_manifest, output):
    entries = pilot_entries(read_entries(extra_manifest))
    text = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    write_once(output, text)
    write_once(Path(output).with_suffix(".plan.json"), encoded(dict(stage="pilot", tasks=16,
        scientific_result=False, full_shape=True, seed=76200, predictor="positive-variance Gaussian", epochs=1,
        manifest_sha256=hashlib.sha256(text.encode()).hexdigest())))


def command(arguments, log, environment):
    print(json.dumps(dict(command=arguments, log=str(log), started_at=now())), flush=True)
    with Path(log).open("a") as stream:
        subprocess.run([sys.executable, "-u", "-m", *arguments], stdout=stream,
                       stderr=subprocess.STDOUT, env=environment, check=True)


def validate_stage(stage_root, manifest, execution_revision, stage):
    entries = read_entries(manifest)
    validate_entries(entries)
    if len(entries) != COUNTS[stage]:
        raise ValueError(f"Unexpected MNIST control stage task count: {stage}")
    summary = json.loads((stage_root / "summary.json").read_text())
    if (summary.get("training_trials") != len(entries) or summary.get("completed_trials") != len(entries)
            or summary.get("failed_trials") or summary.get("failed_evaluations")):
        raise ValueError(f"Incomplete MNIST control stage: {stage}")
    rows = json.loads((stage_root / "training_results.json").read_text())
    expected = {entry["task_id"]: entry for entry in entries}
    if len(rows) != len(expected) or {row["task_id"] for row in rows} != set(expected):
        raise ValueError("Training stage lacks exact declared task coverage")
    for row in rows:
        entry = expected[row["task_id"]]
        config, declared = normalize_config(row["config"]), normalize_config(entry["config"])
        if (row.get("status") != "complete" or config != declared or config["device"] != "cpu"
                or config["threads"] != 2 or config["predictor"] != "gaussian"
                or config["posterior_std"] <= 0
                or any(row.get(key) != entry.get(key) for key in ("group", "method_label", "variant_id", "family", "stage"))
                or row.get("seed") != declared["seed"]):
            raise ValueError("Training result differs from its declared positive-Gaussian configuration")
        saved = completed_trial(stage_root / entry["task_id"], config)
        if saved is None or saved["git_sha"] != execution_revision:
            raise ValueError("Require a complete checkpoint from the committed control checkout")
        if json.loads((stage_root / entry["task_id"] / "metadata.json").read_text()).get("git_dirty") is not False:
            raise ValueError("MNIST control checkpoints must record clean committed training sources")
        for metric in ("validation_accuracy", "validation_cross_entropy", "seconds", "peak_rss_mb"):
            if not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError("Invalid validation or resource result")
    if stage in {"pilot", "screen"}:
        if (stage_root / "test_results.json").exists() or summary.get("test_comparisons"):
            raise ValueError("Pilot and screen must never produce official test results")
    else:
        from .plot_bayesian_campaign import Evidence
        evidence = Evidence(stage_root, manifest)
        if evidence.excluded_task_ids or len(evidence.rows) != len(entries):
            raise ValueError("Require exact complete full-risk confirmation/sweep reports")
        draws = 20 if stage == "confirm" else 10
        for row in evidence.rows:
            config = row["_report"]["evaluation_config"]
            if (config["eval_draws"] != draws or config["eval_seed"] != row["config"]["eval_seed"] + 1000003
                    or config["device"] != "cpu" or config["threads"] != 2):
                raise ValueError("Final MNIST risk evaluation differs from its fixed fresh MC stream")
    return rows


def memory_projection(rows, workers=16):
    if len(rows) != 16 or any(not isinstance(row.get("peak_rss_mb"), (int, float))
                              or not math.isfinite(row["peak_rss_mb"]) or row["peak_rss_mb"] <= 0 for row in rows):
        raise ValueError("All sixteen full-shape pilots must report positive finite peak RSS")
    maximum = max(row["peak_rss_mb"] for row in rows)
    projected = 2048 + 1.25 * workers * maximum
    allocated = int(os.environ.get("SLURM_MEM_PER_NODE", "16384"))
    result = dict(max_worker_peak_rss_mb=maximum, projected_pool_with_headroom_mb=projected,
                  allocated_memory_mb=allocated, workers=workers, multiplier=1.25, coordinator_headroom_mb=2048)
    if projected > allocated:
        raise ValueError("Measured MNIST pilot peak RSS with worker/headroom exceeds allocated memory")
    return result


def pipeline(main_root, output, main_revision="f602987", workers=16, threads=2):
    from .batch_campaign import validate_cpu_budget
    if (workers, threads) != (16, 2):
        raise ValueError("This fixed control allocation uses sixteen workers and two threads each")
    allocated = validate_cpu_budget(workers, threads)
    verification = verify_sources(main_revision)
    execution, main_revision = verification["execution_revision"], verification["main_revision"]
    if os.environ.get("JOURNAL_GIT_SHA", execution) != execution:
        raise ValueError("Submitted MNIST control revision differs from the current checkout")
    main_root, output = Path(main_root).resolve(), Path(output).resolve()
    if main_root == output or main_root in output.parents or output in main_root.parents:
        raise ValueError("Main campaign and MNIST control outputs must be separate non-nested roots")
    if output.is_symlink():
        raise ValueError("The owned MNIST control output must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / "control_status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        identity = dict(git_sha=execution, main_revision=main_revision, main_root=str(main_root),
                        output=str(output), workers=workers, threads=threads)
        if previous and any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume requires the original MNIST control revision, roots and execution settings")
        state = {**identity, "source_verification": verification, "allocated_cpus": allocated,
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": previous.get("started_at", now()),
                 "status": "running", "stages": dict(previous.get("stages", {})),
                 "seed_selection": "All fixed screening and fresh final seeds; no favorable seed selection",
                 "selection": "Complete original16 plus fixed-positive-scale16 per method/regime; validation only"}
        save_json(status_path, state)
        environment = os.environ.copy()
        environment.update(JOURNAL_GIT_SHA=execution, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        try:
            original_manifest, original_results, capture = capture_original(main_root, output, main_revision)
            state["original_capture"] = capture
            manifest_root = output / "manifests"
            manifest_root.mkdir(exist_ok=True)
            screen = manifest_root / "screen.jsonl"
            build_screen(original_manifest, screen)
            write_pilot(screen, manifest_root / "pilot.jsonl")
            manifests = dict(pilot=manifest_root / "pilot.jsonl", screen=screen)
            for stage in STAGES:
                if stage == "confirm":
                    state["current_stage"] = "validation_selection"
                    save_json(status_path, state)
                    protocol = build_selected(original_manifest, original_results, screen,
                                              output / "screen" / "training_results.json", output / "selected")
                    state["validation_selection"] = protocol
                    manifests.update({name: output / "selected" / f"{name}.jsonl" for name in STAGES[2:]})
                manifest = manifests[stage]
                manifest_hash = fingerprint(manifest)["sha256"]
                if stage in state["stages"] and state["stages"][stage]["manifest_sha256"] != manifest_hash:
                    raise ValueError("A recorded immutable MNIST stage manifest changed")
                if verify_sources(main_revision) != verification:
                    raise ValueError("MNIST control source changed during execution")
                state["current_stage"] = stage
                completed = state["stages"].get(stage, {}).get("status") == "complete"
                state["stages"][stage] = dict(status="running", manifest=str(manifest),
                    manifest_sha256=manifest_hash, tasks=COUNTS[stage], started_at=now())
                save_json(status_path, state)
                started = time.monotonic()
                if not completed:
                    stage_environment = {**environment, "JOURNAL_MANIFEST_SHA256": manifest_hash}
                    arguments = ["journal.campaign", "--manifest", str(manifest), "--output-root", str(output / stage),
                                 "--workers", "16", "--threads", "2", "--device", "cpu", "--resume",
                                 "--evaluate-top", "0" if stage in {"pilot", "screen"} else "1",
                                 "--eval-draws", "20" if stage == "confirm" else "10"]
                    if stage not in {"pilot", "screen"}:
                        arguments.append("--full-risks")
                    command(arguments, output / f"{stage}.log", stage_environment)
                rows = validate_stage(output / stage, manifest, execution, stage)
                if fingerprint(manifest)["sha256"] != manifest_hash or verify_sources(main_revision) != verification:
                    raise ValueError("Manifest or source changed during a MNIST stage")
                if stage == "pilot":
                    state["pilot_memory"] = memory_projection(rows, workers)
                state["stages"][stage].update(status="complete", completed_at=now(), seconds=time.monotonic()-started)
                save_json(status_path, state)
            state.update(status="complete", current_stage=None, completed_at=now(),
                         bounds="Deferred to the joint final main/composite confidence family; none computed here")
            save_json(status_path, state)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at=now())
            save_json(status_path, state)
            raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--main-revision", default="f602987")
    parser.add_argument("--output", type=Path, default=Path("results/journal_bayesian_mnist_scale/20261117"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(pipeline(args.main_root, args.output, args.main_revision, args.workers, args.threads), indent=2))


if __name__ == "__main__":
    main()
