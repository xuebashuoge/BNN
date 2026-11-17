"""Run the fixed p_train sweep separately and atomically publish its whole cache.

Only the owned external output is written during computation. Linux
renameat2(RENAME_NOREPLACE) is the sole publication primitive: an existing main
stage, including an empty directory, is never replaced. No unsafe fallback is
provided. The running main controller, status and manifests are never edited.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from .campaign import save_json


SOURCE_FILES = ("journal/run_experiment.py", "journal/theorem1.py",
                "journal/cifar_models.py", "journal/cifar_search_models.py",
                "journal/campaign.py", "journal/build_bayesian_campaign.py",
                "journal/build_campaign.py")
TOTAL = 672
GLOBAL_REPORTS = ("training_results.json", "test_results.json", "summary.json",
                  "campaign_snapshot.json", "all_trials.csv")


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def rename_noreplace(source, destination):
    """Linux atomic absent-only directory rename; fail closed on unsupported FS."""
    if platform.system() != "Linux":
        raise OSError(errno.ENOTSUP, "Atomic no-replace publication requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable; no fallback is permitted")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def probe_publication(parent):
    """Prove whole-directory success and empty/nonempty destination protection."""
    parent = Path(parent).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Publication probe parent must be an existing directory")
    with tempfile.TemporaryDirectory(prefix=".parallel-p-publication-probe-", dir=parent) as temporary:
        base = Path(temporary)
        source, target = base / "source", base / "published"
        source.mkdir()
        (source / "marker").write_bytes(b"whole-stage-probe")
        rename_noreplace(source, target)
        if source.exists() or (target / "marker").read_bytes() != b"whole-stage-probe":
            raise RuntimeError("Atomic publication probe did not move the whole directory")
        for nonempty in (False, True):
            source = base / ("source-nonempty" if nonempty else "source-empty")
            target = base / ("existing-nonempty" if nonempty else "existing-empty")
            source.mkdir()
            (source / "marker").write_bytes(b"preserve-source")
            target.mkdir()
            if nonempty:
                (target / "existing").write_bytes(b"preserve-target")
            try:
                rename_noreplace(source, target)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise RuntimeError("Filesystem did not provide the required EEXIST no-replace behavior") from error
            else:
                raise RuntimeError("Unsafe rename behavior: an existing directory was replaced")
            if (source / "marker").read_bytes() != b"preserve-source":
                raise RuntimeError("No-replace probe modified its source")
            if (nonempty and (target / "existing").read_bytes() != b"preserve-target") or (
                    not nonempty and any(target.iterdir())):
                raise RuntimeError("No-replace probe modified the existing target")
    return dict(parent=str(parent), device=parent.stat().st_dev, successful_whole_stage_rename=True,
                existing_empty_destination="EEXIST; both directories preserved",
                existing_nonempty_destination="EEXIST; both directories preserved",
                primitive="Linux libc renameat2(RENAME_NOREPLACE); no fallback")


def verify_source_blobs(main_revision, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()

    source = git("rev-parse", "--verify", "--end-of-options", main_revision + "^{commit}")
    current = git("rev-parse", "HEAD^{commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use a clean committed checkout for the external sweep")
    files = {}
    for filename in SOURCE_FILES:
        expected = git("rev-parse", source + ":" + filename)
        committed = git("rev-parse", current + ":" + filename)
        working = git("hash-object", "--", filename)
        if expected != committed or expected != working:
            raise ValueError(f"External sweep source differs from the running main campaign: {filename}")
        files[filename] = dict(main_blob=expected, external_blob=committed, working_blob=working)
    coordinator = "journal/run_parallel_p_train_sweep.py"
    coordinator_blob = git("rev-parse", current + ":" + coordinator)
    if git("hash-object", "--", coordinator) != coordinator_blob:
        raise ValueError("External coordinator differs from its committed source")
    return dict(main_revision=source, external_revision=current, files=files,
                coordinator=dict(path=coordinator, committed_blob=coordinator_blob))


def source_inputs(main_root, main_revision):
    main_root = Path(main_root).resolve(strict=True)
    state_path = main_root / "pipeline_status.json"
    state = json.loads(state_path.read_text())
    if state.get("git_sha") != main_revision:
        raise ValueError("Main campaign status identifies a different committed source")
    screen = main_root / "manifests" / "screen.jsonl"
    frozen = main_root / "validation_selection_results.json"
    if any(not path.is_file() or path.is_symlink() for path in (screen, frozen)):
        raise ValueError("Require the complete frozen main screening manifest and validation snapshot")
    files = {str(path): fingerprint(path) for path in (screen, frozen)}
    if state.get("stages", {}).get("screen", {}).get("manifest_sha256") != files[str(screen)]["sha256"]:
        raise ValueError("Main screening manifest differs from its recorded immutable hash")
    return screen, frozen, files, state


def command(arguments, log, environment):
    with Path(log).open("a") as stream:
        subprocess.run([sys.executable, "-u", "-m", *arguments], stdout=stream,
                       stderr=subprocess.STDOUT, env=environment, check=True)


def validate_stage(stage, manifest, execution_revision):
    """Validate every checkpoint, full risk report and exact declared task ID."""
    from .bayesian_bounds import collect_trials, observation
    from .plot_bayesian_campaign import Evidence
    from .stage_bayesian_checkpoints import validate_checkpoint
    stage, manifest = Path(stage), Path(manifest)
    if stage.is_symlink() or any(path.is_symlink() for path in stage.rglob("*")):
        raise ValueError("A portable completed stage cannot contain symbolic links")
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if (len(entries) != TOTAL or len({entry["task_id"] for entry in entries}) != TOTAL
            or any(entry.get("stage") != "bayes_p_train_sweep"
                   or entry.get("experiment") != "p_train_sweep" for entry in entries)):
        raise ValueError("Require the complete prescribed 672-case p_train sweep")
    summary = json.loads((stage / "summary.json").read_text())
    if (summary.get("training_trials") != TOTAL or summary.get("completed_trials") != TOTAL
            or summary.get("failed_trials") or summary.get("failed_evaluations")):
        raise ValueError("External p_train sweep has incomplete training or evaluation")
    training = json.loads((stage / "training_results.json").read_text())
    expected = {entry["task_id"]: entry for entry in entries}
    if len(training) != TOTAL or {row["task_id"] for row in training} != set(expected):
        raise ValueError("External training rows do not cover exactly the prescribed tasks")
    from .run_experiment import normalize_config
    for row in training:
        entry = expected[row["task_id"]]
        config = normalize_config(row["config"])
        declared = normalize_config(entry["config"])
        if (row.get("status") != "complete" or any(row.get(key) != entry.get(key) for key in
                ("group", "method_label", "variant_id", "family", "stage"))
                or row.get("seed") != declared["seed"]
                or any(config[key] != declared[key] for key in config if key not in {"device", "threads"})
                or config["device"] != "cpu" or config["threads"] != 2):
            raise ValueError("External training row identity/configuration differs from the declared sweep")
    # Strictly require all risk components, draw-wise means and paired design.
    evidence = Evidence(stage, manifest)
    if len(evidence.rows) != TOTAL or evidence.excluded_task_ids:
        raise ValueError("External risk reports do not have exact full sweep coverage")
    records = []
    for entry, row, root in collect_trials([(manifest, stage)]):
        checkpoint = stage / entry["task_id"]
        completed, reason = validate_checkpoint(checkpoint, entry, execution_revision)
        if completed is None:
            raise ValueError(f"Invalid external checkpoint {entry['task_id']}: {reason}")
        report_path = checkpoint / "test_full" / "evaluation.json"
        report = json.loads(report_path.read_text())
        config = report["evaluation_config"]
        if (config["eval_draws"] != 10 or config["device"] != "cpu" or config["threads"] != 2
                or config["eval_seed"] != row["config"]["eval_seed"] + 1000003):
            raise ValueError("External sweep must use the main controller's exact ten-draw evaluation stream")
        # This validates source hashes, bank/split integrity and risk units.
        # Its provisional numerical RHS is discarded: main later recomputes
        # bounds with the confidence allocation of the complete final family.
        checked = observation(entry, row, root, TOTAL, .025)
        records.append(dict(task_id=entry["task_id"], original_checkpoint=str(checkpoint.resolve()),
                            original_evaluation=str(report_path.resolve()),
                            source_fingerprints=checked["source_fingerprints"]))
    return dict(tasks=TOTAL, evaluation_draws=10, manifest_fingerprint=fingerprint(manifest),
                complete_checkpoint_and_full_risk_coverage=True,
                source_records=records, files=evidence.files)


def recheck_validated_sources(manifest, validation):
    if (validation.get("tasks") != TOTAL or validation.get("evaluation_draws") != 10
            or validation.get("complete_checkpoint_and_full_risk_coverage") is not True
            or validation.get("manifest_fingerprint") != fingerprint(manifest)):
        raise ValueError("Publication lacks complete unchanged manifest validation")
    records = validation["source_records"]
    declared_ids = {json.loads(line)["task_id"] for line in Path(manifest).read_text().splitlines() if line.strip()}
    if (len(records) != TOTAL or len({record["task_id"] for record in records}) != TOTAL
            or {record["task_id"] for record in records} != declared_ids):
        raise ValueError("Publication validation must cover every distinct checkpoint")
    for record in records:
        paths = {"metadata": Path(record["original_checkpoint"]) / "metadata.json",
                 "weights": Path(record["original_checkpoint"]) / "weights.pt",
                 "evaluation": Path(record["original_evaluation"])}
        if {name: fingerprint(path) for name, path in paths.items()} != record["source_fingerprints"]:
            raise ValueError("A validated checkpoint or full risk report changed before publication")
    if any(fingerprint(path)["sha256"] != checksum for path, checksum in validation["files"].items()):
        raise ValueError("A validated manifest or report changed before publication")


def archive_publication_sources(output, stage, manifest, screen, frozen):
    """Retain original global rows before main can rewrite their absolute paths."""
    archive = Path(output) / "publication_snapshot"
    if archive.is_symlink():
        raise ValueError("Publication archive must be a real owned directory")
    archive.mkdir(exist_ok=True)
    files = {name: Path(stage) / name for name in GLOBAL_REPORTS}
    files.update(manifest=Path(manifest), manifest_plan=Path(manifest).with_suffix(".plan.json"),
                 source_screen=Path(screen), source_validation=Path(frozen))
    retained = {}
    for name, source in files.items():
        target = archive / (name if name in GLOBAL_REPORTS else name + source.suffix)
        if source.is_symlink() or target.is_symlink():
            raise ValueError("Publication archives cannot read or write symbolic links")
        original = fingerprint(source)
        if target.exists():
            if fingerprint(target) != original:
                raise ValueError("Publication archive differs from the completed immutable stage")
        else:
            shutil.copyfile(source, target)
        if fingerprint(source) != original or fingerprint(target) != original:
            raise ValueError("A source changed while publication evidence was retained")
        retained[name] = dict(original=str(source.resolve()), archived=str(target.resolve()), **original)
    return dict(directory=str(archive.resolve()), files=retained)


def publication_committed(audit):
    """Recognize a completed atomic move without touching an active main cache."""
    source, target = Path(audit["source"]), Path(audit["target"])
    try:
        stat = target.stat()
        return (not source.exists() and not source.is_symlink() and not target.is_symlink()
                and target.is_dir() and audit.get("stage_identity") == dict(device=stat.st_dev, inode=stat.st_ino))
    except OSError:
        return False


def recover_publication(output, main_root, verification):
    """Reconcile an interrupted post-rename audit, using only external writes."""
    output = Path(output)
    audit_path = output / "publication_audit.json"
    if not audit_path.is_file():
        return None
    audit = json.loads(audit_path.read_text())
    if (Path(audit["target"]) != Path(main_root) / "p_train_sweep"
            or audit.get("training_source_verification") != verification):
        raise ValueError("Existing publication audit belongs to another immutable run")
    if not publication_committed(audit):
        if audit.get("status") == "published":
            raise ValueError("Previously published cache identity no longer matches its audit")
        return None
    audit.update(status="published", recovered_at=now())
    save_json(audit_path, audit)
    status_path = output / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    state.update(status="published", phase=None, target=audit["target"], completed_tasks=TOTAL,
                 publication_audit=str(audit_path), recovered_at=now())
    save_json(status_path, state)
    return state


def publish(stage, main_root, manifest, inputs, source_verification, validation, output):
    """Advisory stage guard plus atomic absent-only publication; never overwrite."""
    if Path(stage).is_symlink():
        raise ValueError("The completed stage must be a real directory, not a symbolic link")
    stage, main_root, output = Path(stage).resolve(strict=True), Path(main_root).resolve(strict=True), Path(output).resolve()
    target = main_root / "p_train_sweep"
    if stage.is_symlink() or not stage.is_dir() or stage.stat().st_dev != main_root.stat().st_dev:
        raise ValueError("Publication requires a real whole-stage directory on the same filesystem")
    screen, frozen, current_inputs, state = source_inputs(main_root, source_verification["main_revision"])
    if current_inputs != inputs:
        raise ValueError("Main validation selection inputs changed before publication")
    if (state.get("status") != "running" or state.get("current_stage") not in {"confirm", "m_sweep"}
            or "p_train_sweep" in state.get("stages", {})):
        raise ValueError("Main is outside the advisory confirmation/m_sweep publication window")
    future = main_root / "manifests" / "p_train_sweep.jsonl"
    if future.exists() and future.read_bytes() != Path(manifest).read_bytes():
        raise ValueError("Main's sweep manifest differs from the completed external manifest")
    # Preserve the original JSON/CSV path strings before main can resume.
    archive = archive_publication_sources(output, stage, manifest, screen, frozen)
    if any(path.is_symlink() for path in stage.rglob("*")):
        raise ValueError("The completed stage gained a symbolic link before publication")
    recheck_validated_sources(manifest, validation)
    audit_path = output / "publication_audit.json"
    audit = dict(status="ready_to_publish", created_at=now(), source=str(stage), target=str(target),
                 stage_identity=dict(device=stage.stat().st_dev, inode=stage.stat().st_ino),
                 manifest=str(Path(manifest).resolve()), manifest_fingerprint=fingerprint(manifest),
                 original_selection_inputs=inputs, training_source_verification=source_verification,
                 complete_validation=validation, original_global_sources=archive,
                 main_status_observed=state, resource_job_id=os.environ.get("SLURM_JOB_ID"),
                 task_path_map=[dict(task_id=record["task_id"], original=record["original_checkpoint"],
                                     published=str(target / record["task_id"]))
                                for record in validation["source_records"]],
                 publication="Linux renameat2(RENAME_NOREPLACE); existing targets including empty directories are refused",
                 stage_guard="Advisory only; the atomic no-replace rename provides race protection")
    save_json(audit_path, audit)
    try:
        rename_noreplace(stage, target)
    except BaseException as error:
        # A signal may arrive immediately after the kernel committed the move.
        committed = publication_committed(audit)
        audit.update(status="published" if committed else "not_published",
                     error=f"{type(error).__name__}: {error}", ended_at=now())
        save_json(audit_path, audit)
        raise
    audit.update(status="published", published_at=now())
    save_json(audit_path, audit)
    return audit


def _run_locked(main_root, main_revision, output, workers=16, threads=2):
    from .batch_campaign import validate_cpu_budget
    if threads != 2:
        raise ValueError("Use the main controller's exact two-thread checkpoint/evaluation settings")
    allocated = validate_cpu_budget(workers, threads)
    verification = verify_source_blobs(main_revision)
    if os.environ.get("JOURNAL_GIT_SHA") and os.environ["JOURNAL_GIT_SHA"] != verification["external_revision"]:
        raise ValueError("External checkout changed after job submission")
    main_root = Path(main_root).resolve(strict=True)
    output = Path(output).resolve()
    if output == main_root or output in main_root.parents or main_root in output.parents:
        raise ValueError("External output and active main output must be separate non-nested roots")
    recovered = recover_publication(output, main_root, verification)
    if recovered is not None:
        return recovered
    screen, frozen, inputs, main_state = source_inputs(main_root, verification["main_revision"])
    if main_state.get("status") != "running" or main_state.get("current_stage") not in {"screen", "confirm", "m_sweep"}:
        raise ValueError("Main already reached or passed the p_train stage")
    destination = main_root / "p_train_sweep"
    if destination.exists() or destination.is_symlink():
        raise ValueError("Main p_train destination already exists; external computation is refused")
    output.mkdir(parents=True, exist_ok=True)
    if output.stat().st_dev != main_root.stat().st_dev or main_root.parent.stat().st_dev != main_root.stat().st_dev:
        raise ValueError("External stage, main destination and probe parent must share one filesystem")
    with nullcontext():
        probe = probe_publication(main_root.parent)
        memory = main_state.get("pilot_resources", {}).get("max_worker_peak_rss_mb")
        if not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
            raise ValueError("Require the main campaign's completed resource pilot RSS measurement")
        projected = 2048 + 1.25 * workers * memory
        allocated_memory = os.environ.get("SLURM_MEM_PER_NODE")
        if allocated_memory and projected > int(allocated_memory):
            raise RuntimeError("Measured main pilot pool with headroom exceeds this allocation's memory")
        protocol = dict(main_root=str(main_root), main_revision=verification["main_revision"],
                        external_revision=verification["external_revision"], selection_inputs=inputs,
                        workers=workers, threads=threads, evaluation_draws=10,
                        stage="p_train_sweep", tasks=TOTAL)
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("External output belongs to a different immutable run protocol")
        save_json(protocol_path, protocol)
        status_path = output / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "published":
            return json.loads(status_path.read_text())
        state = dict(status="running", phase="build", started_at=now(), protocol=protocol,
                     publication_probe=probe, allocated_cpus=allocated, projected_memory_mb=projected,
                     slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        save_json(status_path, state)
        environment = os.environ.copy()
        environment.update(JOURNAL_GIT_SHA=verification["external_revision"], OMP_NUM_THREADS=str(threads),
                           MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        manifest = output / "p_train_sweep.jsonl"
        candidate = output / "p_train_sweep.candidate.jsonl"
        stage = output / "stage"
        try:
            command(["journal.build_bayesian_campaign", "--stage", "p_train_sweep", "--from-manifest", str(screen),
                     "--from-results", str(frozen), "--output", str(candidate)], output / "build.log", environment)
            from .run_bayesian_campaign import install_immutable_manifest
            install_immutable_manifest(candidate, manifest)
            if inputs != source_inputs(main_root, verification["main_revision"])[2]:
                raise ValueError("Frozen main selection inputs changed while building the sweep")
            environment["JOURNAL_MANIFEST_SHA256"] = fingerprint(manifest)["sha256"]
            state.update(phase="train_and_evaluate", manifest=str(manifest), manifest_fingerprint=fingerprint(manifest))
            save_json(status_path, state)
            validation = None
            if (stage / "test_results.json").is_file():
                try:
                    validation = validate_stage(stage, manifest, verification["external_revision"])
                except (OSError, ValueError, KeyError):
                    if (output / "publication_snapshot").exists() or (output / "publication_audit.json").exists():
                        raise ValueError("The previously frozen completed external stage changed; repair is refused")
            state["resumed_complete_stage"] = validation is not None
            if validation is None:
                command(["journal.campaign", "--manifest", str(manifest), "--output-root", str(stage),
                         "--workers", str(workers), "--threads", str(threads), "--evaluate-top", "1",
                         "--eval-draws", "10", "--full-risks", "--resume"], output / "campaign.log", environment)
            state["phase"] = "validate"
            save_json(status_path, state)
            if validation is None:
                validation = validate_stage(stage, manifest, verification["external_revision"])
            if verify_source_blobs(main_revision) != verification:
                raise ValueError("Execution source changed before publication")
            state["phase"] = "publish"
            save_json(status_path, state)
            audit = publish(stage, main_root, manifest, inputs, verification, validation, output)
            state.update(status="published", phase=None, completed_at=now(), publication_audit=str(output / "publication_audit.json"),
                         target=audit["target"], completed_tasks=TOTAL)
            save_json(status_path, state)
        except BaseException as error:
            audit_path = output / "publication_audit.json"
            committed = (audit_path.is_file() and publication_committed(json.loads(audit_path.read_text())))
            state.update(status="published_with_postprocessing_error" if committed else "failed_unpublished",
                         error=f"{type(error).__name__}: {error}", ended_at=now())
            save_json(status_path, state)
            raise
    return state


def run(main_root, main_revision, output, workers=16, threads=2):
    main_root, output = Path(main_root).resolve(strict=True), Path(output).resolve()
    if output == main_root or output in main_root.parents or main_root in output.parents:
        raise ValueError("External output and active main output must be separate non-nested roots")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".parallel_sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(main_root, main_revision, output, workers, threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-publication", type=Path, metavar="TARGET_PARENT",
                        help="Probe atomic publication on this filesystem and exit without computation")
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--main-revision", default="f602987")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.probe_publication is not None:
        print(json.dumps(probe_publication(args.probe_publication), indent=2))
        return
    if args.main_root is None or args.output is None:
        parser.error("Run requires --main-root and --output; probe mode requires only --probe-publication")
    print(json.dumps(run(args.main_root, args.main_revision, args.output, args.workers, args.threads), indent=2))


if __name__ == "__main__":
    main()
