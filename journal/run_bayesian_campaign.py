"""Run the all-Bayesian pilot, validation search, confirmation, and dense sweeps.

This coordinator is intended for one CPU Slurm allocation in an isolated,
committed checkout. It never selects a seed or configuration using test results.
The pipeline records failures and can resume complete, matching checkpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .campaign import save_json


STAGES = ("pilot", "screen", "confirm", "m_sweep", "p_train_sweep")
ACCURACY_TOLERANCE = 1e-12


def now():
    return datetime.now(timezone.utc).isoformat()


def run(command, log, environment):
    print(json.dumps(dict(command=command, log=str(log), started_at=now())), flush=True)
    with Path(log).open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


def install_immutable_manifest(candidate, destination):
    """Compare a regenerated manifest/plan before retaining their first bytes."""
    pairs = ((Path(candidate), Path(destination)),
             (Path(candidate).with_suffix(".plan.json"), Path(destination).with_suffix(".plan.json")))
    for source, target in pairs:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Immutable campaign manifest or plan changed: {target}")
    for source, target in pairs:
        if not target.exists():
            source.replace(target)
        else:
            source.unlink()


def freeze_validation_results(source, destination):
    """Retain the complete selection input despite resume order/cache flags."""
    records = json.loads(Path(source).read_text())
    canonical = sorted(({key: value for key, value in row.items() if key != "resumed"}
                        for row in records), key=lambda row: row["task_id"])
    target = Path(destination)
    if target.exists():
        if json.loads(target.read_text()) != canonical:
            raise ValueError("Completed validation selection results changed on resume")
    else:
        save_json(target, canonical)


def ranking_outcome(root, manifest=None):
    """Describe all fixed comparisons without switching methods for each seed."""
    rows = json.loads((Path(root) / "test_results.json").read_text())
    labels = {f"{method}-{regime}" for method in ("ERM", "L2", "T1", "T2")
              for regime in ("Matched", "Perfect")}
    if not rows:
        raise ValueError("Ranking requires a nonempty complete confirmation cohort")
    identifiers = [row["task_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate confirmation task IDs")
    if manifest is not None:
        entries = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        expected = {entry["task_id"]: entry for entry in entries}
        if len(expected) != len(entries) or set(expected) != set(identifiers):
            raise ValueError("Ranking requires exact declared confirmation task coverage")
        for row in rows:
            entry = expected[row["task_id"]]
            if (any(row[key] != entry[key] for key in ("group", "method_label"))
                    or row["seed"] != entry["config"]["seed"]):
                raise ValueError("Ranking result differs from its declared confirmation identity")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        accuracy = row.get("test_accuracy")
        if (row["test_status"] != "complete" or isinstance(accuracy, bool)
                or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1):
            raise ValueError("Ranking requires complete finite test results")
        groups[row["group"]][row["method_label"]].append(row)
    output = []
    for group, methods in sorted(groups.items()):
        if set(methods) != labels:
            raise ValueError("Main ranking requires exactly ERM/L2/T1/T2 under matched and perfect training")
        seed_sets = []
        for runs in methods.values():
            seeds = [row["seed"] for row in runs]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("Duplicate or empty confirmation seed set")
            seed_sets.append(set(seeds))
        if any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError("All methods require the same complete paired confirmation seeds")
        means = {method: statistics.fmean(row["test_accuracy"] for row in runs)
                 for method, runs in methods.items()}
        eligible = {key: value for key, value in means.items() if key in {"T1-Matched", "T2-Matched"}}
        if len(eligible) != 2 or len(means) != 8:
            raise ValueError("Main ranking requires ERM/L2/T1/T2 under matched and perfect training")
        candidate = max(eligible, key=eligible.get)
        competitors = {key: value for key, value in means.items() if key != candidate}
        strongest = max(competitors, key=competitors.get)
        per_seed = []
        for seed in sorted(seed_sets[0]):
            scores = {method: next(row["test_accuracy"] for row in runs if row["seed"] == seed)
                      for method, runs in methods.items()}
            margin = scores[candidate] - max(value for method, value in scores.items() if method != candidate)
            per_seed.append(dict(seed=seed, fixed_matched_candidate=candidate, margin=margin,
                                 target_met=margin > ACCURACY_TOLERANCE))
        output.append(dict(group=group, means=means, best_matched_proposed_by_observed_mean=candidate,
                           strongest_other_method=strongest,
                           mean_accuracy_margin=means[candidate] - means[strongest],
                           target_met=means[candidate] - means[strongest] > ACCURACY_TOLERANCE,
                           fixed_candidate_per_seed=per_seed,
                           target_met_for_every_seed=all(item["target_met"] for item in per_seed),
                           seeds={key: sorted(row["seed"] for row in runs) for key, runs in methods.items()}))
    return dict(conditions=output, target_met_in_every_condition=all(item["target_met"] for item in output),
                interpretation="Descriptive comparison of all prespecified confirmation seeds. Hyperparameters used validation only. No guarantee of a proposed-method win and no favorable seeds omitted.")


def pipeline(output, workers, threads, confirm_draws=20, sweep_draws=10):
    from .batch_campaign import validate_cpu_budget
    allocated = validate_cpu_budget(workers, threads)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_root = output / "manifests"
    manifest_root.mkdir(exist_ok=True)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected_git_sha = os.environ.get("JOURNAL_GIT_SHA")
    if expected_git_sha and expected_git_sha != git_sha:
        raise ValueError("Checkout revision changed after submission")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Run from a clean committed checkout; use a separate checkout for experiment execution")
    status_path = output / "pipeline_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous and previous["git_sha"] != git_sha:
        raise ValueError("The output belongs to another code revision; use a fresh output root")
    settings = dict(workers=workers, threads=threads, confirm_draws=confirm_draws, sweep_draws=sweep_draws)
    if previous and any(previous.get(key) != value for key, value in settings.items()):
        raise ValueError("Resume requires the original worker, thread and MC-draw settings")
    state = dict(git_sha=git_sha, output=str(output), workers=workers, threads=threads,
                 allocated_cpus=allocated, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                 started_at=previous["started_at"] if previous else now(), status="running",
                 stages=dict(previous.get("stages", {})) if previous else {},
                 confirm_draws=confirm_draws, sweep_draws=sweep_draws,
                 seed_selection="None; all preregistered paired seeds retained",
                 test_use="Only final confirmation and sweeps, after validation-only selection")
    save_json(status_path, state)
    environment = os.environ.copy()
    environment["JOURNAL_GIT_SHA"] = git_sha
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    environment["OPENBLAS_NUM_THREADS"] = str(threads)
    manifests = {}
    try:
        for stage in STAGES:
            started = time.monotonic()
            manifest = manifest_root / (stage + ".jsonl")
            manifests[stage] = manifest
            if stage in state["stages"]:
                recorded_hash = state["stages"][stage]["manifest_sha256"]
                if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != recorded_hash:
                    raise ValueError(f"Recorded immutable manifest changed: {manifest}")
            candidate_manifest = manifest_root / (stage + ".candidate.jsonl")
            build = [sys.executable, "-m", "journal.build_bayesian_campaign", "--stage", stage,
                     "--output", str(candidate_manifest)]
            if stage not in {"pilot", "screen"}:
                build.extend(["--from-manifest", str(manifests["screen"]), "--from-results",
                              str(output / "validation_selection_results.json")])
            # Generated manifests live under ignored results/ and are immutable
            # once handed to the stage. Store their hash before any training.
            run(build, output / (stage + "_build.log"), environment)
            install_immutable_manifest(candidate_manifest, manifest)
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            environment["JOURNAL_MANIFEST_SHA256"] = manifest_hash
            plan = json.loads(manifest.with_suffix(".plan.json").read_text())
            state["current_stage"] = stage
            state["stages"][stage] = dict(status="running", started_at=now(), tasks=plan["tasks"],
                                         manifest=str(manifest), manifest_sha256=manifest_hash)
            save_json(status_path, state)
            command = [sys.executable, "-u", "-m", "journal.campaign", "--manifest", str(manifest),
                       "--output-root", str(output / stage), "--workers", str(workers),
                       "--threads", str(threads), "--resume", "--evaluate-top",
                       "0" if stage in {"pilot", "screen"} else "1",
                       "--eval-draws", str(confirm_draws if stage == "confirm" else sweep_draws)]
            if stage not in {"pilot", "screen"}:
                command.append("--full-risks")
            run(command, output / (stage + ".log"), environment)
            summary = json.loads((output / stage / "summary.json").read_text())
            if (summary["completed_trials"] != plan["tasks"] or summary["failed_trials"]
                    or summary.get("failed_evaluations")):
                raise RuntimeError(f"Incomplete stage {stage}")
            state["stages"][stage].update(status="complete", completed_at=now(),
                                          seconds=time.monotonic() - started)
            if stage == "pilot":
                rows = json.loads((output / stage / "training_results.json").read_text())
                state["pilot_resources"] = dict(max_worker_peak_rss_mb=max(row["peak_rss_mb"] for row in rows),
                    max_trial_seconds=max(row["seconds"] for row in rows),
                    scientific_result=False, completed_full_shape_methods=len(rows))
                memory_mb = os.environ.get("SLURM_MEM_PER_NODE")
                if memory_mb:
                    projected_mb = 2048 + 1.25 * workers * state["pilot_resources"]["max_worker_peak_rss_mb"]
                    state["pilot_resources"].update(projected_pool_rss_with_headroom_mb=projected_mb,
                                                   allocated_memory_mb=int(memory_mb))
                    if projected_mb > int(memory_mb):
                        raise RuntimeError("Pilot peak RSS with pool/headroom exceeds allocated memory; reduce workers or resubmit with measured memory")
            if stage == "screen":
                freeze_validation_results(output / stage / "training_results.json",
                                          output / "validation_selection_results.json")
            save_json(status_path, state)
            if stage == "confirm":
                save_json(output / "accuracy_outcome.json", ranking_outcome(output / stage, manifest=manifest))
            if stage not in {"pilot", "screen"}:
                run([sys.executable, "-m", "journal.plot_bayesian_campaign", "--source", str(output / stage),
                     str(manifest), "--output", str(output / "figures" / stage)],
                    output / (stage + "_figures.log"), environment)
        state["current_stage"] = "bounds"
        save_json(status_path, state)
        command = [sys.executable, "-m", "journal.bayesian_bounds", "--output", str(output / "bounds")]
        for stage in STAGES[2:]:
            command.extend(["--campaign", str(manifests[stage]), str(output / stage)])
        run(command, output / "bounds.log", environment)
        state.update(status="complete", current_stage=None, completed_at=now())
        save_json(status_path, state)
    except BaseException as error:
        state.update(status="failed", failed_at=now(), error=f"{type(error).__name__}: {error}")
        save_json(status_path, state)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/journal_bayesian/20261117_refined"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--confirm-draws", type=int, default=20)
    parser.add_argument("--sweep-draws", type=int, default=10)
    args = parser.parse_args()
    if min(args.workers, args.threads, args.confirm_draws, args.sweep_draws) < 1:
        parser.error("Worker, thread and draw counts must be positive")
    result = pipeline(args.output, args.workers, args.threads, args.confirm_draws, args.sweep_draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
