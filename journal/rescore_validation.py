"""Rescore the complete 32-case CIFAR search on validation data only.

Run after ``campaign --resume --evaluate-top 0`` has aggregated all training
rows. This command has no training fallback and never reads test reports.
It preserves the original training rows, histories, metadata and checkpoints.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO))

from journal.campaign import completed_trial, save_json, select_variants, validate_entries
from journal.run_task import read_manifest, verify_submission_snapshot


EXPECTED_TRIALS = 32
METHODS = {"ERM-Matched": "erm", "L2-Matched": "l2",
           "T1-Matched": "proposed", "T2-Matched": "joint"}
SEED_OFFSET = 2_000_003
METRICS = ("accuracy", "cross_entropy", "bounded_loss", "feature_power_per_coordinate")


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checkpoint_fingerprints(checkpoint):
    return {name: fingerprint(checkpoint / name)
            for name in ("weights.pt", "metadata.json", "history.json")}


def number(value, name, upper=None):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0
            or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def preflight(entries, output_root):
    """Reject incomplete searches before creating any evaluation outputs."""
    from journal.run_experiment import normalize_config

    if len(entries) != EXPECTED_TRIALS:
        raise ValueError(f"Expected all {EXPECTED_TRIALS} search entries")
    validate_entries(entries)
    if len({entry["group"] for entry in entries}) != 1:
        raise ValueError("Rescore one comparison group at a time")
    if {entry["method_label"] for entry in entries} != set(METHODS):
        raise ValueError("Expected ERM/L2/T1/T2 matched search methods")
    pairs, variant_configs = {}, {}
    for entry in entries:
        config = normalize_config(entry["config"])
        if (config["dataset"] != "cifar10" or config["train_size"] != 45000
                or config["val_size"] != 5000 or config["epochs"] != 100
                or config["device"] != "cpu"
                or config["training_loss"] != "bounded_ce_smooth"
                or config["selection_metric"] != "accuracy"
                or config["method"] != METHODS[entry["method_label"]]
                or config["train_channel"] != config["deployment_channel"]):
            raise ValueError(f"Unexpected full-data matched search protocol: {entry['task_id']}")
        key = (entry["group"], entry["method_label"], entry["variant_id"])
        pairs.setdefault(key, []).append(config["seed"])
        fixed = {name: value for name, value in config.items()
                 if name not in {"seed", "channel_seed", "eval_seed", "threads"}}
        if key in variant_configs and fixed != variant_configs[key]:
            raise ValueError("A variant must keep hyperparameters fixed across its paired seeds")
        variant_configs[key] = fixed
    seed_sets = {tuple(sorted(seeds)) for seeds in pairs.values()}
    if len(seed_sets) != 1 or any(len(seeds) != 2 or len(set(seeds)) != 2
                                  for seeds in pairs.values()):
        raise ValueError("Every search variant must retain the same two distinct seeds")

    rows = read_json(output_root / "training_results.json")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TRIALS:
        raise ValueError("training_results.json must contain all 32 completed rows")
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {entry["task_id"] for entry in entries}:
        raise ValueError("Training rows must match the manifest exactly, without duplicates")
    verified, data_splits, banks = [], set(), {}
    for entry in entries:
        row = by_id[entry["task_id"]]
        expected = normalize_config(entry["config"])
        checkpoint = (output_root / entry["task_id"]).resolve()
        if (row.get("status") != "complete"
                or Path(row["checkpoint"]).resolve() != checkpoint
                or normalize_config(row["config"]) != expected
                or row["seed"] != expected["seed"]
                or any(row[key] != entry[key]
                       for key in ("group", "method_label", "family", "variant_id"))):
            raise ValueError(f"Incomplete or mismatched training row: {entry['task_id']}")
        cached = completed_trial(checkpoint, expected)
        if cached is None:
            raise ValueError(f"No complete matching checkpoint: {entry['task_id']}; training is never attempted")
        metadata = read_json(checkpoint / "metadata.json")
        history = read_json(checkpoint / "history.json")
        data_splits.add((metadata["split"]["train_indices_hash"],
                         metadata["split"]["validation_indices_hash"]))
        banks.setdefault(expected["seed"], set()).add(metadata["training_channel_bank_hash"])
        epoch = metadata["selected_epoch"]
        if (not isinstance(epoch, int) or isinstance(epoch, bool)
                or not 1 <= epoch <= expected["epochs"]
                or [record["epoch"] for record in history] != list(range(1, expected["epochs"] + 1))
                or row["selected_epoch"] != epoch):
            raise ValueError(f"Invalid checkpoint/history epoch: {entry['task_id']}")
        for record in history:
            number(record["validation"]["accuracy"], "historical validation accuracy", 1)
            number(record["validation"]["cross_entropy"], "historical validation CE")
        for key in ("validation_accuracy", "validation_cross_entropy"):
            number(row[key], key, 1 if key == "validation_accuracy" else None)
            if abs(row[key] - cached[key]) > 1e-12:
                raise ValueError(f"Original training score disagrees with checkpoint: {entry['task_id']}")
        verified.append(dict(row=copy.deepcopy(row), config=expected, checkpoint=checkpoint,
                             sources=checkpoint_fingerprints(checkpoint)))
    if len(data_splits) != 1 or any(len(values) != 1 for values in banks.values()):
        raise ValueError("Search cases must share data splits and paired channel banks")
    return rows, verified


def validate_report(report, checkpoint, expected, selected_epoch, draws):
    if (report.get("split") != "validation" or report.get("risk_scope") != "deployment"
            or Path(report["checkpoint"]).resolve() != checkpoint
            or report.get("selected_epoch") != selected_epoch
            or report.get("evaluation_config") != expected):
        raise ValueError("Validation report has a mismatched split/checkpoint/configuration")
    heldout = report["heldout_gibbs_mc"]
    if (heldout.get("examples") != expected["val_size"]
            or heldout.get("draws") != draws
            or heldout.get("channel_source") != "deployment"
            or heldout.get("posterior_rule") != "one weight draw fixed across entire loader per replicate"
            or len(heldout["replicate_statistics"]) != draws):
        raise ValueError("Expected full validation data and independent full-loader Gibbs draws")
    from journal.run_experiment import bounded_loss_description
    if heldout.get("bounded_loss_definition") != bounded_loss_description(expected):
        raise ValueError("Validation report uses a different theoretical loss")
    for metric in METRICS:
        upper = 1 if metric in {"accuracy", "bounded_loss"} else None
        number(heldout[metric], metric, upper)
        values = [number(replicate[metric], f"replicate {metric}", upper)
                  for replicate in heldout["replicate_statistics"]]
        if abs(statistics.fmean(values) - heldout[metric]) > 1e-12:
            raise ValueError(f"Validation aggregate disagrees with replicates: {metric}")
    return heldout


def rescore_one(item, draws, threads, manifest_hash):
    row, checkpoint = item["row"], item["checkpoint"]
    output = checkpoint / "validation_rescore"
    overrides = dict(eval_draws=draws, eval_seed=item["config"]["eval_seed"] + SEED_OFFSET,
                     device="cpu", threads=threads)
    expected = {**item["config"], **overrides}
    provenance = dict(manifest_sha256=manifest_hash, checkpoint=str(checkpoint),
                      source_fingerprints=item["sources"], selected_epoch=row["selected_epoch"],
                      split="validation", risk_scope="deployment", overrides=overrides)
    result = {**row, "original_validation_accuracy": row["validation_accuracy"],
              "original_validation_cross_entropy": row["validation_cross_entropy"]}
    try:
        if checkpoint_fingerprints(checkpoint) != item["sources"]:
            raise ValueError("Checkpoint inputs changed after preflight")
        report_path = output / "evaluation.json"
        snapshot_path = output / "rescore_snapshot.json"
        overrides_path = output / "overrides.json"
        # A partial directory may be resumed only when its declared inputs match.
        for path, value in ((snapshot_path, provenance), (overrides_path, overrides)):
            if path.exists() and read_json(path) != value:
                raise ValueError(f"Existing validation rescore has different inputs: {path}")
        if report_path.exists():
            if not snapshot_path.is_file() or not overrides_path.is_file():
                raise ValueError("Existing validation report has no matching provenance/overrides")
            report = read_json(report_path)
            reused = True
        else:
            output.mkdir(parents=True, exist_ok=True)
            save_json(snapshot_path, provenance)
            save_json(overrides_path, overrides)
            command = [sys.executable, "-u", "-m", "journal.run_experiment", "--mode", "evaluate",
                       "--checkpoint", str(checkpoint), "--config", str(overrides_path),
                       "--output", str(output), "--split", "validation", "--risk-scope", "deployment"]
            environment = os.environ.copy()
            for key in ("OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
                environment.pop(key, None)
            environment["OMP_PROC_BIND"] = "false"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                environment[key] = str(threads)
            with (output / "evaluate.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               env=environment, cwd=REPO, check=True)
            report = read_json(report_path)
            reused = False
        heldout = validate_report(report, checkpoint, expected, row["selected_epoch"], draws)
        if checkpoint_fingerprints(checkpoint) != item["sources"]:
            raise ValueError("Checkpoint inputs changed during validation evaluation")
        result.update(validation_accuracy=heldout["accuracy"],
                      validation_cross_entropy=heldout["cross_entropy"],
                      validation_rescore_status="complete", validation_rescore_reused=reused,
                      validation_rescore_draws=draws, validation_rescore_eval_seed=overrides["eval_seed"],
                      validation_rescore_report=str(report_path),
                      validation_rescore_bounded_loss=heldout["bounded_loss"],
                      validation_rescore_report_fingerprint=fingerprint(report_path))
    except Exception as error:
        result.update(validation_rescore_status="failed",
                      validation_rescore_error=f"{type(error).__name__}: {error}")
    return result


def rescore_search(manifest, output_root, workers=4, threads=2, draws=100):
    if min(workers, threads, draws) <= 0:
        raise ValueError("workers, threads and draws must be positive")
    if "SLURM_CPUS_PER_TASK" in os.environ and workers * threads > int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("Validation worker threads exceed allocated CPUs")
    manifest, output_root = Path(manifest).resolve(), Path(output_root).resolve()
    verify_submission_snapshot(manifest)
    manifest_source = fingerprint(manifest)
    entries = read_manifest(manifest)
    original_rows, items = preflight(entries, output_root)
    training_source = fingerprint(output_root / "training_results.json")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(rescore_one, item, draws, threads, manifest_source["sha256"]): item
                   for item in items}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results[row["task_id"]] = row
            print(json.dumps(dict(task_id=row["task_id"], status=row["validation_rescore_status"],
                                  finished=len(results), total=len(items))), flush=True)
    if (fingerprint(manifest) != manifest_source
            or fingerprint(output_root / "training_results.json") != training_source
            or read_json(output_root / "training_results.json") != original_rows):
        raise ValueError("Master manifest or original training rows changed during rescoring")
    if any(checkpoint_fingerprints(item["checkpoint"]) != item["sources"] for item in items):
        raise ValueError("Checkpoint inputs changed before selection publication")
    verify_submission_snapshot(manifest)
    rows = [results[entry["task_id"]] for entry in entries]
    failed = [row for row in rows if row["validation_rescore_status"] != "complete"]
    # Never substitute the old training-time scores if even one rescore fails.
    variants, selected = ([], []) if failed else select_variants(rows, entries, top=1)
    if not failed and (len(selected) != len(METHODS)
                       or any(not item["complete_seed_set"] or len(item["seeds"]) != 2 for item in selected)):
        raise ValueError("Expected four selected configurations retaining both seeds")
    summary = dict(snapshot=dict(manifest=str(manifest), manifest_fingerprint=manifest_source,
                                 training_results_fingerprint=training_source,
                                 submitted_git_sha=os.environ.get("JOURNAL_GIT_SHA"),
                                 slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                                 workers=workers, threads=threads),
                   training_trials=len(entries), completed_trials=len(items), failed_trials=[],
                   rescored_trials=len(rows) - len(failed), failed_evaluations=failed,
                   variant_validation_summaries=variants, selected_variants=selected,
                   selection="aggregate fresh validation accuracy; cross-entropy tiebreak",
                   selection_uses_test=False, validation_draws=draws,
                   inference_threads=threads,
                   training_threads_by_task={row["task_id"]: row["config"]["threads"] for row in rows},
                   validation_seed_offset=SEED_OFFSET,
                   checkpoint_selection="Frozen training-time checkpoints; rescore selects configurations only",
                   no_training_fallback=True)
    save_json(output_root / "validation_rescored_training_results.json", rows)
    save_json(output_root / "validation_selection.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=100)
    args = parser.parse_args()
    try:
        summary = rescore_search(args.manifest, args.output_root, args.workers, args.threads, args.draws)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Validation selection not generated: {error}\n")
    print(json.dumps(dict(rescored=summary["rescored_trials"], failed=len(summary["failed_evaluations"]),
                          selected=len(summary["selected_variants"]), selection_uses_test=False)))
    if summary["failed_evaluations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
