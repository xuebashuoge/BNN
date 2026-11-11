"""Execute a fully recorded CPU campaign and evaluate validation-selected models.

All exploratory outcomes, seeds, deployment conditions and failures are retained.
Test rankings are descriptive search results; confirmation uses fresh seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def describe(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0, mean=None, sd=None, ci95_low=None, ci95_high=None)
    n, mean = len(values), statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    if n > 1:
        from scipy.stats import t
        half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
    else:
        half = None
    return dict(n=n, mean=mean, sd=sd,
                ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None)


def validate_entries(entries):
    from journal.run_experiment import normalize_config
    seen, conditions, labels = set(), {}, {}
    for entry in entries:
        if entry["task_id"] in seen:
            raise ValueError(f"Duplicate task id {entry['task_id']}")
        seen.add(entry["task_id"])
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in entry["task_id"]):
            raise ValueError("Unsafe task id")
        config = normalize_config(entry["config"])
        if entry.get("mode", "train") != "train":
            raise ValueError("Campaign entries are explicit training trials")
        fixed = {key: config[key] for key in ("dataset", "architecture", "train_size", "val_size", "width", "m", "deployment_channel", "data_seed")}
        key = entry["group"]
        if key in conditions and conditions[key] != fixed:
            raise ValueError(f"Comparison group {key} mixes deployment/data/model/bank conditions")
        conditions[key] = fixed
        label_key = (key, entry["method_label"])
        semantics = {key: config[key] for key in ("method", "training_loss", "train_channel")}
        if label_key in labels and labels[label_key] != semantics:
            raise ValueError(f"Method label {label_key} mixes distinct algorithms")
        labels[label_key] = semantics


def completed_trial(path, expected_config):
    try:
        from journal.run_experiment import normalize_config
        meta = json.loads((path / "metadata.json").read_text())
        history = json.loads((path / "history.json").read_text())
        complete = ((path / "weights.pt").is_file() and len(history) == expected_config["epochs"]
                    and history[-1]["epoch"] == expected_config["epochs"]
                    and math.isfinite(meta["total_seconds"]) and meta["total_seconds"] > 0
                    and normalize_config(meta["config"]) == normalize_config(expected_config))
        if not complete:
            return None
        selected = history[int(meta["selected_epoch"]) - 1]["validation"]
        if not all(math.isfinite(selected[key]) for key in ("accuracy", "cross_entropy")):
            return None
        return dict(validation_accuracy=selected["accuracy"], validation_cross_entropy=selected["cross_entropy"],
                    selected_epoch=meta["selected_epoch"], seconds=meta["total_seconds"],
                    peak_rss_mb=meta["peak_rss_mb"], git_sha=meta["git_sha"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def train_trial(entry, output_root, threads, resume, device="cpu"):
    from journal.run_experiment import normalize_config
    output = Path(output_root) / entry["task_id"]
    config = normalize_config({**entry["config"], "device": device, "threads": threads})
    base = dict(task_id=entry["task_id"], group=entry["group"], method_label=entry["method_label"],
                family=entry["family"], variant_id=entry["variant_id"], stage=entry["stage"],
                role=entry.get("role", "comparison"), seed=config["seed"], config=config,
                checkpoint=str(output.resolve()))
    cached = completed_trial(output, config) if resume else None
    if cached:
        return {**base, **cached, "status": "complete", "resumed": True}
    if (output / "weights.pt").exists():
        return {**base, "status": "failed", "error": "Incomplete or mismatched checkpoint exists; use a fresh campaign root"}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "task_config.json", config)
    save_json(output / "manifest_entry.json", entry)
    command = [sys.executable, "-u", "-m", "journal.run_experiment", "--mode", "train",
               "--config", str(output / "task_config.json"), "--output", str(output)]
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(threads)
    try:
        with (output / "train.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        summary = completed_trial(output, config)
        if result.returncode or summary is None:
            return {**base, "status": "failed", "error": f"Training exit {result.returncode}; see train.log"}
        return {**base, **summary, "status": "complete", "resumed": False}
    except Exception as error:
        return {**base, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def select_variants(rows, entries, top):
    expected = defaultdict(set)
    for entry in entries:
        expected[(entry["group"], entry["method_label"], entry["variant_id"])].add(entry["config"]["seed"])
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(row["group"], row["method_label"], row["variant_id"])].append(row)
    choices, variant_summaries = defaultdict(list), []
    for key, runs in grouped.items():
        seed_set = {run["seed"] for run in runs}
        item = dict(group=key[0], method_label=key[1], variant_id=key[2], family=runs[0]["family"],
                    config=runs[0]["config"], seeds=sorted(seed_set),
                    complete_seed_set=seed_set == expected[key],
                    validation_accuracy=describe([row["validation_accuracy"] for row in runs]),
                    validation_cross_entropy=describe([row["validation_cross_entropy"] for row in runs]),
                    task_ids=[row["task_id"] for row in runs])
        variant_summaries.append(item)
        if item["complete_seed_set"]:
            choices[key[:2]].append(item)
    selected = []
    for key, variants in sorted(choices.items()):
        variants.sort(key=lambda item: (-item["validation_accuracy"]["mean"], item["validation_cross_entropy"]["mean"], item["variant_id"]))
        selected.extend({**item, "rank": rank} for rank, item in enumerate(variants[:top], 1))
    return variant_summaries, selected


def evaluate_trial(row, draws, full_risks, resume):
    checkpoint = Path(row["checkpoint"])
    output = checkpoint / ("test_full" if full_risks else "test_deployment")
    overrides = dict(eval_draws=draws, eval_seed=row["config"]["eval_seed"] + 1000003,
                     device=row["config"].get("device", "cpu"), threads=row["config"]["threads"])
    scope = "full" if full_risks else "deployment"
    try:
        report = json.loads((output / "evaluation.json").read_text()) if resume and (output / "evaluation.json").exists() else None
        if report and any(report["evaluation_config"].get(key) != value for key, value in overrides.items()):
            raise ValueError("Existing evaluation has different settings")
        if report is None:
            output.mkdir(parents=True, exist_ok=True)
            save_json(output / "overrides.json", overrides)
            command = [sys.executable, "-u", "-m", "journal.run_experiment", "--mode", "evaluate",
                       "--checkpoint", str(checkpoint), "--config", str(output / "overrides.json"),
                       "--output", str(output), "--split", "test", "--risk-scope", scope]
            environment = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                environment[key] = str(row["config"]["threads"])
            with (output / "evaluate.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
            report = json.loads((output / "evaluation.json").read_text())
        return {**row, "test_status": "complete", "test_accuracy": report["heldout_gibbs_mc"]["accuracy"],
                "test_cross_entropy": report["heldout_gibbs_mc"]["cross_entropy"],
                "test_report": str((output / "evaluation.json").resolve())}
    except Exception as error:
        return {**row, "test_status": "failed", "test_error": f"{type(error).__name__}: {error}"}


def test_comparisons(evaluations, expected_labels=None, expected_seeds=None):
    groups = defaultdict(list)
    for row in evaluations:
        if row.get("test_status") == "complete":
            groups[row["group"]].append(row)
    reports = []
    for group, rows in sorted(groups.items()):
        by_method = defaultdict(list)
        for row in rows:
            by_method[row["method_label"]].append(row)
        method_results = []
        for method, runs in sorted(by_method.items()):
            # Top=1 is the intended confirmatory comparison. With top>1 each
            # variant remains a distinct record rather than pooling a search.
            variants = defaultdict(list)
            for row in runs:
                variants[row["variant_id"]].append(row)
            for variant_id, runs in variants.items():
                method_results.append(dict(method_label=method, variant_id=variant_id, family=runs[0]["family"],
                                           seeds=sorted(row["seed"] for row in runs),
                                           test_accuracy=describe([row["test_accuracy"] for row in runs]),
                                           test_cross_entropy=describe([row["test_cross_entropy"] for row in runs])))
        baselines = [item for item in method_results if item["family"] == "baseline"]
        proposed = [item for item in method_results if item["family"] == "proposed"]
        strongest = max(baselines, key=lambda item: item["test_accuracy"]["mean"]) if baselines else None
        candidate = max(proposed, key=lambda item: item["test_accuracy"]["mean"]) if proposed else None
        seed_comparisons = []
        for seed in sorted({row["seed"] for row in rows}):
            seed_rows = [row for row in rows if row["seed"] == seed]
            base_rows = [row for row in seed_rows if row["family"] == "baseline"]
            prop_rows = [row for row in seed_rows if row["family"] == "proposed"]
            if base_rows and prop_rows:
                baseline = max(base_rows, key=lambda row: row["test_accuracy"])
                best = max(prop_rows, key=lambda row: row["test_accuracy"])
                margin = best["test_accuracy"] - baseline["test_accuracy"]
                seed_comparisons.append(dict(seed=seed, best_proposed=best["method_label"],
                                             strongest_baseline=baseline["method_label"], margin=margin,
                                             proposed_strictly_best=margin > 1e-12, proposed_ties_best=abs(margin) <= 1e-12,
                                             status="exploratory best objective per seed; all seeds retained"))
        report = dict(group=group, methods=method_results, strongest_baseline_by_test_mean=strongest,
                      best_proposed_by_test_mean=candidate, seed_comparisons=seed_comparisons,
                      evaluated_methods=sorted(by_method),
                      interpretation="Hyperparameters selected on aggregate validation; test-based winner and seed rankings are exploratory descriptions")
        expected = set((expected_labels or {}).get(group, by_method))
        report["all_expected_methods_evaluated"] = expected <= set(by_method)
        report["missing_methods"] = sorted(expected - set(by_method))
        missing_seeds = {}
        for method in expected:
            needed = set((expected_seeds or {}).get(group, {}).get(method, []))
            observed = {row["seed"] for row in by_method.get(method, [])}
            if needed - observed:
                missing_seeds[method] = sorted(needed - observed)
        report["missing_seeds_by_method"] = missing_seeds
        report["all_expected_seeds_evaluated"] = not missing_seeds
        complete_coverage = report["all_expected_methods_evaluated"] and report["all_expected_seeds_evaluated"]
        base_rows = [row for row in rows if row["family"] == "baseline"]
        prop_rows = [row for row in rows if row["family"] == "proposed"]
        if base_rows and prop_rows:
            best_base = max(base_rows, key=lambda row: row["test_accuracy"])
            best_prop = max(prop_rows, key=lambda row: row["test_accuracy"])
            compact_fields = ("task_id", "method_label", "variant_id", "seed", "config", "test_accuracy", "test_report")
            report["exploratory_best_proposed_single_run"] = {key: best_prop.get(key) for key in compact_fields}
            report["exploratory_best_baseline_single_run"] = {key: best_base.get(key) for key in compact_fields}
            report["exploratory_best_single_run_margin"] = best_prop["test_accuracy"] - best_base["test_accuracy"]
        if strongest and candidate:
            margin = candidate["test_accuracy"]["mean"] - strongest["test_accuracy"]["mean"]
            paired = []
            for seed in set(strongest["seeds"]) & set(candidate["seeds"]):
                p = next(row for row in rows if row["seed"] == seed and row["variant_id"] == candidate["variant_id"])
                b = next(row for row in rows if row["seed"] == seed and row["variant_id"] == strongest["variant_id"])
                paired.append(p["test_accuracy"] - b["test_accuracy"])
            report.update(mean_accuracy_margin=margin, proposed_strictly_highest_mean=margin > 1e-12,
                          proposed_ties_highest_mean=abs(margin) <= 1e-12, paired_difference=describe(paired),
                          goal_met_for_all_evaluated_seeds=complete_coverage and bool(seed_comparisons) and all(item["proposed_strictly_best"] for item in seed_comparisons),
                          goal_met_with_complete_method_coverage=margin > 1e-12 and complete_coverage)
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-pilot", type=Path, help="Completed matching CPU pilot root, required for CUDA execution")
    parser.add_argument("--evaluate-top", type=int, default=1,
                        help="Validation-ranked variants per method; 0 skips all test evaluation")
    parser.add_argument("--eval-draws", type=int, default=5)
    parser.add_argument("--full-risks", action="store_true", help="Also compute frozen empirical/channel/data decomposition")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true", help="Read saved JSON outcomes and recompute summaries without training or inference")
    args = parser.parse_args()
    if min(args.workers, args.threads, args.eval_draws) <= 0 or args.evaluate_top < 0:
        parser.error("workers/threads/draws must be positive and evaluate-top nonnegative")
    if args.device == "cuda" and args.workers != 1:
        parser.error("Use one sequential campaign worker per allocated GPU")
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    validate_entries(entries)
    if args.validate_only:
        print(json.dumps({"tasks": len(entries), "workers": args.workers, "threads": args.threads,
                          "groups": sorted({entry["group"] for entry in entries})}))
        return
    expected_labels = defaultdict(set)
    expected_seeds = defaultdict(lambda: defaultdict(set))
    for entry in entries:
        expected_labels[entry["group"]].add(entry["method_label"])
        expected_seeds[entry["group"]][entry["method_label"]].add(entry["config"]["seed"])
    if args.analyze_only:
        rows = json.loads((args.output_root / "training_results.json").read_text())
        evaluations_path = args.output_root / "test_results.json"
        evaluations = json.loads(evaluations_path.read_text()) if evaluations_path.exists() else []
        variants, selected = select_variants(rows, entries, args.evaluate_top)
        summary = dict(training_trials=len(entries), completed_trials=sum(row["status"] == "complete" for row in rows),
                       failed_trials=[row for row in rows if row["status"] != "complete"],
                       variant_validation_summaries=variants, selected_variants=selected,
                       test_comparisons=test_comparisons(evaluations, expected_labels, expected_seeds), selection_uses_test=False,
                       failed_evaluations=[row for row in evaluations if row.get("test_status") != "complete"],
                       analysis_only=True)
        save_json(args.output_root / "summary.json", summary)
        print(json.dumps({"summary": str((args.output_root / "summary.json").resolve()),
                          "training_complete": summary["completed_trials"], "evaluated": len(evaluations)}))
        return
    # The same submitted revision/manifest contract as individual array tasks.
    from journal.run_task import verify_submission_snapshot
    verify_submission_snapshot(args.manifest)
    cpu_pilot_matches = []
    if args.device == "cuda":
        if args.cpu_pilot is None:
            parser.error("CUDA execution requires --cpu-pilot with completed matching CPU trials")
        from journal.run_task import require_cpu_pilots
        cpu_pilot_matches = require_cpu_pilots(entries, args.cpu_pilot)
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    snapshot = dict(manifest=str(args.manifest.resolve()), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    submitted_git_sha=os.environ.get("JOURNAL_GIT_SHA"), workers=args.workers, threads=args.threads,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), device=args.device,
                    cpu_pilot_matches=cpu_pilot_matches, selection="aggregate validation accuracy; CE tiebreak",
                    all_outcomes_retained=True)
    save_json(args.output_root / "campaign_snapshot.json", snapshot)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_trial, entry, args.output_root, args.threads, args.resume, args.device): entry for entry in entries}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            save_json(args.output_root / "training_results.json", rows)
            print(json.dumps({"phase": "train", "finished": len(rows), "total": len(entries),
                              "task_id": row["task_id"], "status": row["status"],
                              "validation_accuracy": row.get("validation_accuracy")}), flush=True)
    variants, selected = select_variants(rows, entries, args.evaluate_top)
    summary = dict(snapshot=snapshot, training_trials=len(entries),
                   completed_trials=sum(row["status"] == "complete" for row in rows),
                   failed_trials=[row for row in rows if row["status"] != "complete"],
                   variant_validation_summaries=variants, selected_variants=selected,
                   test_comparisons=[], selection_uses_test=False)
    save_json(args.output_root / "summary.json", summary)
    selected_tasks = {task for choice in selected for task in choice["task_ids"]}
    candidates = [row for row in rows if row["task_id"] in selected_tasks]
    evaluations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_trial, row, args.eval_draws,
                                   args.full_risks or row.get("role") in {"mechanism", "mismatch", "reference"},
                                   args.resume) for row in candidates]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            evaluations.append(row)
            save_json(args.output_root / "test_results.json", evaluations)
            print(json.dumps({"phase": "test", "finished": len(evaluations), "total": len(candidates),
                              "task_id": row["task_id"], "status": row["test_status"],
                              "test_accuracy": row.get("test_accuracy")}), flush=True)
    summary.update(test_comparisons=test_comparisons(evaluations, expected_labels, expected_seeds),
                   failed_evaluations=[row for row in evaluations if row.get("test_status") != "complete"],
                   elapsed_seconds=time.perf_counter()-start)
    save_json(args.output_root / "summary.json", summary)
    fields = ["task_id", "group", "method_label", "family", "variant_id", "seed", "status",
              "validation_accuracy", "validation_cross_entropy", "selected_epoch", "seconds", "test_status", "test_accuracy", "test_cross_entropy"]
    evaluated_by_id = {row["task_id"]: row for row in evaluations}
    with (args.output_root / "all_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, **evaluated_by_id.get(row["task_id"], {})} for row in rows)
    print(json.dumps({"summary": str((args.output_root / "summary.json").resolve()),
                      "complete": summary["completed_trials"], "failed": len(summary["failed_trials"]),
                      "conditions": [{"group": item["group"], "proposed_highest_mean": item.get("proposed_strictly_highest_mean"),
                                      "margin": item.get("mean_accuracy_margin")} for item in summary["test_comparisons"]]}), flush=True)
    if summary["failed_trials"] or summary["failed_evaluations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
