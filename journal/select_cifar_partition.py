"""Select a convolutional-receiver partition from frozen validation reports.

Every declared training case and fresh validation evaluation must be complete.
This module only reads saved metadata/reports and hashes checkpoint files; it
never loads model tensors, opens test reports, trains or performs inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .build_campaign import make_entry
from .campaign import save_json, select_variants, validate_entries
from .run_experiment import bounded_loss_description, normalize_config


ARCHITECTURES = {"cnn_mid_relu", "cnn_rxconv", "cnn_spatial3"}
METHODS = {"ERM-Matched": "erm", "L2-Matched": "l2",
           "T1-Matched": "proposed", "T2-Matched": "joint"}
EXPECTED_TRIALS = 27
SEED_OFFSET = 2_000_003
MINIMUM_MARGIN = .005
CONFIRM_SEEDS = (8101, 8102, 8103)
METRICS = ("accuracy", "cross_entropy", "bounded_loss", "feature_power_per_coordinate")


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    """Stream opaque checkpoint bytes without deserializing model weights."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return dict(sha256=digest.hexdigest(), bytes=size)


def read_manifest(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [entry["task_id"] for entry in entries]
    if len(entries) != EXPECTED_TRIALS or len(set(ids)) != EXPECTED_TRIALS:
        raise ValueError("Require all 27 unique manifest tasks")
    if any(not task or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for character in task) for task in ids):
        raise ValueError("Unsafe manifest task identifier")
    return entries


def number(value, name, upper=None, positive=False):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)
            or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def validate_training_manifest(entries):
    validate_entries(entries)
    grouped, common = defaultdict(list), None
    fixed_fields = ("dataset", "train_size", "val_size", "width", "epochs", "m", "data_seed",
                    "train_channel", "deployment_channel", "training_loss", "loss_epsilon",
                    "prior_std", "clip_ce", "delta", "batch_size", "val_draws",
                    "selection_metric", "validation_seed_policy", "cpu_flush_denormal")
    for entry in entries:
        config = normalize_config(entry["config"])
        label = entry["method_label"]
        if (entry.get("mode", "train") != "train" or label not in METHODS
                or config["method"] != METHODS[label]
                or config["architecture"] not in ARCHITECTURES
                or config["dataset"] != "cifar10" or config["train_size"] != 45000
                or config["val_size"] != 5000 or config["width"] != 128
                or config["epochs"] != 25 or config["seed"] != 7101
                or config["channel_seed"] != config["seed"] + 10000
                or config["eval_seed"] != config["seed"] + 20000
                or config["device"] != "cpu" or config["threads"] != 1
                or config["training_loss"] != "bounded_ce_smooth"
                or config["selection_metric"] != "accuracy"
                or config["validation_seed_policy"] != "fixed"
                or config["val_draws"] != 2 or config["cpu_flush_denormal"] is not True
                or config["train_channel"] != config["deployment_channel"]):
            raise ValueError(f"Unexpected partition-discovery protocol: {entry['task_id']}")
        family = "baseline" if config["method"] in {"erm", "l2"} else "proposed"
        if entry["family"] != family:
            raise ValueError(f"Mismatched method family: {entry['task_id']}")
        condition = {key: config[key] for key in fixed_fields}
        if common is None:
            common = condition
        elif common != condition:
            raise ValueError("Partition groups must share data, loss, channel and training budget")
        grouped[entry["group"]].append((entry, config))
    if (len(grouped) != 3 or {runs[0][1]["architecture"] for runs in grouped.values()} != ARCHITECTURES):
        raise ValueError("Require one group for each of the three declared architectures")
    for runs in grouped.values():
        if (len(runs) != 9 or len({config["architecture"] for _, config in runs}) != 1
                or Counter(entry["method_label"] for entry, _ in runs) !=
                {"ERM-Matched": 2, "L2-Matched": 3, "T1-Matched": 3, "T2-Matched": 1}):
            raise ValueError("Every architecture needs all nine ERM/L2/T1/T2 controls")
        by_method = {label: [config for entry, config in runs if entry["method_label"] == label]
                     for label in METHODS}
        if ({config["lr"] for config in by_method["ERM-Matched"]} != {.001, .003}
                or {(config["lr"], config["l2"]) for config in by_method["L2-Matched"]} !=
                {(.001, 1e-6), (.003, 1e-6), (.003, 1e-5)}
                or {config["rho_lr_scale"] for config in by_method["T1-Matched"]} != {1., .1, .03}
                or any(config["lr"] != .003 or config["posterior_std"] != .003
                       or config["eta_channel"] != .001 or config["eta_data"] != .01
                       for config in by_method["T1-Matched"])
                or any(config["rho_lr_scale"] != 1. for label in ("ERM-Matched", "L2-Matched")
                       for config in by_method[label])
                or any(config["lr"] != .003 or config["posterior_std"] != .003
                       or config["eta_data"] != .002 or config["rho_lr_scale"] != .1
                       for config in by_method["T2-Matched"])):
            raise ValueError("Partition controls do not match the declared discovery grid")


def same_checkpoint(report_path, declared_path, task_id):
    """Permit syncing relative repository paths from Spartan to a local copy."""
    report, declared = Path(report_path), Path(declared_path)
    if report.name != task_id or declared.name != task_id:
        return False
    if declared.is_absolute():
        return report.resolve() == declared.resolve()
    if report.resolve() == declared.resolve():
        return True
    return len(report.parts) >= len(declared.parts) and report.parts[-len(declared.parts):] == declared.parts


def validate_score(report, evaluation, expected, epoch, source_task):
    if (report.get("split") != "validation" or report.get("risk_scope") != "deployment"
            or report.get("selected_epoch") != epoch
            or not same_checkpoint(report["checkpoint"], evaluation["checkpoint"], source_task)
            or normalize_config(report["evaluation_config"]) != expected):
        raise ValueError(f"Mismatched fresh validation report: {source_task}")
    heldout = report["heldout_gibbs_mc"]
    if (heldout.get("examples") != 5000 or heldout.get("draws") != 100
            or heldout.get("channel_source") != "deployment"
            or heldout.get("posterior_rule") != "one weight draw fixed across entire loader per replicate"
            or heldout.get("bounded_loss_definition") != bounded_loss_description(expected)
            or len(heldout["replicate_statistics"]) != 100):
        raise ValueError(f"Expected 100 full-loader validation draws: {source_task}")
    for metric in METRICS:
        upper = 1 if metric in {"accuracy", "bounded_loss"} else None
        number(heldout[metric], metric, upper)
        values = [number(replicate[metric], f"replicate {metric}", upper)
                  for replicate in heldout["replicate_statistics"]]
        if abs(statistics.fmean(values) - heldout[metric]) > 1e-12:
            raise ValueError(f"Validation aggregate disagrees with replicates: {source_task}/{metric}")
    return heldout


def confirmation_entries(selected):
    entries = []
    for choice in selected:
        config = {**choice["config"], "epochs": 25, "train_size": 45000, "val_size": 5000,
                  "width": 128, "threads": 1, "device": "cpu", "eval_draws": 100, "val_draws": 2}
        for seed in CONFIRM_SEEDS:
            entry = make_entry(config, seed, choice["group"], choice["method_label"], "partition_confirm")
            entry.update(selected_from_validation_variant=choice["variant_id"],
                         selected_from_validation_tasks=choice["task_ids"],
                         selection_source="100-draw validation-only partition discovery; no test selection",
                         selection_uses_test=False)
            entries.append(entry)
    return entries


def save_manifest(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
                                 for entry in entries))
    temporary.replace(path)


def select_partition(train_manifest, eval_manifest, train_root, scores_root, output, confirm_manifest=None):
    train_manifest, eval_manifest = Path(train_manifest), Path(eval_manifest)
    train_root, scores_root, output = Path(train_root), Path(scores_root), Path(output)
    sources = {}

    def record(path):
        sources[str(Path(path).resolve())] = fingerprint(path)

    record(train_manifest)
    record(eval_manifest)
    training, evaluations = read_manifest(train_manifest), read_manifest(eval_manifest)
    validate_training_manifest(training)
    train_by_id = {entry["task_id"]: entry for entry in training}
    eval_by_source = {}
    for entry in evaluations:
        source = entry["source_task_id"]
        if source not in train_by_id or source in eval_by_source:
            raise ValueError("Evaluation tasks must cover each training task exactly once")
        train = train_by_id[source]
        config = normalize_config(train["config"])
        overrides = entry["config"]
        if (entry.get("mode") != "evaluate" or entry.get("split") != "validation"
                or entry.get("risk_scope") != "deployment"
                or Path(entry["checkpoint"]).name != source
                or any(entry[key] != train[key] for key in ("group", "method_label", "variant_id"))
                or set(overrides) - {"device", "threads", "eval_draws", "eval_seed"}
                or overrides.get("eval_draws") != 100
                or overrides.get("eval_seed") != config["eval_seed"] + SEED_OFFSET
                or overrides.get("device", config["device"]) != "cpu"):
            raise ValueError(f"Unexpected validation evaluation manifest: {entry['task_id']}")
        expected = normalize_config({**config, **overrides})
        eval_by_source[source] = (entry, expected)
    if set(eval_by_source) != set(train_by_id):
        raise ValueError("All 27 validation scores are required before selection")

    rows, splits, banks = [], set(), set()
    for entry in training:
        task = entry["task_id"]
        config = normalize_config(entry["config"])
        checkpoint = train_root / task
        meta_path, history_path = checkpoint / "metadata.json", checkpoint / "history.json"
        record(meta_path)
        record(history_path)
        metadata, history = read_json(meta_path), read_json(history_path)
        if normalize_config(metadata["config"]) != config:
            raise ValueError(f"Training configuration mismatch: {task}")
        epoch = metadata["selected_epoch"]
        if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 25
                or len(history) != 25 or [item["epoch"] for item in history] != list(range(1, 26))):
            raise ValueError(f"Incomplete training history or selected epoch: {task}")
        number(metadata["total_seconds"], "training completion time", positive=True)
        number(metadata["peak_rss_mb"], "training peak memory", positive=True)
        for item in history:
            number(item["validation"]["accuracy"], "historical validation accuracy", 1)
            number(item["validation"]["cross_entropy"], "historical validation CE")
        splits.add((metadata["split"]["train_indices_hash"], metadata["split"]["validation_indices_hash"]))
        banks.add(metadata["training_channel_bank_hash"])
        weight_path = checkpoint / "weights.pt"
        if weight_path.is_file():
            record(weight_path)
        evaluation, expected = eval_by_source[task]
        report_path = scores_root / evaluation["task_id"] / "evaluation.json"
        record(report_path)
        score = validate_score(read_json(report_path), evaluation, expected, epoch, task)
        original = history[epoch - 1]["validation"]
        rows.append(dict(task_id=task, group=entry["group"], method_label=entry["method_label"],
                         family=entry["family"], variant_id=entry["variant_id"], stage=entry["stage"],
                         role=entry.get("role", "comparison"), seed=config["seed"], config=config,
                         checkpoint=str(checkpoint.resolve()), status="complete", selected_epoch=epoch,
                         seconds=metadata["total_seconds"], peak_rss_mb=metadata["peak_rss_mb"],
                         git_sha=metadata["git_sha"], validation_accuracy=score["accuracy"],
                         validation_cross_entropy=score["cross_entropy"],
                         original_validation_accuracy=original["accuracy"],
                         original_validation_cross_entropy=original["cross_entropy"],
                         validation_bounded_loss=score["bounded_loss"],
                         validation_feature_power=score["feature_power_per_coordinate"],
                         validation_report=str(report_path.resolve()), validation_draws=100,
                         validation_eval_seed=expected["eval_seed"],
                         validation_threads=expected["threads"], weights_available=weight_path.is_file()))
    if len(splits) != 1 or len(banks) != 1:
        raise ValueError("All partition cases must share data splits and the paired 128-symbol channel bank")
    variants, selected = select_variants(rows, training, top=1)
    if len(selected) != 12 or any(not item["complete_seed_set"] or item["seeds"] != [7101] for item in selected):
        raise ValueError("Require all four selected methods in all three architecture groups")
    groups = []
    for group in sorted({entry["group"] for entry in training}):
        choices = [item for item in selected if item["group"] == group]
        order = lambda item: (-item["validation_accuracy"]["mean"], item["validation_cross_entropy"]["mean"],
                              item["method_label"])
        baseline = min((item for item in choices if item["family"] == "baseline"), key=order)
        proposed = min((item for item in choices if item["family"] == "proposed"), key=order)
        margin = proposed["validation_accuracy"]["mean"] - baseline["validation_accuracy"]["mean"]
        groups.append(dict(group=group, architecture=choices[0]["config"]["architecture"],
                           best_baseline_method=baseline["method_label"],
                           best_baseline_accuracy=baseline["validation_accuracy"]["mean"],
                           best_proposed_method=proposed["method_label"],
                           best_proposed_accuracy=proposed["validation_accuracy"]["mean"],
                           best_proposed_cross_entropy=proposed["validation_cross_entropy"]["mean"],
                           margin=margin, eligible=margin + 1e-12 >= MINIMUM_MARGIN,
                           selected_variants=choices))
    eligible = [group for group in groups if group["eligible"]]
    winner = min(eligible, key=lambda group: (-group["best_proposed_accuracy"],
                                             group["best_proposed_cross_entropy"], group["group"])) if eligible else None
    confirm = confirmation_entries(winner["selected_variants"]) if winner else []
    result = dict(status="selected" if winner else "no_eligible_group", rows=rows,
                  training_trials=27, completed_trials=27, validation_draws=100,
                  selection_uses_test=False, new_optimization=0, checkpoint_inference=False,
                  variant_validation_summaries=variants, selected_variants=selected, groups=groups,
                  selected_group=winner["group"] if winner else None,
                  selected_architecture=winner["architecture"] if winner else None,
                  eligibility_margin=MINIMUM_MARGIN, eligibility_tolerance=1e-12,
                  selection_rule="Among groups with proposed-minus-best-baseline validation accuracy >=0.005, maximize absolute proposed validation accuracy; CE then group name break ties",
                  interpretation="Single-seed validation-only architecture discovery; all unfavorable variants/groups retained; frozen fresh-seed confirmation required",
                  source_fingerprints=sources, confirmation_entries=confirm,
                  confirmation_manifest=str(Path(confirm_manifest)) if winner and confirm_manifest else None,
                  confirmation_seeds=list(CONFIRM_SEEDS) if winner else [])
    for path, expected_hash in sources.items():
        if fingerprint(path) != expected_hash:
            raise ValueError(f"Selection source changed before publication: {path}")
    if winner and confirm_manifest is not None:
        save_manifest(confirm_manifest, confirm)
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--scores-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-manifest", type=Path)
    args = parser.parse_args()
    try:
        result = select_partition(args.train_manifest, args.eval_manifest, args.train_root,
                                  args.scores_root, args.output, args.confirm_manifest)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Partition selection not published: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_group", "selected_architecture")}))


if __name__ == "__main__":
    main()
