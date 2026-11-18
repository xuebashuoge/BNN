"""Screen, confirm and report the full-matrix fading accuracy comparison.

Each array task has one fixed configuration and writes its own result. The
selection stage reads complete validation records. Test inference runs only for
the fixed confirmation cohorts, with a common seed for every paired method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from .build_bayesian_campaign import DATASETS, base_config
from .build_campaign import make_entry
from .campaign import describe, evaluate_trial, save_json, select_variants, train_trial, validate_entries
from .run_experiment import file_fingerprint, normalize_config
from .run_task import read_manifest, verify_submission_snapshot


IMAGE_DATASETS = ("mnist", "fashionmnist", "cifar10")
DATASET_NAMES = dict(moons="Two moons", mnist="MNIST", fashionmnist="Fashion-MNIST", cifar10="CIFAR-10")
CHANNELS = {
    "full_fading": dict(kind="full_fading", rx_correlation=.25, tx_correlation=.25, noise_std=1.),
    "outage": dict(kind="outage", outage_prob=.5),
}
METHODS = {
    "ERM-Matched": ("erm", False), "L2-Matched": ("l2", False),
    "T1-Matched": ("proposed", False), "T2-Matched": ("joint", False),
    "ERM-Perfect": ("erm", True), "L2-Perfect": ("l2", True),
    "Proposed-Perfect": ("proposed", True),
}
SCREEN_SEEDS = (92101, 92102)
CONFIRM_SEEDS = (93101, 93102, 93103)
REFINED_CONFIRM_SEEDS = (94101, 94102, 94103)
PILOT_SEEDS = (92000,)
STAGES = {"pilot": "full_fading_accuracy_pilot", "screen": "full_fading_accuracy_screen",
          "confirm": "full_fading_accuracy_confirm", "refine": "full_fading_accuracy_refine",
          "refined_confirm": "full_fading_accuracy_refined_confirm"}
CONFIRMATION_COHORTS = {"confirm": CONFIRM_SEEDS, "refined_confirm": REFINED_CONFIRM_SEEDS}
TEST_SEED_OFFSET = 1000003
EXECUTION_SETTINGS = {"device", "threads"}
GAUSSIAN_BASE_CHANNEL = {"full_fading": "analog", "outage": "outage"}


def variants(config, method, perfect, refinement=False):
    """Four candidates per algorithm and training regime."""
    config = {**config, "method": method}
    if method == "erm":
        learning_rates = (.0006, .002, .006, .02) if refinement else (.0003, .001, .003, .01)
        for lr in learning_rates:
            yield dict(config, lr=lr)
        return
    if method == "l2":
        strengths = (1e-8, 1e-4) if refinement else (1e-7, 1e-5)
    elif perfect:
        strengths = (.001, .3) if refinement else (.01, .1)
    else:
        strengths = (.0001, .03) if refinement else (.001, .01)
    learning_rates = (.006, .01) if refinement else (.001, .003)
    for lr in learning_rates:
        for strength in strengths:
            candidate = dict(config, lr=lr)
            if method == "l2":
                candidate["l2"] = strength
            elif perfect:
                candidate.update(eta_channel=0., eta_data=strength)
            else:
                ratio = math.sqrt((config["train_size"] - 1) / (config["m"] - 1))
                if method == "proposed":
                    candidate.update(eta_channel=strength, eta_data=strength * ratio)
                else:
                    candidate.update(eta_channel=0., eta_data=2 * strength / math.sqrt(1 + 1 / ratio**2))
            yield candidate


def entry(config, seed, group, label, stage):
    item = make_entry(config, seed, group, label, STAGES[stage])
    item["family"] = "proposed" if config["method"] in {"proposed", "joint"} else "baseline"
    return item


def screening_entries(datasets, channels, m, pilot, refinement=False):
    if m < 2:
        raise ValueError("A matched training bank requires m >= 2")
    if pilot:
        stage, seeds = "pilot", PILOT_SEEDS
    else:
        stage, seeds = "refine" if refinement else "screen", SCREEN_SEEDS
    entries = []
    for dataset in datasets:
        for channel in channels:
            deployment = CHANNELS[channel]
            group = f"{dataset}_{channel}_m{m}"
            base = base_config(dataset, GAUSSIAN_BASE_CHANNEL[channel], stage)
            base.update(m=m, val_draws=3, eval_draws=20, data_seed=20261118,
                        deployment_channel=deployment)
            for label, (method, perfect) in METHODS.items():
                training = {"kind": "perfect"} if perfect else deployment
                candidates = list(variants(dict(base, train_channel=training), method, perfect, refinement))
                for config in candidates[:1] if pilot else candidates:
                    entries.extend(entry(config, seed, group, label, stage) for seed in seeds)
    return entries


def write_manifest(path, entries):
    validate_entries(entries)
    encoded = "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in entries)
    write_once(path, encoded)
    return dict(manifest=str(path), tasks=len(entries), sha256=hashlib.sha256(encoded.encode()).hexdigest())


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if path.read_text() != content:
                raise ValueError(f"Existing comparison input differs: {path}")
        else:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content)
            temporary.replace(path)


def stage_entries(manifest, stage=None):
    entries = read_manifest(manifest)
    stages = {item["stage"] for item in entries}
    allowed = {STAGES[stage]} if stage else set(STAGES.values())
    if len(stages) != 1 or not stages <= allowed:
        raise ValueError("Comparison manifest has the wrong or mixed stages")
    return entries


def report_file(root, item):
    return root / "reports" / (item["task_id"] + ".json")


def validate_result(item, row):
    for key in ("task_id", "stage", "group", "method_label", "family", "variant_id"):
        if row.get(key) != item[key]:
            raise ValueError(f"Result {key} differs from its declared task: {item['task_id']}")
    expected, observed = normalize_config(item["config"]), normalize_config(row["config"])
    if (row["status"] != "complete" or row["seed"] != expected["seed"]
            or any(observed[key] != expected[key] for key in expected if key not in EXECUTION_SETTINGS)):
        raise ValueError(f"Failed task or changed result configuration: {item['task_id']}")


def run_task(manifest, root, index, device, threads):
    verify_submission_snapshot(manifest)
    entries = stage_entries(manifest)
    if not 0 <= index < len(entries):
        raise ValueError(f"Task index {index} is outside a manifest of {len(entries)} entries")
    item = entries[index]
    result_path = report_file(root, item)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = train_trial(item, root, threads, resume=True, device=device)
        save_json(result_path, row)
        if row["status"] != "complete":
            raise RuntimeError(f"{item['task_id']}: {row['error']}")
        if item["stage"] in {STAGES[stage] for stage in CONFIRMATION_COHORTS}:
            row = evaluate_trial(row, item["config"]["eval_draws"], full_risks=False, resume=True)
            save_json(result_path, row)
            if row["test_status"] != "complete":
                raise RuntimeError(f"{item['task_id']}: {row['test_error']}")
            validate_test(item, row, root)
    return dict(task_id=item["task_id"], status="complete", validation_accuracy=row["validation_accuracy"],
                test_accuracy=row.get("test_accuracy"), seconds=row["seconds"], resumed=row["resumed"])


def read_results(root, entries):
    rows = []
    for item in entries:
        path = report_file(root, item)
        if not path.exists():
            raise FileNotFoundError(f"Missing declared result: {path}")
        row = json.loads(path.read_text())
        validate_result(item, row)
        rows.append(row)
    return rows


def variant_grid(entries, expected):
    if len(entries) != len(expected):
        return False
    return {item["task_id"]: item for item in entries} == {item["task_id"]: item for item in expected}


def screen_scope(entries):
    """Recover scope from the manifest, then regenerate the complete design."""
    groups = defaultdict(list)
    for item in entries:
        groups[item["group"]].append(item)
    conditions = []
    for group, items in groups.items():
        sample = items[0]["config"]
        condition = (sample["dataset"], sample["deployment_channel"]["kind"], sample["m"])
        dataset, channel, m = condition
        expected = screening_entries((dataset,), (channel,), m, pilot=False)
        if not variant_grid(items, expected):
            raise ValueError(f"Incomplete or altered comparison validation grid: {group}")
        conditions.append(condition)
    return conditions


def refinement_entries(conditions):
    return [item for dataset, channel, m in conditions
            for item in screening_entries((dataset,), (channel,), m, pilot=False, refinement=True)]


def confirmation_entries(selected, eval_draws, stage="confirm"):
    final = []
    for choice in selected:
        config = {**choice["config"], "eval_draws": eval_draws}
        final.extend({**entry(config, seed, choice["group"], choice["method_label"], stage),
                      "selected_from_validation_variant": choice["variant_id"]} for seed in CONFIRMATION_COHORTS[stage])
    return final


def prepare_confirmation(screen_manifest, screen_root, output, eval_draws,
                         refinement_manifest=None, refinement_root=None):
    if (refinement_manifest is None) != (refinement_root is None):
        raise ValueError("Specify both the refinement manifest and its results root")
    entries = stage_entries(screen_manifest, "screen")
    conditions = screen_scope(entries)
    rows = read_results(screen_root, entries)
    stage = "confirm"
    if refinement_manifest is not None:
        additional = stage_entries(refinement_manifest, "refine")
        if not variant_grid(additional, refinement_entries(conditions)):
            raise ValueError("Refinement must cover every initial validation group and candidate")
        rows.extend(read_results(refinement_root, additional))
        entries.extend(additional)
        stage = "refined_confirm"
    for row in rows:
        if any(metric in row for metric in ("test_accuracy", "test_cross_entropy", "test_status")):
            raise ValueError("Confirmation selection requires validation-only training results")
        if (not all(math.isfinite(row[metric]) for metric in ("validation_accuracy", "validation_cross_entropy"))
                or not 0 <= row["validation_accuracy"] <= 1 or row["validation_cross_entropy"] < 0):
            raise ValueError(f"Invalid validation metric: {row['task_id']}")
    summaries, selected = select_variants(rows, entries, top=1)
    final = confirmation_entries(selected, eval_draws, stage)
    selection = dict(screen_manifest=file_fingerprint(screen_manifest), screening_seeds=SCREEN_SEEDS,
                     confirmation_seeds=CONFIRMATION_COHORTS[stage], eval_draws=eval_draws,
                     rule="Highest mean validation accuracy, then lowest mean validation CE, then variant ID",
                     validation_trials=sorted(summaries, key=lambda row: (row["group"], row["method_label"], row["variant_id"])),
                     selected_variants=selected)
    if refinement_manifest is not None:
        fields = ("task_id", "stage", "variant_id", "seed", "validation_accuracy", "validation_cross_entropy")
        selection.update(refinement_manifest=file_fingerprint(refinement_manifest),
                         validation_candidates_per_method=8,
                         validation_by_seed=[{key: row[key] for key in fields} for row in rows],
                         confirmation_sampling="Fresh initialization, bank, posterior and channel seeds on the same official test examples")
    write_once(output.with_suffix(".selection.json"), json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return write_manifest(output, final)


def validate_test(item, row, root):
    checkpoint = root / item["task_id"]
    path = checkpoint / "test_deployment" / "evaluation.json"
    evaluation = json.loads(path.read_text())
    trained = normalize_config(item["config"])
    expected = {**trained, "eval_seed": trained["eval_seed"] + TEST_SEED_OFFSET}
    evaluated = normalize_config(evaluation["evaluation_config"])
    risk = evaluation["heldout_gibbs_mc"]
    if (row.get("test_status") != "complete" or evaluation["split"] != "test"
            or evaluation["risk_scope"] != "deployment" or risk["channel_source"] != "deployment"
            or evaluation["checkpoint_verification"]["bank_hash_matches_metadata"] is not True
            or evaluation["checkpoint_source_fingerprints"]["metadata"] != file_fingerprint(checkpoint / "metadata.json")
            or any(evaluated[key] != expected[key] for key in expected if key not in EXECUTION_SETTINGS)
            or risk["draws"] != expected["eval_draws"] or risk["examples"] != 10000
            or not risk["predictor"]["stochastic"]
            or not math.isfinite(risk["cross_entropy"]) or risk["cross_entropy"] < 0
            or not 0 <= risk["accuracy"] <= 1 or not 0 <= risk["bounded_loss"] <= 1
            or row["test_accuracy"] != risk["accuracy"] or row["test_cross_entropy"] != risk["cross_entropy"]):
        raise ValueError(f"Invalid frozen test report: {item['task_id']}")
    return {**row, "dataset": trained["dataset"], "channel": trained["deployment_channel"]["kind"],
            "m": trained["m"], "test_bounded_loss": risk["bounded_loss"]}


def group_results(rows, seeds=CONFIRM_SEEDS):
    groups = defaultdict(lambda: defaultdict(dict))
    fields = ("test_accuracy", "test_cross_entropy", "test_bounded_loss")
    for row in rows:
        methods = groups[row["group"]]
        runs = methods[row["method_label"]]
        if row["seed"] in runs:
            raise ValueError(f"Duplicate method/seed in {row['group']}")
        runs[row["seed"]] = row
    summaries = []
    for group, methods in sorted(groups.items()):
        if set(methods) != set(METHODS) or any(set(runs) != set(seeds) for runs in methods.values()):
            raise ValueError(f"Incomplete paired confirmation cohort: {group}")
        results = {}
        for method, runs in methods.items():
            if len({run["variant_id"] for run in runs.values()}) != 1:
                raise ValueError(f"Confirm the same variant across all seeds: {group}, {method}")
            results[method] = {metric: describe(run[metric] for run in runs.values()) for metric in fields}
        means = {method: result["test_accuracy"]["mean"] for method, result in results.items()}
        best = max(means.values())
        winners = sorted(label for label, mean in means.items() if abs(mean - best) <= 1e-12)
        paired = {method: {baseline: describe(methods[method][seed]["test_accuracy"] -
                                             methods[baseline][seed]["test_accuracy"] for seed in seeds)
                           for baseline in ("ERM-Matched", "L2-Matched")}
                  for method in ("T1-Matched", "T2-Matched")}
        sample = methods["ERM-Matched"][seeds[0]]
        summaries.append(dict(group=group, dataset=sample["dataset"], channel=sample["channel"], m=sample["m"],
                              methods=results, highest_mean_methods=winners,
                              t1_or_t2_strictly_highest=len(winners) == 1 and winners[0] in paired,
                              paired_accuracy_differences=paired))
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(groups, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#686868", "#B28647", "#0072B2", "#D55E00", "#686868", "#B28647", "#0072B2")
    for channel in sorted({item["channel"] for item in groups}):
        cohort = sorted((item for item in groups if item["channel"] == channel),
                        key=lambda item: DATASETS.index(item["dataset"]))
        figure, axes = plt.subplots(1, len(cohort), squeeze=False, figsize=(3.05 * len(cohort), 3.6), sharey=True)
        for ax, item in zip(axes[0], cohort):
            labels = list(METHODS)
            heights = [100 * item["methods"][label]["test_accuracy"]["mean"] for label in labels]
            for index, (label, height, color) in enumerate(zip(labels, heights, colors)):
                perfect = METHODS[label][1]
                ax.bar(index, height, color="white" if perfect else color, edgecolor=color,
                       hatch="///" if perfect else None, linewidth=1.2)
            ax.set_xticks(range(len(labels)), ["ERM", "L2", "T1", "T2", "ERM", "L2", "Prop."], fontsize=8)
            ax.tick_params(axis="x", length=0)
            ax.text(1.5, -.13, "Matched", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
            ax.text(5, -.13, "Perfect", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
            ax.axvline(3.5, color="#CCCCCC", linewidth=.8)
            ax.set_title(f"{DATASET_NAMES[item['dataset']]} ($m={item['m']}$)")
            ax.set_ylim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Deployment accuracy (%)")
        figure.tight_layout()
        for extension in ("png", "pdf", "eps"):
            figure.savefig(root / f"{channel}_deployment_accuracy.{extension}", dpi=220, bbox_inches="tight")
        plt.close(figure)


def report(manifest, root):
    entries = stage_entries(manifest)
    stage = next((stage for stage in CONFIRMATION_COHORTS if entries[0]["stage"] == STAGES[stage]), None)
    if stage is None:
        raise ValueError("An accuracy report requires a confirmation manifest")
    seeds = CONFIRMATION_COHORTS[stage]
    selection_path = manifest.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    if entries != confirmation_entries(selection["selected_variants"], selection["eval_draws"], stage):
        raise ValueError("Confirmation manifest does not match the frozen validation selection")
    rows = [validate_test(item, row, root) for item, row in zip(entries, read_results(root, entries))]
    groups = group_results(rows, seeds)
    summary = dict(manifest=file_fingerprint(manifest), selection=file_fingerprint(selection_path), confirmation_seeds=seeds,
                   trials=len(rows), groups=groups,
                   all_groups_t1_or_t2_strictly_highest=all(group["t1_or_t2_strictly_highest"] for group in groups))
    if stage == "refined_confirm":
        summary["confirmation_sampling"] = selection["confirmation_sampling"]
    save_json(root / "accuracy_summary.json", summary)
    write_csv(root / "accuracy_by_seed.csv", rows,
              ["dataset", "channel", "m", "method_label", "seed", "variant_id", "validation_accuracy",
               "test_accuracy", "test_cross_entropy", "test_bounded_loss", "task_id"])
    means = [dict(dataset=group["dataset"], channel=group["channel"], m=group["m"], method=method,
                  mean_accuracy=metrics["test_accuracy"]["mean"], sd_accuracy=metrics["test_accuracy"]["sd"],
                  mean_bounded_loss=metrics["test_bounded_loss"]["mean"],
                  mean_cross_entropy=metrics["test_cross_entropy"]["mean"])
             for group in groups for method, metrics in group["methods"].items()]
    write_csv(root / "accuracy_means.csv", means,
              ["dataset", "channel", "m", "method", "mean_accuracy", "sd_accuracy", "mean_bounded_loss", "mean_cross_entropy"])
    plot_accuracy(groups, root)
    return summary


def scope(value, allowed):
    items = tuple(value.split(","))
    if len(set(items)) != len(items) or not set(items) <= set(allowed):
        raise ValueError(f"Require unique names in {', '.join(allowed)}")
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--datasets", default=",".join(IMAGE_DATASETS))
    screen.add_argument("--channels", default="full_fading")
    screen.add_argument("--m", type=int, default=2000)
    screen.add_argument("--pilot", action="store_true")
    refine = commands.add_parser("refine-screen")
    refine.add_argument("--screen-manifest", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    for name in ("task", "report"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
    task = commands.choices["task"]
    task.add_argument("--index", type=int)
    task.add_argument("--threads", type=int, default=2)
    task.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--screen-manifest", type=Path, required=True)
    confirm.add_argument("--screen-root", type=Path, required=True)
    confirm.add_argument("--refinement-manifest", type=Path)
    confirm.add_argument("--refinement-root", type=Path)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--eval-draws", type=int, default=20)
    args = parser.parse_args()

    if args.command == "screen":
        entries = screening_entries(scope(args.datasets, DATASETS), scope(args.channels, CHANNELS), args.m, args.pilot)
        result = write_manifest(args.output, entries)
    elif args.command == "refine-screen":
        entries = stage_entries(args.screen_manifest, "screen")
        result = write_manifest(args.output, refinement_entries(screen_scope(entries)))
    elif args.command == "task":
        if args.threads < 1:
            parser.error("--threads must be positive")
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_task(args.manifest, args.output_root, index, args.device, args.threads)
    elif args.command == "confirm":
        if args.eval_draws < 1:
            parser.error("--eval-draws must be positive")
        result = prepare_confirmation(args.screen_manifest, args.screen_root, args.output, args.eval_draws,
                                      args.refinement_manifest, args.refinement_root)
    else:
        result = report(args.manifest, args.output_root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
