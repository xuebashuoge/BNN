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
PILOT_SEEDS = (92000,)
STAGES = {"pilot": "full_fading_accuracy_pilot", "screen": "full_fading_accuracy_screen",
          "confirm": "full_fading_accuracy_confirm"}
TEST_SEED_OFFSET = 1000003
EXECUTION_SETTINGS = {"device", "threads"}
GAUSSIAN_BASE_CHANNEL = {"full_fading": "analog", "outage": "outage"}


def variants(config, method, perfect):
    """Four candidates per algorithm and training regime."""
    config = {**config, "method": method}
    if method == "erm":
        for lr in (.0003, .001, .003, .01):
            yield dict(config, lr=lr)
        return
    for lr in (.001, .003):
        for strength in (0, 1):
            candidate = dict(config, lr=lr)
            if method == "l2":
                candidate["l2"] = (1e-7, 1e-5)[strength]
            elif perfect:
                candidate.update(eta_channel=0., eta_data=(.01, .1)[strength])
            else:
                scale = (.001, .01)[strength]
                ratio = math.sqrt((config["train_size"] - 1) / (config["m"] - 1))
                if method == "proposed":
                    candidate.update(eta_channel=scale, eta_data=scale * ratio)
                else:
                    candidate.update(eta_channel=0., eta_data=2 * scale / math.sqrt(1 + 1 / ratio**2))
            yield candidate


def entry(config, seed, group, label, stage):
    item = make_entry(config, seed, group, label, STAGES[stage])
    item["family"] = "proposed" if config["method"] in {"proposed", "joint"} else "baseline"
    return item


def screening_entries(datasets, channels, m, pilot):
    if m < 2:
        raise ValueError("A matched training bank requires m >= 2")
    stage, seeds = ("pilot", PILOT_SEEDS) if pilot else ("screen", SCREEN_SEEDS)
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
                candidates = list(variants(dict(base, train_channel=training), method, perfect))
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
        if item["stage"] == STAGES["confirm"]:
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


def screen_scope(entries):
    """Recover scope from the manifest, then regenerate the complete design."""
    groups = defaultdict(list)
    for item in entries:
        groups[item["group"]].append(item)
    for group, items in groups.items():
        sample = items[0]["config"]
        expected = screening_entries((sample["dataset"],), (sample["deployment_channel"]["kind"],),
                                     sample["m"], pilot=False)
        if (len(items) != len(expected)
                or {item["task_id"]: item for item in items} != {item["task_id"]: item for item in expected}):
            raise ValueError(f"Incomplete or altered comparison validation grid: {group}")


def confirmation_entries(selected, eval_draws):
    final = []
    for choice in selected:
        config = {**choice["config"], "eval_draws": eval_draws}
        final.extend({**entry(config, seed, choice["group"], choice["method_label"], "confirm"),
                      "selected_from_validation_variant": choice["variant_id"]} for seed in CONFIRM_SEEDS)
    return final


def prepare_confirmation(screen_manifest, screen_root, output, eval_draws):
    entries = stage_entries(screen_manifest, "screen")
    screen_scope(entries)
    rows = read_results(screen_root, entries)
    for row in rows:
        if any(metric in row for metric in ("test_accuracy", "test_cross_entropy", "test_status")):
            raise ValueError("Confirmation selection requires validation-only training results")
        if (not all(math.isfinite(row[metric]) for metric in ("validation_accuracy", "validation_cross_entropy"))
                or not 0 <= row["validation_accuracy"] <= 1 or row["validation_cross_entropy"] < 0):
            raise ValueError(f"Invalid validation metric: {row['task_id']}")
    summaries, selected = select_variants(rows, entries, top=1)
    final = confirmation_entries(selected, eval_draws)
    selection = dict(screen_manifest=file_fingerprint(screen_manifest), screening_seeds=SCREEN_SEEDS,
                     confirmation_seeds=CONFIRM_SEEDS, eval_draws=eval_draws,
                     rule="Highest mean validation accuracy, then lowest mean validation CE, then variant ID",
                     validation_trials=sorted(summaries, key=lambda row: (row["group"], row["method_label"], row["variant_id"])),
                     selected_variants=selected)
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


def group_results(rows):
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
        if set(methods) != set(METHODS) or any(set(runs) != set(CONFIRM_SEEDS) for runs in methods.values()):
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
                                             methods[baseline][seed]["test_accuracy"] for seed in CONFIRM_SEEDS)
                           for baseline in ("ERM-Matched", "L2-Matched")}
                  for method in ("T1-Matched", "T2-Matched")}
        sample = methods["ERM-Matched"][CONFIRM_SEEDS[0]]
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
    entries = stage_entries(manifest, "confirm")
    selection_path = manifest.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    if entries != confirmation_entries(selection["selected_variants"], selection["eval_draws"]):
        raise ValueError("Confirmation manifest does not match the frozen validation selection")
    rows = [validate_test(item, row, root) for item, row in zip(entries, read_results(root, entries))]
    groups = group_results(rows)
    summary = dict(manifest=file_fingerprint(manifest), selection=file_fingerprint(selection_path), confirmation_seeds=CONFIRM_SEEDS,
                   trials=len(rows), groups=groups,
                   all_groups_t1_or_t2_strictly_highest=all(group["t1_or_t2_strictly_highest"] for group in groups))
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
    confirm.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--eval-draws", type=int, default=20)
    args = parser.parse_args()

    if args.command == "screen":
        entries = screening_entries(scope(args.datasets, DATASETS), scope(args.channels, CHANNELS), args.m, args.pilot)
        result = write_manifest(args.output, entries)
    elif args.command == "task":
        if args.threads < 1:
            parser.error("--threads must be positive")
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_task(args.manifest, args.output_root, index, args.device, args.threads)
    elif args.command == "confirm":
        if args.eval_draws < 1:
            parser.error("--eval-draws must be positive")
        result = prepare_confirmation(args.screen_manifest, args.screen_root, args.output, args.eval_draws)
    else:
        result = report(args.manifest, args.output_root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
