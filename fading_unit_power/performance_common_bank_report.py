"""Combine fixed-bank unit-power fading and archived erasure evaluations."""

import argparse
import csv
import io
import math
from pathlib import Path
import re
import statistics

from . import performance
from . import performance_seed_sweep as seed_sweep
from . import performance_sweep as sweep


ALLOWED_BANK_SIZES = (128, 200, 2000)
STEM = "common_bank_comparison"
HISTORICAL = performance.historical_table
PROTOCOLS = {
    "Fading": "One validation-selected shared training seed with 128 held-out deployment draws per method.",
    "Erasure": "Original matched erasure results averaged across paired training seeds, with 20 draws per fit.",
}


def valid_accuracy(value):
    return type(value) in (int, float) and 0 <= value <= 100


def valid_fingerprints(record):
    return set(record) == {"weights", "metadata"} and all(
        re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        and type(value["bytes"]) is int and value["bytes"] > 0
        for value in record.values()
    )


def comparison_row(dataset, channel, bank_size, seeds, draws, examples, scores, proposed):
    if set(scores) != set(sweep.METHODS) or not all(map(valid_accuracy, scores.values())):
        raise ValueError(f"Invalid method accuracies for {dataset}/{channel}")
    baseline = max(sweep.BASELINES, key=scores.__getitem__)
    gap = scores[proposed] - scores[baseline]
    return dict(dataset=dataset, channel=channel, protocol=PROTOCOLS[channel], bank_size=bank_size,
                training_seeds=seeds, test_draws=draws, test_examples=examples, accuracy_pct=scores,
                proposed=proposed, strongest_baseline=baseline, gap_pp=gap, beats_both_baselines=gap > 0)


def fading_row(root, dataset, bank_size):
    cohort = root / dataset
    path = seed_sweep.selection_path(cohort, dataset)
    selection = performance.read_json(path)
    seed_sweep.validate_bank(selection, dataset)
    summary = performance.read_json(cohort / f"seed_sweep_summary-{dataset}.json")
    if set(summary["datasets"]) != {dataset}:
        raise ValueError(f"Expected one fading dataset in {cohort}")
    result = summary["datasets"][dataset]
    selected = selection["selected"]
    seed, proposed, baseline = selected["seed"], selected["proposed"], selected["baseline"]
    digest = sweep.sha256(path)
    if (selection["dataset"] != dataset or type(seed) is not int or
            selected["m"] != bank_size or result["bank_size"] != bank_size or
            summary.get("fixed_bank_size", bank_size) != bank_size or
            (result["training_seed"], result["validation_proposed"], result["validation_baseline"],
             result["validation_gap_pp"], result["selection_sha256"]) !=
            (seed, proposed, baseline, selected["validation_gap_pp"], digest)):
        raise ValueError(f"Fading summary differs from the frozen selection at m={bank_size} for {dataset}")

    tasks = {task["entry"]["method"]: task for task in selection["tasks"]}
    methods = set(sweep.METHODS)
    winners = selected["winners"]
    if (len(selection["tasks"]) != len(tasks) or set(tasks) != methods or
            set(result["methods"]) != methods or set(winners) != methods or
            len({task["entry"]["task_id"] for task in tasks.values()}) != len(methods)):
        raise ValueError(f"Incomplete or duplicated fading methods for {dataset}")
    rank = lambda method: seed_sweep.rank(winners[method])
    if (proposed != min(sweep.PROPOSED, key=rank) or baseline != min(sweep.BASELINES, key=rank) or
            not math.isclose(selected["validation_gap_pp"],
                             100 * (winners[proposed]["accuracy"] - winners[baseline]["accuracy"]),
                             abs_tol=1e-10)):
        raise ValueError(f"Invalid validation winner for {dataset}")

    scores, trials, examples = {}, {}, set()
    for method in sweep.METHODS:
        task, recorded = tasks[method], result["methods"][method]
        entry = task["entry"]
        config, evaluation = entry["config"], entry["evaluation"]
        archive_config, _ = sweep.archive_config(dataset, method)
        if ((entry["phase"], config["dataset"], config["method"], config["m"], config["seed"]) !=
            ("confirmation", dataset, sweep.METHOD_OBJECTIVE[method], bank_size, seed) or
                config["train_channel"] != performance.FADING or
                config["deployment_channel"] != performance.FADING or
                config["transmit_unit_power"] is not True or
                any(config[field] != archive_config[field] for field in ("architecture", "width", "predictor")) or
                evaluation != dict(eval_draws=seed_sweep.TEST_DRAWS,
                                   eval_seed=seed + seed_sweep.TEST_EVAL_OFFSET)):
            raise ValueError(f"Invalid matched unit-power fading configuration for {dataset}/{method}")
        fingerprints = task["checkpoint_fingerprints"]
        if (recorded["task_id"] != entry["task_id"] or recorded["task_id"] != winners[method]["task_id"] or
                recorded["selected_epoch"] != task["selected_epoch"] or
                recorded["test_draws"] != seed_sweep.TEST_DRAWS or
                not math.isclose(recorded["validation_accuracy_pct"], 100 * winners[method]["accuracy"],
                                 abs_tol=1e-10) or
                recorded["checkpoint_fingerprints"] != fingerprints or not valid_fingerprints(fingerprints)):
            raise ValueError(f"Fading result differs from the selected checkpoint for {dataset}/{method}")
        examples.add(recorded["test_examples"])
        scores[method] = recorded["test_accuracy_pct"]
        trials[method] = dict(task_id=entry["task_id"], checkpoint_fingerprints=fingerprints)
    if len(examples) != 1 or any(type(count) is not int or count <= 0 for count in examples):
        raise ValueError(f"Unpaired fading test example counts for {dataset}")
    row = comparison_row(dataset, "Fading", bank_size, [seed], seed_sweep.TEST_DRAWS, examples.pop(), scores, proposed)
    expected_gaps = {method: scores[proposed] - scores[method] for method in sweep.BASELINES}
    gaps = result["test_gaps_pp"]
    if (gaps.keys() != expected_gaps.keys() or
            any(not math.isclose(gaps[key], value, abs_tol=1e-10) for key, value in expected_gaps.items()) or
            result["beats_both_baselines"] is not row["beats_both_baselines"] or
            summary["all_reported_datasets_have_validation_selected_proposed_wins"] is not row["beats_both_baselines"]):
        raise ValueError(f"Stored fading baseline gaps disagree with held-out accuracies for {dataset}")
    return {**row, "validation_gap_pp": selected["validation_gap_pp"], "selection_sha256": digest,
            "methods": trials}


def erasure_rows(bank_size):
    index = performance.read_json(HISTORICAL.DATA / "provenance.json")
    if index["bank_size"] != bank_size:
        raise ValueError(f"Erasure archive uses m={index['bank_size']}, requested m={bank_size}")
    observations = [row for row in HISTORICAL.seed_observations(index)
                    if row["channel"] == "Erasure" and row["method"] in sweep.METHODS]
    if any(row["m"] != bank_size for row in observations):
        raise ValueError(f"Erasure test report differs from common bank size {bank_size}")
    archive_sha = index["files"][str(HISTORICAL.REPORTS.relative_to(HISTORICAL.ROOT))]["sha256"]
    rows = {}
    for dataset in sweep.DATASETS:
        cohort = [row for row in observations if row["dataset"] == dataset]
        seeds = sorted({row["seed"] for row in cohort})
        if (len(seeds) < 2 or len(cohort) != len(seeds) * len(sweep.METHODS) or
                {(row["method"], row["seed"]) for row in cohort} !=
                {(method, seed) for method in sweep.METHODS for seed in seeds} or
                {row["draws"] for row in cohort} != {20} or len({row["examples"] for row in cohort}) != 1):
            raise ValueError(f"Incomplete paired erasure cohort for {dataset} at m={bank_size}")
        scores = {method: statistics.fmean(row["test_accuracy_pct"] for row in cohort if row["method"] == method)
                  for method in sweep.METHODS}
        proposed = max(sweep.PROPOSED, key=scores.__getitem__)
        row = comparison_row(dataset, "Erasure", bank_size, seeds, 20, cohort[0]["examples"], scores, proposed)
        rows[dataset] = {**row, "archive_sha256": archive_sha}
    return rows


def render_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["dataset", "channel", "bank_size", "protocol", "training_seeds", "test_draws",
                     "test_examples", *(method.split("-")[0] + "_accuracy_pct" for method in sweep.METHODS),
                     "proposed", "strongest_baseline", "gap_pp", "beats_both_baselines"])
    for row in rows:
        writer.writerow([row[key] for key in ("dataset", "channel", "bank_size", "protocol")] +
                        [" ".join(map(str, row["training_seeds"])), row["test_draws"], row["test_examples"]] +
                        [row["accuracy_pct"][method] for method in sweep.METHODS] +
                        [row[key] for key in ("proposed", "strongest_baseline", "gap_pp", "beats_both_baselines")])
    return stream.getvalue()


def render_markdown(bank_size, rows):
    lines = [
        f"# Common channel-bank size m = {bank_size}", "",
        "Fading uses matched diagonal fading with correlation 0.35 and unit transmit power. "
        "Each row reports 128 held-out deployment draws for the validation-selected seed. "
        "Erasure uses matched independent erasure at probability 0.50 under the original transmit-power protocol. "
        "Each erasure row reports a paired training-seed mean with 20 deployment draws per fit.", "",
        "Accuracy is in percent. The gap subtracts the stronger test baseline from the validation-selected "
        "proposed method for fading or the stronger proposed mean for erasure.", "",
        "| Dataset | Channel | Training seeds | ERM | L2 | T1 | T2 | Proposed | Gap (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        cells = [HISTORICAL.DATASETS[row["dataset"]], row["channel"], str(len(row["training_seeds"]))]
        cells += [f'{row["accuracy_pct"][method]:.3f}' for method in sweep.METHODS]
        cells += [row["proposed"].split("-")[0], f'{row["gap_pp"]:+.3f}']
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summarize(bank_size, fading_root, output):
    erasure = erasure_rows(bank_size)
    rows = [row for dataset in sweep.DATASETS
            for row in (fading_row(fading_root, dataset, bank_size), erasure[dataset])]
    report = dict(bank_size=bank_size, all_eight_wins=all(row["beats_both_baselines"] for row in rows), rows=rows)
    performance.write_json(output / f"{STEM}.json", report)
    performance.write_text(output / f"{STEM}.csv", render_csv(rows))
    performance.write_text(output / f"{STEM}.md", render_markdown(bank_size, rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-size", type=int, choices=ALLOWED_BANK_SIZES, required=True)
    parser.add_argument("--fading-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.bank_size, args.fading_root, args.output)
    print(f"Reported {len(report['rows'])} matched comparisons at m={args.bank_size} "
          f"({sum(row['beats_both_baselines'] for row in report['rows'])} positive gaps).")


if __name__ == "__main__":
    main()
