"""Export deployment accuracy for the complete ten-seed paired replication."""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from tempfile import TemporaryDirectory

from results.final_chosen_results.rebuild_performance_table import DATA, seed_observations
from .paired_accuracy_replication import (
    CHANNELS, DATASETS, METHODS, SEEDS, build_entries, validate_evaluation,
)


PROPOSED = ("T1-Matched", "T2-Matched")
BASELINES = ("ERM-Matched", "L2-Matched")


def collect_scores(root):
    entries = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
               if line.strip()]
    if entries != build_entries():
        raise ValueError("The replication manifest differs from the frozen design")

    scores = {}
    for entry in entries:
        task_id, config = entry["task_id"], entry["config"]
        key = (config["dataset"], entry["channel"], entry["method_label"], config["seed"])
        if key in scores:
            raise ValueError(f"Duplicate paired observation for {key}")
        report = validate_evaluation(entry, root / "runs" / task_id)
        mc = report["heldout_gibbs_mc"]
        examples, draws = mc["examples"], mc["draws"]
        # The evaluator defines each replicate's accuracy as correct / examples.
        # Recover its integer count so exact ties are preserved when pairing seeds.
        counts = [rep["accuracy"] * examples for rep in mc["replicate_statistics"]]
        if len(counts) != draws or any(not math.isclose(n, round(n), rel_tol=0, abs_tol=1e-7)
                                      for n in counts):
            raise ValueError(f"Invalid Gibbs replicate counts in {task_id}")
        correct = sum(round(n) for n in counts)
        accuracy = correct / (draws * examples)
        if not math.isclose(mc["accuracy"], accuracy, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Reported mean accuracy disagrees with replicate counts in {task_id}")
        scores[key] = dict(dataset=key[0], channel=key[1], method=key[2], seed=key[3],
                           m=config["m"], task_id=task_id, examples=examples, draws=draws,
                           total_correct=correct, test_accuracy_pct=100 * accuracy,
                           weights_sha256=report["checkpoint_source_fingerprints"]["weights"]["sha256"])
    if set(scores) != set(product(DATASETS, CHANNELS, METHODS, SEEDS)):
        raise ValueError("Expected all five methods on every paired seed in all eight settings")
    return scores


def archived_scores():
    index = json.loads((DATA / "provenance.json").read_text())
    archived = {}
    for row in seed_observations(index):
        by_seed = archived.setdefault((row["dataset"], row["channel"], row["method"]), {})
        if row["seed"] in by_seed:
            raise ValueError(f"Duplicate archived observation for {row['task_id']}")
        by_seed[row["seed"]] = row["test_accuracy_pct"]
    return archived


def summarize(scores, archived):
    methods, contrasts = [], []
    for dataset, channel in product(DATASETS, CHANNELS):
        group = {(method, seed): scores[dataset, channel, method, seed]
                 for method, seed in product(METHODS, SEEDS)}
        budgets = {(r["draws"], r["examples"]) for r in group.values()}
        if len(budgets) != 1:
            raise ValueError(f"Different evaluation budgets in {dataset} {channel}")
        draws, examples = budgets.pop()
        n_predictions = draws * examples
        counts = {key: row["total_correct"] for key, row in group.items()}
        setting = dict(dataset=dataset, channel=channel, n_seeds=len(SEEDS))
        for method in METHODS:
            accuracies = [group[method, seed]["test_accuracy_pct"] for seed in SEEDS]
            wins_both = None
            if method in PROPOSED:
                wins_both = sum(all(counts[method, seed] > counts[b, seed] for b in BASELINES)
                                for seed in SEEDS)
            methods.append(dict(**setting, method=method, mean_accuracy_pct=fmean(accuracies),
                                sd_accuracy_pct=stdev(accuracies), wins_over_both_baselines=wins_both))
        for proposed, baseline in product(PROPOSED, BASELINES):
            deltas = [counts[proposed, seed] - counts[baseline, seed] for seed in SEEDS]
            gaps = [100 * delta / n_predictions for delta in deltas]
            mean_gap = fmean(gaps)
            previous_p = archived[dataset, channel, proposed]
            previous_b = archived[dataset, channel, baseline]
            if previous_p.keys() != previous_b.keys() or previous_p.keys() & set(SEEDS):
                raise ValueError(f"Unpaired or overlapping archived seeds for {dataset} {channel}")
            old_gap = fmean(previous_p[s] - previous_b[s] for s in previous_p)
            contrasts.append(dict(**setting, proposed=proposed, baseline=baseline, mean_gap_pp=mean_gap,
                                  sd_gap_pp=stdev(gaps), min_gap_pp=min(gaps), max_gap_pp=max(gaps),
                                  wins=sum(d > 0 for d in deltas), ties=deltas.count(0),
                                  losses=sum(d < 0 for d in deltas), archived_n_seeds=len(previous_p),
                                  archived_mean_gap_pp=old_gap, change_from_archived_gap_pp=mean_gap - old_gap))
    return methods, contrasts


def latex_table(caption, label, columns, cells):
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}",
             r"\label{tab:" + label + "}", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll"
             + "c" * len(columns) + "@{}}", r"\toprule",
             " & ".join(["Dataset", "Channel", *columns]) + r" \\", r"\midrule"]
    for dataset, channel in product(DATASETS, CHANNELS):
        lines.append(" & ".join([DATASETS[dataset], channel, *cells[dataset, channel]]) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])


def export(root):
    scores = collect_scores(root)
    methods, contrasts = summarize(scores, archived_scores())
    accuracy, differences = {}, {}
    for row in methods:
        accuracy.setdefault((row["dataset"], row["channel"]), []).append(
            rf'\({row["mean_accuracy_pct"]:.2f}\pm{row["sd_accuracy_pct"]:.2f}\)')
    for row in contrasts:
        differences.setdefault((row["dataset"], row["channel"]), []).append(
            rf'\({row["mean_gap_pp"]:+.3f}\pm{row["sd_gap_pp"]:.3f}\)')
    tables = {
        "performance_table.tex": latex_table(
            r"Deployment accuracy (\%) in the replication at channel-set size $m=2000$. "
            r"Each entry gives the mean and sample standard deviation over ten paired training seeds. "
            r"Perfect denotes T1/T2 trained on the perfect channel.",
            "replication-accuracy", ["ERM", "L2", "T1", "T2", "Perfect"], accuracy),
        "paired_differences_table.tex": latex_table(
            r"Paired differences in deployment accuracy (percentage points) for matched training at "
            r"$m=2000$. Each entry gives the mean and sample standard deviation of the ten "
            r"within-seed differences, with the first method minus the second.",
            "replication-differences", [r"T1 $-$ ERM", r"T1 $-$ L2", r"T2 $-$ ERM", r"T2 $-$ L2"],
            differences),
    }
    # Finish validation and stage every output before publishing individual files.
    output = root / "report"
    with TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary)
        for name, rows in [("deployment_by_seed.csv", list(scores.values())), ("method_summary.csv", methods),
                           ("paired_differences.csv", contrasts)]:
            with (stage / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        for name, content in tables.items():
            (stage / name).write_text(content)
        output.mkdir(exist_ok=True)
        for path in stage.iterdir():
            path.replace(output / path.name)
    print(f"Verified {len(scores)} trials. Exported {len(methods)} method summaries and "
          f"{len(contrasts)} paired contrasts to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="Replication root containing manifest.jsonl and runs/")
    export(parser.parse_args().output_root)
