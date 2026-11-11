"""Summarize completed CIFAR campaigns without training or inference.

Only aggregate-validation-selected configurations are included in the table.
Every evaluation seed for each selected configuration is retained. Source
hashes allow the paper assets to be traced to the saved inference reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


METHODS = ["ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched",
           "ERM-Perfect", "L2-Perfect", "PAC-Perfect"]
LABELS = {"ERM-Matched": "ERM (matched)", "L2-Matched": "L2 (matched)",
          "T1-Matched": "Proposed T1 (matched)", "T2-Matched": "Proposed T2 (matched)",
          "ERM-Perfect": "ERM (perfect)", "L2-Perfect": "L2 (perfect)",
          "PAC-Perfect": "Proposed (perfect)"}
COLORS = dict(zip(METHODS, ["#777777", "#9b6a46", "#0072b2", "#009e73",
                           "#aaaaaa", "#c59c76", "#cc79a7"]))


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None)


def load_campaign(root, sources):
    root = Path(root)

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    summary = read(root / "summary.json")
    if summary["completed_trials"] != summary["training_trials"] or summary["failed_evaluations"]:
        raise ValueError(f"Incomplete campaign: {root}")
    selected = {task for choice in summary["selected_variants"] if choice.get("rank", 1) == 1
                for task in choice["task_ids"]}
    rows = [row for row in read(root / "test_results.json") if row["task_id"] in selected]
    if len(rows) != len(selected) or any(row["test_status"] != "complete" for row in rows):
        raise ValueError("Every selected seed must have a completed evaluation")
    if {row["method_label"] for row in rows} != set(METHODS):
        raise ValueError("Expected all seven matched/perfect methods")
    result = []
    for method in METHODS:
        runs = sorted((row for row in rows if row["method_label"] == method), key=lambda row: row["seed"])
        if len({run["variant_id"] for run in runs}) != 1:
            raise ValueError("Do not pool distinct hyperparameter configurations")
        observations = []
        for run in runs:
            directory = root / run["task_id"]
            report = read(directory / "test_deployment" / "evaluation.json")
            heldout = report["heldout_gibbs_mc"]
            if abs(heldout["accuracy"] - run["test_accuracy"]) > 1e-12:
                raise ValueError("Summary and frozen evaluation disagree")
            if heldout["draws"] != 10 or not 0 <= heldout["bounded_loss"] <= 1:
                raise ValueError("Expected common ten-draw evaluation and bounded risk")
            observations.append(dict(seed=run["seed"], task_id=run["task_id"],
                                     accuracy=heldout["accuracy"], bounded_loss=heldout["bounded_loss"],
                                     cross_entropy=heldout["cross_entropy"],
                                     feature_power_per_coordinate=heldout["feature_power_per_coordinate"],
                                     selected_epoch=run["selected_epoch"],
                                     history=read(directory / "history.json")))
        result.append(dict(method=method, display=LABELS[method], config=runs[0]["config"],
                           accuracy=stats(x["accuracy"] for x in observations),
                           bounded_loss=stats(x["bounded_loss"] for x in observations),
                           cross_entropy=stats(x["cross_entropy"] for x in observations),
                           observations=observations))
    seed_sets = {tuple(x["seed"] for x in item["observations"]) for item in result}
    if len(seed_sets) != 1:
        raise ValueError("Methods must retain the same seed set")
    return result


def tex_metric(item, key, percent=False):
    scale, digits = (100, 2) if percent else (1, 4)
    metric = item[key]
    value = f"{metric['mean'] * scale:.{digits}f}"
    if metric["sd"] is not None:
        value += rf"\pm{metric['sd'] * scale:.{digits}f}"
    return "$" + value + "$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, default=Path("results/journal_cifar/report"))
    args = parser.parse_args()
    sources, stages = {}, {}
    for specification in args.campaign:
        name, path = specification.split("=", 1)
        if name in stages:
            raise ValueError("Duplicate campaign name")
        stages[name] = load_campaign(path, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, methods in stages.items():
        for item in methods:
            config = item["config"]
            records.append(dict(stage=stage, method=item["method"],
                                n_train=config["train_size"], epochs=config["epochs"],
                                seeds=[x["seed"] for x in item["observations"]],
                                accuracy_percent=100 * item["accuracy"]["mean"],
                                accuracy_sd_pp=100 * item["accuracy"]["sd"] if item["accuracy"]["sd"] is not None else None,
                                bounded_risk=item["bounded_loss"]["mean"],
                                bounded_risk_sd=item["bounded_loss"]["sd"],
                                selected_epochs=[x["selected_epoch"] for x in item["observations"]]))
    with (args.output / "cifar_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    tex = [r"\begin{table*}[t]", r"\centering\small",
           r"\caption{CIFAR-10 deployment performance under Gaussian fading and additive noise. Entries are mean $\pm$ sample standard deviation over all listed initialization/channel-bank seeds, using ten fresh Gibbs/channel draws per checkpoint. Hyperparameters and checkpoints use validation accuracy. Perfect refers to training only.}",
           r"\label{tab:sim_cifar}",
           r"\begin{tabular}{l" + "cc" * len(stages) + "}", r"\toprule"]
    tex.append("Method & " + " & ".join(r"\multicolumn{2}{c}{" + stage.replace("_", r"\_") + "}" for stage in stages) + r"\\")
    tex.append(" & " + " & ".join([r"Accuracy (\%) & Bounded risk"] * len(stages)) + r"\\\midrule")
    for index, method in enumerate(METHODS):
        cells = [LABELS[method]]
        for methods in stages.values():
            cells += [tex_metric(methods[index], "accuracy", True), tex_metric(methods[index], "bounded_loss")]
        tex.append(" & ".join(cells) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (args.output / "cifar_results.tex").write_text("\n".join(tex) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(stages), 2, figsize=(10, 3.3 * len(stages)), squeeze=False)
    for index, (stage, methods) in enumerate(stages.items()):
        for column, subset in enumerate((methods[:4], methods[4:])):
            ax = axes[index, column]
            for item in subset:
                trajectories = [[record["validation"]["accuracy"] for record in obs["history"]]
                                for obs in item["observations"]]
                means = [100 * statistics.fmean(vals) for vals in zip(*trajectories)]
                ax.plot(range(1, len(means) + 1), means, label=item["display"], color=COLORS[item["method"]])
            ax.set(title=f"{stage}: {'matched' if column == 0 else 'perfect'} training",
                   xlabel="Epoch", ylabel="Validation accuracy (%)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "cifar_validation.pdf", bbox_inches="tight")
    fig.savefig(args.output / "cifar_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (args.output / "cifar_results.json").write_text(json.dumps(dict(campaigns=stages, source_hashes=sources), indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
