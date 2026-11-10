"""Plot saved high-Monte-Carlo checkpoint inference; no training or inference.

Intervals condition on one selected training seed, each fixed checkpoint, and
the fixed test dataset. They quantify posterior/channel Monte Carlo error only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tit-high-mc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ORDER = ["ERM-RawCE-Matched", "ERM-Perfect", "ERM-Matched", "L2-Matched",
         "StochERM-Matched", "PAC-Perfect", "PAC-Matched", "T1-Matched", "T2-Matched"]
COLORS = {"ERM": "#737373", "L2": "#b89256", "StochERM": "#80a6b8",
          "PAC": "#32916e", "T1": "#2973b0", "T2": "#b55f98"}


def color(label):
    return next((COLORS[key] for key in ("StochERM", "T1", "T2", "PAC", "L2", "ERM")
                 if label.startswith(key)), "#737373")


def describe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Conditional Monte Carlo intervals require at least two finite draws")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    half = float(t.ppf(.975, len(values)-1)) * se
    return dict(n=len(values), mean=mean, sd=sd, standard_error=se,
                ci95_low=mean-half, ci95_high=mean+half, ci95_half_width=half)


def load_models(root, seed, group):
    rows = json.loads((root / "test_results.json").read_text())
    chosen = [row for row in rows if row.get("seed") == seed and row.get("group") == group]
    models, seen = [], set()
    for row in chosen:
        label = row["method_label"]
        if label in seen:
            raise ValueError(f"Multiple selected variants for {label}; specify a campaign with one validation-selected variant per method")
        seen.add(label)
        path = root / row["task_id"] / "test_high_mc" / "evaluation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing high-MC inference: {path}")
        report = json.loads(path.read_text())
        risk = report["heldout_gibbs_mc"]
        replicates = risk["replicate_statistics"]
        if len(replicates) != risk["draws"]:
            raise ValueError(f"Draw count mismatch for {label}")
        draws = np.array([replicate["accuracy"] for replicate in replicates], dtype=float)
        accuracy = describe(draws)
        if abs(accuracy["mean"] - risk["accuracy"]) > 1e-12:
            raise ValueError(f"Reported mean does not match saved replicate values for {label}")
        models.append(dict(method_label=label, family=row["family"], task_id=row["task_id"],
                           config=report["evaluation_config"], split=report["split"],
                           examples=risk["examples"], draws=risk["draws"],
                           posterior_rule=risk["posterior_rule"],
                           accuracy=accuracy, accuracy_draws=draws.tolist(),
                           source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not models:
        raise ValueError(f"No selected models for group={group}, training seed={seed}")
    common = {(item["config"]["eval_seed"], item["draws"], item["examples"], item["split"],
               item["config"]["dataset"], item["config"]["width"], item["config"]["batch_size"],
               json.dumps(item["config"]["deployment_channel"], sort_keys=True)) for item in models}
    if len(common) != 1:
        raise ValueError("Paired-draw comparison requires common evaluation seeds, draw counts, dataset, model dimensions, batch size and deployment channel")
    models.sort(key=lambda item: ORDER.index(item["method_label"]) if item["method_label"] in ORDER else len(ORDER))
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/journal_campaign/screen"))
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--group", default="mnist_analog")
    parser.add_argument("--output", type=Path, default=Path("results/journal_campaign/figures_high_mc"))
    args = parser.parse_args()
    models = load_models(args.root, args.seed, args.group)
    baselines = [item for item in models if item["family"] == "baseline"]
    proposals = [item for item in models if item["family"] == "proposed"]
    if not baselines or not proposals:
        raise ValueError("Need both proposed and baseline models")
    reference = max(baselines, key=lambda item: item["accuracy"]["mean"])
    candidate = max(proposals, key=lambda item: item["accuracy"]["mean"])
    reference_draws = np.array(reference["accuracy_draws"])
    for item in models:
        item["paired_accuracy_difference_vs_reference"] = describe(np.array(item["accuracy_draws"]) - reference_draws)
    gap = candidate["paired_accuracy_difference_vs_reference"]
    summary = dict(root=str(args.root.resolve()), group=args.group, selected_training_seed=args.seed,
                   scope="Exploratory selected training seed; fixed checkpoints and fixed test dataset; Monte Carlo uncertainty only",
                   interval_method="Pointwise two-sided 95% Student t intervals over independent full-loader posterior/channel draws",
                   paired_difference_method="Within-replicate differences under common evaluation random-number streams",
                   uncertainty_excluded=["training-seed variation", "dataset sampling", "hyperparameter/seed/environment selection", "multiple comparisons"],
                   evaluation_seed=models[0]["config"]["eval_seed"], draws_per_model=models[0]["draws"],
                   examples_per_draw=models[0]["examples"],
                   strongest_baseline=reference["method_label"], best_proposed=candidate["method_label"],
                   proposed_has_highest_point_estimate=gap["mean"] > 1e-12,
                   proposed_minus_strongest_baseline=gap,
                   conditional_mc_interval_excludes_zero=gap["ci95_low"] > 0,
                   models=models)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "high_mc_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    csv_rows = []
    for item in models:
        acc, margin = item["accuracy"], item["paired_accuracy_difference_vs_reference"]
        csv_rows.append(dict(method=item["method_label"], family=item["family"], training_seed=args.seed,
                             draws=item["draws"], accuracy_percent=acc["mean"]*100,
                             accuracy_mc_ci95_low_percent=acc["ci95_low"]*100,
                             accuracy_mc_ci95_high_percent=acc["ci95_high"]*100,
                             reference=reference["method_label"], paired_margin_pp=margin["mean"]*100,
                             paired_mc_ci95_low_pp=margin["ci95_low"]*100,
                             paired_mc_ci95_high_pp=margin["ci95_high"]*100,
                             source=item["source"], source_sha256=item["source_sha256"]))
    with (args.output / "high_mc_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.7), gridspec_kw={"width_ratios": [1.3, 1.]})
    for position, item in enumerate(models):
        accuracy = item["accuracy"]
        axes[0].bar(position, accuracy["mean"]*100, width=.68, color=color(item["method_label"]), alpha=.83,
                    hatch="//" if "Perfect" in item["method_label"] else None)
        axes[0].errorbar(position, accuracy["mean"]*100, yerr=accuracy["ci95_half_width"]*100,
                         fmt="none", color="black", capsize=3, linewidth=1.2)
    axes[0].set(ylim=(0, 100), ylabel="Deployment test accuracy (%)", title="All selected models, including perfect training")
    axes[0].set_xticks(range(len(models)), [item["method_label"].replace("-", "\n", 1) for item in models], rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    matched = [item for item in models if "Perfect" not in item["method_label"]]
    for position, item in enumerate(matched):
        difference = item["paired_accuracy_difference_vs_reference"]
        axes[1].errorbar(difference["mean"]*100, position, xerr=difference["ci95_half_width"]*100,
                         fmt="o", markersize=6, color=color(item["method_label"]), capsize=3, linewidth=1.5)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set_yticks(range(len(matched)), [item["method_label"] for item in matched])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Paired accuracy difference (percentage points)",
                title=f"Matched-trained models versus {reference['method_label']}")
    axes[1].grid(axis="x", alpha=.2)
    fig.suptitle(f"MNIST severe fading + AWGN — selected training seed {args.seed}", fontsize=14)
    footer = (f"Exploratory selected training seed {args.seed}; {models[0]['draws']} fresh Gibbs/channel draws per fixed checkpoint\n"
              "Whiskers: conditional 95% Monte Carlo t intervals only; fixed test set; training-seed and selection uncertainty excluded")
    fig.text(.5, -.025, footer, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"high_mc_seed{args.seed}.{extension}", dpi=220 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"models": len(models), "draws": models[0]["draws"], "best_proposed": candidate["method_label"],
                      "strongest_baseline": reference["method_label"], "paired_margin_pp": gap["mean"]*100,
                      "paired_mc_ci95_pp": [gap["ci95_low"]*100, gap["ci95_high"]*100],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
