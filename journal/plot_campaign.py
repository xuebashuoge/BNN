"""Plot saved journal campaign JSON only; never train or run checkpoint inference.

Each source campaign remains separate. Screening and test-based rankings are
explicitly exploratory; all available seed outcomes appear in the figures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tit-campaign-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "savefig.facecolor": "white"})
COLORS = {"ERM": "#737373", "L2": "#b89256", "StochERM": "#80a6b8",
          "PAC": "#32916e", "T1": "#2973b0", "T2": "#b55f98"}
METHOD_ORDER = {"ERM-RawCE-Matched": 0, "ERM-Perfect": 1, "ERM-Matched": 2,
                "L2-Matched": 3, "StochERM-Matched": 4, "PAC-Perfect": 5,
                "PAC-Matched": 6, "T1-Matched": 7, "T2-Matched": 8}


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def stats(values):
    values = [float(value) for value in values if finite(value)]
    if not values:
        return dict(n=0, mean=None, half=None)
    n, mean = len(values), statistics.fmean(values)
    if n > 1:
        from scipy.stats import t
        half = float(t.ppf(.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
    else:
        half = None
    return dict(n=n, mean=mean, half=half)


def color(label):
    for prefix in ("StochERM", "T1", "T2", "PAC", "L2", "ERM"):
        if label.startswith(prefix):
            return COLORS[prefix]
    return "#737373"


def method_order(label):
    return METHOD_ORDER.get(label, 20), label


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def find_report(root, row):
    candidates = [root / row["task_id"] / name / "evaluation.json"
                  for name in ("test_full", "test_deployment")]
    if row.get("test_report"):
        candidates.append(Path(row["test_report"]))
    for path in candidates:
        report = load_json(path)
        if isinstance(report, dict):
            return report, path
    return None, None


def read_campaign(root):
    rows = load_json(root / "test_results.json", [])
    summary = load_json(root / "summary.json", {})
    if not isinstance(rows, list):
        rows = []
    rows = [row for row in rows if row.get("test_status") == "complete" and finite(row.get("test_accuracy"))]
    selected = {task_id for item in summary.get("selected_variants", []) if item.get("rank", 1) == 1
                for task_id in item.get("task_ids", [])}
    if selected:
        rows = [row for row in rows if row["task_id"] in selected]
    seen, distinct = set(), []
    for row in rows:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        report, path = find_report(root, row)
        row = dict(row)
        row["_report"] = report
        row["_report_path"] = str(path) if path else None
        distinct.append(row)
    return distinct, summary


def save_figure(fig, output, name, description, index):
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    index["figures"].append(dict(name=name, description=description, files=paths))


def provenance_note(rows):
    stages = {row.get("stage") for row in rows}
    if stages == {"confirm"}:
        return "Independent-seed confirmation; configurations selected by preceding validation search"
    if stages <= {"mechanism", "mismatch"}:
        return "Controlled experiment; all saved seeds shown; pointwise 95% seed t intervals"
    if stages & {"refine", "full_refine"}:
        return "Exploratory refinement after earlier results; all saved seeds retained; sources are not pooled"
    if any(str(stage).startswith("fashion") for stage in stages):
        return "Exploratory FashionMNIST comparison with added controls; all saved seeds retained"
    return "Exploratory screening; validation-selected configurations; test rankings remain exploratory"


def errorbar(ax, x, summary, scale=1., **kwargs):
    if summary["n"]:
        ax.errorbar(x, summary["mean"] * scale,
                    yerr=(summary["half"] * scale if summary["half"] is not None else None),
                    capsize=3, **kwargs)


def performance_figures(rows, root_label, output, index):
    grouped = defaultdict(list)
    for row in rows:
        # Refinement/control stages use descriptive role names. They are still
        # performance comparisons; only mechanism/mismatch/reference roles
        # belong exclusively to the other figures.
        if row.get("role", "comparison") not in {"mechanism", "mismatch", "reference"}:
            grouped[row["group"]].append(row)
    for group, runs in sorted(grouped.items()):
        methods = defaultdict(list)
        variants_by_label = defaultdict(set)
        for row in runs:
            variants_by_label[row["method_label"]].add(row.get("variant_id", "unspecified"))
        for row in runs:
            label = row["method_label"]
            if len(variants_by_label[label]) > 1:
                label += " [" + row.get("variant_id", "unspecified")[:5] + "]"
            methods[label].append(row)
        labels = sorted(methods, key=method_order)
        if not labels:
            continue
        baseline_labels = [label for label in labels if methods[label][0]["family"] == "baseline"]
        baseline_label = max(baseline_labels, key=lambda label: statistics.fmean(row["test_accuracy"] for row in methods[label])) if baseline_labels else None
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.3, 1.]})
        for position, label in enumerate(labels):
            values = [row["test_accuracy"] for row in methods[label]]
            summary = stats(values)
            axes[0].bar(position, summary["mean"] * 100, width=.68, color=color(label), alpha=.8,
                        hatch="//" if "Perfect" in label else None)
            errorbar(axes[0], position, summary, 100, fmt="none", color="black", linewidth=1.2)
            jitter = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0]
            axes[0].scatter(position + np.asarray(jitter), np.asarray(values) * 100,
                            s=24, color="black", alpha=.48, zorder=4)
        axes[0].set(ylim=(0, 100), ylabel="Deployment test accuracy (%)", title="Mean accuracy and all seed outcomes")
        axes[0].set_xticks(range(len(labels)), [label.replace("-", "\n", 1) for label in labels], rotation=35, ha="right")
        axes[0].grid(axis="y", alpha=.2)
        axes[0].set_axisbelow(True)
        if baseline_label:
            reference = {row["seed"]: row["test_accuracy"] for row in methods[baseline_label]}
            # Perfect-trained references remain in the left panel. Omitting
            # them here keeps small matched-training differences legible.
            paired_labels = [label for label in labels if "Perfect" not in label] or labels
            for position, label in enumerate(paired_labels):
                paired = [row["test_accuracy"] - reference[row["seed"]] for row in methods[label] if row["seed"] in reference]
                summary = stats(paired)
                if summary["n"]:
                    errorbar(axes[1], position, summary, 100, fmt="o", color=color(label), markersize=6)
                    jitter = np.linspace(-.13, .13, len(paired)) if len(paired) > 1 else [0]
                    axes[1].scatter(position + np.asarray(jitter), np.asarray(paired) * 100,
                                    s=22, color=color(label), alpha=.35)
            axes[1].axhline(0, color="black", linewidth=.8)
            axes[1].set(ylabel="Paired accuracy difference (percentage points)",
                        title=f"Matched models versus {baseline_label}")
            axes[1].set_xticks(range(len(paired_labels)), [label.replace("-", "\n", 1) for label in paired_labels], rotation=35, ha="right")
            axes[1].grid(axis="y", alpha=.2)
        else:
            axes[1].text(.5, .5, "No evaluated baseline available", ha="center", transform=axes[1].transAxes)
        config = runs[0]["config"]
        fig.suptitle(f"{config['dataset'].upper()} — {group.replace('_', ' ')}", fontsize=14)
        fig.text(.5, -.025, provenance_note(runs) + "\nBars/dots: means; whiskers: pointwise 95% seed t intervals; no seed selection", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .03, 1, .95))
        save_figure(fig, output, f"{root_label}_{slug(group)}_accuracy", "Accuracy and paired comparisons", index)
        index["figures"][-1].update(method_labels=labels,
                                     evaluated_seed_counts={label: len(methods[label]) for label in labels},
                                     source_task_ids=[row["task_id"] for row in runs],
                                     stages=sorted({str(row.get("stage")) for row in runs}))


def observed(row, component, metric="bounded_loss"):
    report = row.get("_report") or {}
    value = report.get("signed_gap_decomposition_mc", {}).get(metric, {}).get(component)
    return value if finite(value) else None


def curve(ax, runs, parameter, value, label, linestyle="-", x_transform=None):
    grouped = defaultdict(list)
    for row in runs:
        x, y = parameter(row), value(row)
        if finite(x) and finite(y):
            grouped[x].append(y)
    if not grouped:
        return False
    x = sorted(grouped)
    summaries = [stats(grouped[item]) for item in x]
    means = np.array([item["mean"] for item in summaries])
    halves = np.array([item["half"] if item["half"] is not None else 0 for item in summaries])
    x_plot = np.array([x_transform(item) for item in x]) if x_transform else np.array(x)
    ax.plot(x_plot, means, marker="o", markersize=4, linewidth=1.6, color=color(label),
            linestyle=linestyle, label=label)
    ax.fill_between(x_plot, means-halves, means+halves, color=color(label), alpha=.13)
    for position, key in enumerate(x):
        ax.scatter(np.full(len(grouped[key]), x_plot[position]), grouped[key], s=16,
                   color=color(label), alpha=.35)
    return True


def mechanism_figures(rows, root_label, output, index):
    mechanism = [row for row in rows if row.get("role") == "mechanism"]
    references = [row for row in rows if row.get("role") == "reference"]
    if not mechanism:
        return
    # Separate model/dataset/deployment settings; never pool distinct experiments.
    settings = defaultdict(list)
    for row in mechanism:
        config = row["config"]
        key = (config["dataset"], config["width"], json.dumps(config["deployment_channel"], sort_keys=True))
        settings[key].append(row)
    for setting_index, (key, runs) in enumerate(sorted(settings.items())):
        n_values = {row["config"]["train_size"] for row in runs}
        fixed_n = max(n_values, key=lambda n: len({row["config"]["m"] for row in runs if row["config"]["train_size"] == n}))
        m_values = {row["config"]["m"] for row in runs}
        fixed_m = max(m_values, key=lambda m: len({row["config"]["train_size"] for row in runs if row["config"]["m"] == m}))
        for varying, fixed_key, fixed_value in (("m", "train_size", fixed_n), ("train_size", "m", fixed_m)):
            subset = [row for row in runs if row["config"][fixed_key] == fixed_value]
            if len({row["config"][varying] for row in subset}) < 2:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
            component = "channel_generalization" if varying == "m" else "data_generalization"
            labels = sorted({row["method_label"] for row in subset}, key=method_order)
            for label in labels:
                selected = [row for row in subset if row["method_label"] == label]
                curve(axes[0], selected, lambda row: row["config"][varying], lambda row: row["test_accuracy"] * 100, label)
                curve(axes[1], selected, lambda row: row["config"][varying], lambda row: observed(row, component), label)
            matched_refs = [row for row in references if row["config"]["dataset"] == key[0]
                            and row["config"]["width"] == key[1]
                            and json.dumps(row["config"]["deployment_channel"], sort_keys=True) == key[2]]
            for label in sorted({row["method_label"] for row in matched_refs}, key=method_order):
                selected = [row for row in matched_refs if row["method_label"] == label]
                if varying == "m":
                    selected = [row for row in selected if row["config"]["train_size"] == fixed_n]
                    summary = stats([row["test_accuracy"] * 100 for row in selected])
                    if summary["n"]:
                        axes[0].axhline(summary["mean"], color=color(label), linestyle="--", label=label)
                        if summary["half"] is not None:
                            axes[0].axhspan(summary["mean"]-summary["half"], summary["mean"]+summary["half"], color=color(label), alpha=.07)
                    # A perfect bank and fresh perfect channel are identical.
                    # The inferred signed channel gap is zero up to MC sampling.
                    axes[1].axhline(0, color=color(label), linestyle="--", alpha=.6)
                else:
                    curve(axes[0], selected, lambda row: row["config"][varying], lambda row: row["test_accuracy"] * 100, label, "--")
                    curve(axes[1], selected, lambda row: row["config"][varying], lambda row: observed(row, component), label, "--")
            for ax in axes:
                ax.set_xscale("log")
                ax.set_xlabel("Training channel-bank size m" if varying == "m" else "Training sample size n")
                ax.grid(alpha=.2)
            axes[0].set_ylabel("Deployment test accuracy (%)")
            axes[1].set_ylabel(f"Observed {component.replace('_', ' ')} gap")
            axes[1].axhline(0, color="black", linewidth=.7)
            if not any(finite(observed(row, component)) for row in subset):
                axes[1].text(.5, .5, "Full-risk reports unavailable", ha="center", transform=axes[1].transAxes)
            axes[0].legend(fontsize=8, ncol=2)
            fig.suptitle(f"{key[0].upper()}: varying {'m' if varying == 'm' else 'n'}, fixed {'n' if varying == 'm' else 'm'}={fixed_value}")
            fig.text(.5, -.025, "Normalized bounded-loss gaps; all saved seeds; pointwise 95% seed t intervals\nPerfect training is reused across m and retrained for each n", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .03, 1, .94))
            save_figure(fig, output, f"{root_label}_mechanism_{varying}_{setting_index}", "Finite-channel/data generalization mechanism", index)


def mismatch_figures(rows, root_label, output, index):
    mismatch = [row for row in rows if row.get("role") == "mismatch"]
    if not mismatch:
        return
    settings = defaultdict(list)
    for row in mismatch:
        cfg = row["config"]
        if cfg["deployment_channel"]["kind"] == "outage":
            settings[(cfg["dataset"], cfg["width"], cfg["train_size"], cfg["deployment_channel"]["outage_prob"])].append(row)
    for index_number, (key, runs) in enumerate(sorted(settings.items())):
        parameter = lambda row: row["config"]["train_channel"].get("outage_prob", 0.)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for label in sorted({row["method_label"] for row in runs}, key=method_order):
            selected = [row for row in runs if row["method_label"] == label]
            curve(axes[0], selected, parameter, lambda row: row["test_accuracy"] * 100, label)
            curve(axes[1], selected, parameter, lambda row: row["test_cross_entropy"], label)
        for ax in axes:
            ax.axvline(key[3], color="black", linestyle=":", linewidth=1.2)
            ax.set_xlabel("Training feature-erasure probability")
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Deployment test accuracy (%)")
        axes[1].set_ylabel("Deployment cross entropy")
        axes[0].legend(fontsize=9)
        fig.suptitle(f"Fixed deployment erasure probability {key[3]:g}: {key[0].upper()}")
        fig.text(.5, -.025, "Dotted: matched distribution; p_train=0: perfect training; PAC at zero also represents Proposed Perfect\nAll saved seeds; pointwise 95% seed t intervals; no test-peak selection", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .03, 1, .94))
        save_figure(fig, output, f"{root_label}_mismatch_{index_number}", "Fixed deployment / varying training channel", index)


def decomposition_figures(rows, root_label, output, index):
    eligible = [row for row in rows if (row.get("_report") or {}).get("signed_gap_decomposition_mc")
                and row["config"].get("training_loss") != "raw_ce"
                and row["config"]["method"] not in {"erm", "l2"}]
    if not eligible:
        return
    # Compare methods on a representative shared n,m and include perfect models.
    settings = defaultdict(list)
    for row in eligible:
        cfg = row["config"]
        key = (cfg["dataset"], cfg["width"], cfg["train_size"], cfg["m"],
               json.dumps(cfg["deployment_channel"], sort_keys=True))
        settings[key].append(row)
    n_counts = defaultdict(int)
    for key in settings:
        n_counts[key[2]] += 1
    preferred_n = max(n_counts, key=lambda n: (n_counts[n], -abs(n-512)))
    primary = sorted([key for key in settings if key[2] == preferred_n], key=lambda key: key[3])
    representatives = [min(primary, key=lambda key: abs(key[3]-100)), primary[0], primary[-1]]
    chosen = list(dict.fromkeys(representatives))
    for number, key in enumerate(chosen):
        runs = settings[key]
        methods = defaultdict(list)
        for row in runs:
            cfg = row["config"]
            train = cfg["train_channel"]
            label = row["method_label"]
            if row.get("role") == "mismatch":
                label += f" p={train.get('outage_prob', 0):g}"
            methods[label].append(row)
        labels = sorted(methods, key=method_order)
        if len(labels) > 12:
            labels = [label for label in labels if label.startswith(("T1", "PAC"))]
        if len(labels) > 12:
            labels = labels[:12]
        fig, axes = plt.subplots(1, 2, figsize=(max(11.5, len(labels)*.85), 5.7))
        term_names = (("channel_shift", "Channel shift", "#c49a55"),
                      ("channel_complexity", "Channel complexity", "#5f94bf"),
                      ("sample_complexity", "Data complexity", "#88ad85"))
        bottom = np.zeros(len(labels))
        unknown = []
        for component, legend, shade in term_names:
            heights = []
            for position, label in enumerate(labels):
                values = []
                for row in methods[label]:
                    report = row["_report"]
                    value = (report.get("channel_shift_diagnostic", {}).get("channel_shift_complexity")
                             if component == "channel_shift" else report["final_posterior_complexities"].get(component))
                    if finite(value):
                        values.append(value)
                if len(values) != len(methods[label]):
                    unknown.append(position)
                heights.append(statistics.fmean(values) if values else 0.)
            axes[0].bar(range(len(labels)), heights, bottom=bottom, color=shade, label=legend, width=.7)
            bottom += np.asarray(heights)
        joint_overlay_available = False
        for position, label in enumerate(labels):
            gap = stats([observed(row, "observed_gap") for row in methods[label]])
            errorbar(axes[0], position, gap, fmt="D", color="black", markersize=4,
                     label="Observed gap" if position == 0 else None)
            joint_bounds = []
            for row in methods[label]:
                report = row["_report"]
                shift = report.get("channel_shift_diagnostic", {}).get("channel_shift_complexity")
                joint = report.get("final_posterior_complexities", {}).get("joint_complexity")
                if finite(shift) and finite(joint):
                    joint_bounds.append(shift + joint)
            if len(joint_bounds) == len(methods[label]):
                joint_overlay_available = True
                axes[0].plot(position, statistics.fmean(joint_bounds), marker="s", linestyle="none",
                             markerfacecolor="none", markeredgecolor="#633b7a", markeredgewidth=1.4,
                             markersize=6, label="Theorem 2 gap bound" if position == 0 else None)
        axes[0].axhline(1, color="gray", linestyle=":", linewidth=1., label="Maximum normalized loss")
        axes[0].set(ylabel="Normalized bounded-loss units",
                    title="Unweighted Theorem 1 terms\nat each frozen posterior")
        for position in sorted(set(unknown)):
            axes[0].annotate("†", (position, bottom[position]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=14)
        positive, negative = np.zeros(len(labels)), np.zeros(len(labels))
        signed_names = (("channel_generalization", "Observed channel gap", "#5f94bf"),
                        ("data_generalization", "Observed data gap", "#88ad85"),
                        ("channel_shift", "Observed shift", "#c49a55"))
        for component, legend, shade in signed_names:
            heights = np.asarray([stats([observed(row, component) for row in methods[label]])["mean"] or 0. for label in labels])
            starts = np.where(heights >= 0, positive, negative)
            axes[1].bar(range(len(labels)), heights, bottom=starts, color=shade, label=legend, width=.7)
            positive += np.maximum(heights, 0)
            negative += np.minimum(heights, 0)
        for position, label in enumerate(labels):
            errorbar(axes[1], position, stats([observed(row, "observed_gap") for row in methods[label]]),
                     fmt="D", color="black", markersize=4, label="Total observed gap" if position == 0 else None)
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set(ylabel="Normalized bounded-loss units", title="Signed observed decomposition")
        # Zero-height stack segments add sticky edges at the total height in
        # Matplotlib, so give observed markers explicit headroom.
        low, high = min(0., float(negative.min())), max(0., float(positive.max()))
        for label in labels:
            total = stats([observed(row, "observed_gap") for row in methods[label]])
            if total["n"]:
                half = total["half"] or 0.
                low, high = min(low, total["mean"]-half), max(high, total["mean"]+half)
        span = max(high-low, .001)
        axes[1].set_ylim(low - (.08*span if low < 0 else 0), high + .18*span)
        for ax in axes:
            ax.set_xticks(range(len(labels)), [label.replace("-", "\n", 1) for label in labels], rotation=35, ha="right")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            ax.margins(y=.18)
        fig.suptitle(f"Frozen-posterior decomposition: {key[0].upper()}, n={key[2]}, m={key[3]}")
        handles = [Patch(facecolor="#c49a55", label="Shift contribution"),
                   Patch(facecolor="#5f94bf", label="Channel contribution"),
                   Patch(facecolor="#88ad85", label="Data contribution"),
                   Line2D([], [], color="black", marker="D", linestyle="none", markersize=5,
                          label="Observed total gap")]
        if joint_overlay_available:
            handles.append(Line2D([], [], color="#633b7a", marker="s", markerfacecolor="none",
                                  linestyle="none", markersize=6, label="Theorem 2 gap bound"))
        fig.legend(handles=handles,
                   loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, fontsize=9)
        note = ("Bars: Theorem 1 terms; hollow squares: shift + Theorem 2 joint term at the same posterior; no training weights applied"
                if joint_overlay_available else "Bars: unweighted Theorem 1 terms at each frozen posterior; joint bounds unavailable in saved reports")
        note += "\nMeans over saved seeds; observed risks are Monte Carlo estimates, not automatic certificates"
        shift_methods = {(row.get("_report") or {}).get("channel_shift_diagnostic", {}).get("shift_method") for row in runs}
        if "exact_total_variation_feature_erasure" in shift_methods:
            note += "\nErasure-channel shift uses exact channel total variation"
        if unknown:
            note += "\n† A shift term is unavailable: displayed theoretical stack is incomplete"
        fig.text(.5, -.03, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .83))
        save_figure(fig, output, f"{root_label}_decomposition_{number}",
                    "Unweighted Theorem 1 terms and Theorem 2 gap bounds at each posterior, with signed empirical decomposition", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = dict(figures=[], sources=[], warnings=[], policy="Saved JSON only; no training or checkpoint inference; sources never pooled")
    used_names = set()
    for root in args.roots:
        root = root.resolve()
        name = slug(root.name)
        if name in used_names:
            name += "_" + hashlib.sha256(str(root).encode()).hexdigest()[:6]
        used_names.add(name)
        rows, summary = read_campaign(root)
        index["sources"].append(dict(root=str(root), evaluated_runs=len(rows),
                                     complete_trials=summary.get("completed_trials")))
        if not rows:
            index["warnings"].append(f"No completed test evaluations in {root}; stage skipped")
            continue
        if any(row.get("role") in {"mechanism", "mismatch", "reference"} for row in rows) and not any(
                (row.get("_report") or {}).get("signed_gap_decomposition_mc") for row in rows):
            index["warnings"].append(f"Full-risk JSON reports unavailable in {root}; decomposition skipped")
        performance_figures(rows, name, args.output, index)
        mechanism_figures(rows, name, args.output, index)
        mismatch_figures(rows, name, args.output, index)
        decomposition_figures(rows, name, args.output, index)
    (args.output / "figure_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"figures": len(index["figures"]), "warnings": index["warnings"],
                      "index": str((args.output / "figure_index.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
