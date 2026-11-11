"""Prepare compact journal figures from saved experiment JSON; no learning code.

All displayed models and seeds are fixed by the recorded experiment design.
Means and sample standard deviations describe the saved training/bank seeds.
The numerical certificate has a separate, explicitly allocated confidence budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tit-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


COLORS = dict(t1="#0072B2", t2="#D55E00", perfect="#009E73", erm="#666666",
              shift="#E69F00", channel="#0072B2", data="#009E73", joint="#CC79A7",
              empirical="#696969", mc="#C5CBD1")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5,
                     "mathtext.fontset": "dejavuserif", "axes.labelsize": 9,
                     "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def summary(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Each displayed quantity needs finite observations")
    return {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0., "n": len(values),
            "values": values}


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.cache = {}
        self.rows = self.read(self.root / "test_results.json")
        self.rows = [row for row in self.rows if row.get("test_status") == "complete"]
        if len({row["task_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate test rows would duplicate seed evidence")

    def read(self, path):
        path = Path(path)
        content = path.read_bytes()
        self.files[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def report(self, row):
        key = row["task_id"]
        if key not in self.cache:
            path = self.root / key / "test_full" / "evaluation.json"
            report = self.read(path)
            if report["evaluation_config"]["training_loss"] != "bounded_ce_smooth":
                raise ValueError("The paper gap panels require the common smooth bounded loss")
            for metric in ("bounded_loss", "cross_entropy"):
                signed = report["signed_gap_decomposition_mc"][metric]
                if abs(signed["sum"] - signed["observed_gap"]) > 1e-10:
                    raise ValueError("Saved risk decomposition does not telescope")
            self.cache[key] = report
        return self.cache[key]

    def select(self, group, method):
        rows = sorted([row for row in self.rows if row["group"] == group and
                       row["method_label"] == method], key=lambda row: row["seed"])
        if len(rows) != 3:
            raise ValueError(f"Expected all three saved seeds for {group}/{method}, got {len(rows)}")
        return rows


def finish(fig, output, name):
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=260 if suffix == "png" else None,
                    pad_inches=.045)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def axes_style(ax):
    ax.grid(alpha=.18, linewidth=.5)
    ax.set_axisbelow(True)


def draw_curve(ax, x, arrays, label, color, marker, scale=1., linestyle="-"):
    stats = [summary(values) for values in arrays]
    mean = np.asarray([s["mean"] for s in stats]) * scale
    sd = np.asarray([s["sd"] for s in stats]) * scale
    ax.plot(x, mean, linestyle=linestyle, marker=marker, markersize=4.2,
            linewidth=1.55, color=color, markerfacecolor="white" if marker == "s" else color,
            label=label, zorder=4)
    ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.10, linewidth=0)
    for xx, values in zip(x, arrays):
        ax.scatter(np.repeat(xx, len(values)), np.asarray(values)*scale, s=12,
                   color=color, alpha=.3, linewidths=0, zorder=3)
    return stats


def channel_bank(evidence, output):
    sizes = [5, 20, 100, 500, 2000]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    data = {"n": 512, "width": 8, "deployment_erasure_probability": .5,
            "m": sizes, "seed_summary": "mean and sample SD across all three saved seeds", "methods": {}}
    for method, label, color, marker, line in (("T1-Matched", "Proposed T1", COLORS["t1"], "o", "-"),
                                             ("T2-Matched", "Proposed T2", COLORS["t2"], "s", "--")):
        runs = [evidence.select(f"moons_outage_n512_m{m}", method) for m in sizes]
        accuracy = [[row["test_accuracy"] for row in rows] for rows in runs]
        gaps = [[evidence.report(row)["signed_gap_decomposition_mc"]["bounded_loss"]["channel_generalization"]
                 for row in rows] for rows in runs]
        accuracy_stats = draw_curve(axes[0], sizes, accuracy, label, color, marker, 100., line)
        gap_stats = draw_curve(axes[1], sizes, gaps, label, color, marker, 1., line)
        data["methods"][label] = {"accuracy": accuracy_stats, "channel_gap": gap_stats,
                                    "task_ids": [[row["task_id"] for row in rows] for rows in runs]}
    for method, label, color, line in (("ERM-Perfect", "ERM, perfect training", COLORS["erm"], ":"),
                                      ("PAC-Perfect", "Proposed, perfect training", COLORS["perfect"], "-.")):
        rows = evidence.select("moons_outage_perfect_n512", method)
        stats = summary([row["test_accuracy"] for row in rows])
        axes[0].axhline(stats["mean"]*100, color=color, linestyle=line, linewidth=1.3, label=label)
        axes[0].axhspan((stats["mean"]-stats["sd"])*100, (stats["mean"]+stats["sd"])*100,
                       color=color, alpha=.035)
        data["methods"][label] = {"accuracy": stats, "channel_gap": 0.,
                                    "task_ids": [row["task_id"] for row in rows]}
    axes[1].axhline(0, color="black", linewidth=.7, linestyle=":")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(x) for x in sizes])
        ax.set_xlabel(r"Channel-bank size $m$")
        axes_style(ax)
    axes[0].set_ylabel("Deployment test accuracy (%)")
    axes[1].set_ylabel("Observed channel gap")
    axes[0].set_title("(a) Accuracy")
    axes[1].set_title("(b) Finite-channel generalization")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2,
               columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=1.7)
    return {"files": finish(fig, output, "fig_channel_bank"), "data": data,
            "caption_note": "Bands are sample SD across three initialization/channel-bank seeds, conditional on the fixed dataset. Perfect models are trained once per seed and reused across m. The observed gap uses normalized bounded loss."}


def training_mismatch(evidence, output):
    probabilities = [0., .1, .3, .5, .7, .9]
    perfect = evidence.select("moons_outage_ptr00", "PAC")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    data = {"n": 800, "width": 16, "m": 100, "deployment_erasure_probability": .5,
            "training_erasure_probability": probabilities, "methods": {}}
    for method, label, color, marker, line in (("T1", "Proposed T1", COLORS["t1"], "o", "-"),
                                             ("T2", "Proposed T2", COLORS["t2"], "s", "--")):
        runs = [perfect] + [evidence.select(f"moons_outage_ptr{int(p*100):02d}", method)
                            for p in probabilities[1:]]
        accuracy = [[row["test_accuracy"] for row in rows] for rows in runs]
        ce = [[row["test_cross_entropy"] for row in rows] for rows in runs]
        accuracy_stats = draw_curve(axes[0], probabilities, accuracy, label, color, marker, 100., line)
        ce_stats = draw_curve(axes[1], probabilities, ce, label, color, marker, 1., line)
        data["methods"][label] = {"accuracy": accuracy_stats, "cross_entropy": ce_stats,
                                    "task_ids": [[row["task_id"] for row in rows] for rows in runs]}
    for ax, key, scale in ((axes[0], "test_accuracy", 100.), (axes[1], "test_cross_entropy", 1.)):
        stats = summary([row[key] for row in perfect])
        ax.scatter([0], [stats["mean"]*scale], color=COLORS["perfect"], marker="D", s=25,
                   label="Proposed, perfect training", zorder=7)
        ax.axvline(.5, color="#555555", linestyle=":", linewidth=1.0, label="Matched law")
        ax.set_xlabel(r"Training erasure probability $p_{\rm tr}$")
        ax.set_xticks(probabilities)
        axes_style(ax)
    axes[0].set_ylabel("Deployment test accuracy (%)")
    axes[1].set_ylabel("Deployment cross entropy")
    axes[0].set_title("(a) Accuracy")
    axes[1].set_title("(b) Loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2,
               columnspacing=1.7, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=1.7)
    return {"files": finish(fig, output, "fig_training_mismatch"), "data": data,
            "caption_note": "All six training laws are predeclared; no test-peak or seed selection. The p_tr=0 point is the shared perfect-channel special case of T1/T2. Bands show sample SD across three seeds. Feature erasure is not a bit-level digital BEC."}


def bound_decomposition(evidence, output):
    cases = [("Perfect\n" + r"$p_{\rm tr}=0$", "moons_outage_ptr00", "PAC"),
             ("Matched\n" + r"$p_{\rm tr}=0.5$", "moons_outage_ptr50", "T1"),
             ("Mismatched\n" + r"$p_{\rm tr}=0.1$", "moons_outage_ptr10", "T1")]
    data = {"n": 800, "width": 16, "m": 100, "deployment_erasure_probability": .5,
            "confidence_delta_per_run": .05, "cases": {}}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    xpos = np.arange(len(cases))
    reports = []
    for label, group, method in cases:
        rows = evidence.select(group, method)
        rr = [evidence.report(row) for row in rows]
        reports.append(rr)
        terms = {"shift": summary([r["channel_shift_diagnostic"]["channel_shift_complexity"] for r in rr]),
                 "channel": summary([r["final_posterior_complexities"]["channel_complexity"] for r in rr]),
                 "data": summary([r["final_posterior_complexities"]["sample_complexity"] for r in rr])}
        observed = {key: summary([r["signed_gap_decomposition_mc"]["bounded_loss"][key] for r in rr])
                    for key in ("channel_shift", "channel_generalization", "data_generalization", "observed_gap")}
        joint = summary([r["channel_shift_diagnostic"]["channel_shift_complexity"]+
                         r["final_posterior_complexities"]["joint_complexity"] for r in rr])
        data["cases"][label.replace("\n", " ")] = dict(terms=terms, observed=observed, joint=joint,
                                                            task_ids=[row["task_id"] for row in rows])
    bottoms = np.zeros(len(cases))
    for key in ("shift", "channel", "data"):
        heights = [item["terms"][key]["mean"] for item in data["cases"].values()]
        axes[0].bar(xpos, heights, bottom=bottoms, width=.62, color=COLORS[key], label=key)
        bottoms += np.asarray(heights)
    joint_values = [case["joint"]["mean"] for case in data["cases"].values()]
    axes[0].plot(xpos, joint_values, marker="s", linestyle="none", markersize=5.5,
                 markerfacecolor="none", markeredgecolor=COLORS["joint"], markeredgewidth=1.5)
    axes[0].axhline(1, color="#777777", linestyle=":", linewidth=.9)
    positive, negative = np.zeros(len(cases)), np.zeros(len(cases))
    for key, color_key in (("channel_shift", "shift"), ("channel_generalization", "channel"),
                           ("data_generalization", "data")):
        values = np.asarray([case["observed"][key]["mean"] for case in data["cases"].values()])
        starts = np.where(values >= 0, positive, negative)
        axes[1].bar(xpos, values, bottom=starts, width=.62, color=COLORS[color_key])
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    for ax in axes:
        values = [case["observed"]["observed_gap"]["mean"] for case in data["cases"].values()]
        sd = [case["observed"]["observed_gap"]["sd"] for case in data["cases"].values()]
        ax.errorbar(xpos, values, yerr=sd, fmt="D", color="black", markersize=3.5,
                    capsize=2.5, linewidth=1.0, zorder=6)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(xpos, [item[0] for item in cases])
        axes_style(ax)
    axes[0].set_ylabel("Bounded-loss gap units")
    axes[1].set_ylabel("Bounded-loss gap units")
    axes[0].set_title("(a) Theorem 1 terms and Theorem 2 bound")
    axes[1].set_title("(b) Signed observed decomposition")
    handles = [Patch(color=COLORS[key], label=label) for key, label in
               (("shift", "Channel shift"), ("channel", "Channel component"), ("data", "Data component"))]
    handles += [Line2D([], [], color=COLORS["joint"], marker="s", markerfacecolor="none", linestyle="none",
                       label="Theorem 2 bound"),
                Line2D([], [], color="black", marker="D", linestyle="none", label="Observed total gap")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=3,
               columnspacing=1.1, handlelength=1.7)
    fig.tight_layout(rect=(0, 0, 1, .90), w_pad=1.5)
    return {"files": finish(fig, output, "fig_bound_decomposition"), "data": data,
            "caption_note": "Identical width/data/bank/deployment setting; only the training law differs. Left bars are unweighted Theorem 1 terms, evaluated on each frozen posterior; square markers are the corresponding Theorem 2 gap bound. Right bars are signed finite-data/Monte Carlo estimates. These gap-bound diagnostics do not include the empirical-risk MC upper correction; values above 1 are vacuous. Perfect/mismatched shift uses the separate bounded-loss TV specialization."}


def risk_certificate(evidence, path, output):
    cert = evidence.read(path)
    empirical = cert["empirical_gibbs_mc"]["bounded_loss"]
    mc = cert["confidence_allocation"]["empirical_mc_upper_radius"]
    shift = cert["channel_shift"]["channel_shift_complexity"]
    channel, sample, joint = cert["channel_complexity"], cert["sample_complexity"], cert["joint_complexity"]
    first = empirical + mc + shift + channel + sample
    second = empirical + mc + shift + joint
    if abs(first-cert["theorem_1"]["population_risk_upper_uncapped"]) > 1e-10 or abs(second-cert["theorem_2"]["population_risk_upper_uncapped"]) > 1e-10:
        raise ValueError("Certificate stack does not reproduce the saved bound")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bottoms = np.zeros(2)
    components = [("empirical", "Empirical loss", [empirical, empirical]),
                  ("mc", "MC correction", [mc, mc]),
                  ("channel", "Channel term", [channel, 0]),
                  ("data", "Data term", [sample, 0]),
                  ("joint", "Joint term", [0, joint])]
    if shift:
        components.insert(2, ("shift", "Shift term", [shift, shift]))
    for key, label, values in components:
        ax.bar([0, 1], values, bottom=bottoms, width=.55, color=COLORS[key], label=label)
        for x, value in enumerate(values):
            if value > .075:
                ax.text(x, bottoms[x]+value/2, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if key in {"empirical", "channel", "data", "joint"} else "black")
        bottoms += np.asarray(values)
    ax.axhline(1, color="#777777", linestyle=":", linewidth=.9)
    for x, total in enumerate(bottoms):
        ax.text(x, total+.018, f"{total:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks([0, 1], ["T1 (separate)", "T2 (joint)"])
    ax.set_ylabel("Normalized population-risk upper bound")
    ax.set_ylim(0, 1.055)
    ax.set_xlim(-.65, 1.65)
    axes_style(ax)
    fig.legend(loc="upper center", bbox_to_anchor=(.5, 1.075), ncol=2,
               columnspacing=1.0, handlelength=1.4, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, .85))
    return {"files": finish(fig, output, "fig_risk_certificate"),
            "data": {"n": 512, "m": 2000, "width": 8, "empirical": empirical, "mc_correction": mc,
                     "shift": shift, "channel_complexity": channel, "sample_complexity": sample,
                     "joint_complexity": joint, "t1_population_bound": first, "t2_population_bound": second,
                     "confidence_allocation": cert["confidence_allocation"], "source_checkpoint": cert["checkpoint"]},
            "caption_note": "One declared selected posterior, not a seed average. Includes the fresh empirical MC upper correction. The 0.05 failure budget covers both theorem families over all 21 declared setups plus one fresh MC evaluation. These bounds concern normalized smooth loss, not accuracy or raw CE, and are conditional on the supplied theorem assumptions."}


def bound_structure(output):
    """Analytic comparison of theorem expressions, unrelated to fitted models."""
    divergence, delta, sigma_sq = 2., .05, .25
    eps_uniform = -math.expm1(.5 * math.log1p(-delta))
    eps_channel_adaptive = 2 * delta / (100 + math.sqrt(10000 - 396 * delta))
    eps_data_adaptive = 99 * eps_channel_adaptive
    for channel_budget, data_budget in ((eps_uniform, eps_uniform),
                                         (eps_channel_adaptive, eps_data_adaptive)):
        if abs((1-channel_budget)*(1-data_budget) - (1-delta)) > 1e-12:
            raise ValueError("Analytic confidence allocation is not budget matched")

    def separate(n, m, eps_channel, eps_data):
        return (np.sqrt(2*sigma_sq/(np.asarray(m)-1) *
                        (divergence+np.log(np.sqrt(m)/eps_channel))) +
                np.sqrt(2*sigma_sq/(np.asarray(n)-1) *
                        (divergence+np.log(np.sqrt(n)/eps_data))))

    def joint(n, m):
        return np.sqrt(2*sigma_sq/(np.asarray(n)-1) + 2*sigma_sq/(np.asarray(m)-1)) * np.sqrt(
            divergence + np.log(np.sqrt(np.asarray(n)*m)/delta))

    def crossing(eps_channel, eps_data):
        grid = np.geomspace(2., 1e12, 10000)
        differences = joint(10, grid) - separate(10, grid, eps_channel, eps_data)
        found = np.where((differences[:-1] < 0) & (differences[1:] >= 0))[0]
        if not len(found):
            return None
        low, high = grid[found[0]], grid[found[0]+1]
        for _ in range(70):
            middle = math.sqrt(low*high)
            if joint(10, middle) < separate(10, middle, eps_channel, eps_data):
                low = middle
            else:
                high = middle
        return math.sqrt(low*high)

    uniform_crossing = crossing(eps_uniform, eps_uniform)
    adaptive_crossing = crossing(eps_channel_adaptive, eps_data_adaptive)
    m = np.geomspace(10., 1e9, 700)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    curves = [("Separate, uniform budget", COLORS["t1"], "-", eps_uniform, eps_uniform),
              ("Separate, adaptive budget", COLORS["t2"], "--", eps_channel_adaptive, eps_data_adaptive)]
    for ax, balanced in ((axes[0], False), (axes[1], True)):
        n = m if balanced else 10
        for label, color, line, eps_channel, eps_data in curves:
            ax.plot(m, separate(n, m, eps_channel, eps_data), color=color, linestyle=line,
                    linewidth=1.6, label=label)
        ax.plot(m, joint(n, m), color=COLORS["joint"], linestyle="-.", linewidth=1.6, label="Joint")
        ax.set_xscale("log")
        if balanced:
            ax.set_yscale("log")
        ax.set_xlabel(r"Channel sample size $m$")
        ax.set_ylabel("Complexity penalty")
        axes_style(ax)
    axes[0].set_title(r"(a) Asymmetric: $n=10$")
    axes[1].set_title(r"(b) Balanced: $n=m$")
    uniform_floor = math.sqrt(2*sigma_sq/9*(divergence+math.log(math.sqrt(10)/eps_uniform)))
    adaptive_floor = math.sqrt(2*sigma_sq/9*(divergence+math.log(math.sqrt(10)/eps_data_adaptive)))
    axes[0].axhline(uniform_floor, color=COLORS["t1"], linewidth=.7, alpha=.5, linestyle=":")
    axes[0].axhline(adaptive_floor, color=COLORS["t2"], linewidth=.7, alpha=.5, linestyle=":")
    if uniform_crossing is not None:
        height = float(joint(10, uniform_crossing))
        axes[0].plot([uniform_crossing], [height], marker="x", color="black", markersize=4)
        axes[0].annotate(r"$m\approx$" + f"{uniform_crossing:.0f}",
                         (uniform_crossing, height), xytext=(2e4, .83),
                         arrowprops={"arrowstyle": "->", "linewidth": .7}, fontsize=8)
    axes[0].set_ylim(.52, 1.42)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
               handlelength=2.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.7)
    return {"files": finish(fig, output, "fig_bound_structure"),
            "data": {"type": "analytic_theorem_comparison_not_learning_results", "kl": divergence,
                     "delta": delta, "sigma_sq": sigma_sq, "sigma_channel_sq": sigma_sq,
                     "uniform_epsilon_channel": eps_uniform, "uniform_epsilon_data": eps_uniform,
                     "adaptive_epsilon_channel": eps_channel_adaptive, "adaptive_epsilon_data": eps_data_adaptive,
                     "adaptive_data_to_channel_budget_ratio": 99,
                     "joint_epsilon": delta, "asymmetric_n": 10,
                     "asymmetric_uniform_crossing_m": uniform_crossing,
                     "asymmetric_adaptive_crossing_m": adaptive_crossing,
                     "asymmetric_uniform_floor": uniform_floor, "asymmetric_adaptive_floor": adaptive_floor,
                     "m_range": [10, 1e9]},
            "caption_note": "Analytic theorem expressions at fixed KL, not simulation training outcomes. Both separate allocations satisfy (1-eps_channel)(1-eps_data)=1-delta, while the joint bound uses epsilon=delta. Dotted horizontal lines are the separate bounds' asymptotic data floors in the asymmetric panel."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/journal_campaign/followup"))
    parser.add_argument("--certificate", type=Path, default=Path("results/journal_campaign/certificate_m2000/certificate.json"))
    parser.add_argument("--output", type=Path, default=Path("journal/paper/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(args.source)
    figures = {"channel_bank": channel_bank(evidence, args.output),
               "training_mismatch": training_mismatch(evidence, args.output),
               "bound_decomposition": bound_decomposition(evidence, args.output),
               "risk_certificate": risk_certificate(evidence, args.certificate, args.output),
               "bound_structure": bound_structure(args.output)}
    manifest = dict(figures=figures, source_files_sha256=evidence.files,
                    policy="Saved JSON only; no training, checkpoint inference, seed selection, or test-peak selection; source data unchanged")
    (args.output / "figure_data.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"figures": len(figures), "output": str(args.output.resolve()),
                      "source_json_files": len(evidence.files)}, indent=2))


if __name__ == "__main__":
    main()
