"""Render the two fixed-study figures from the completed reports."""

from collections import defaultdict
import math
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


def save_figure(figure, output):
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.035)
    figure.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=.035)
    plt.close(figure)


def plot_mnist(observations, output):
    runs = defaultdict(dict)
    for row in observations:
        size, seed = row["m"], row["seed"]
        if seed in runs[size]:
            raise ValueError(f"Repeated MNIST observation for m={size}, seed={seed}")
        runs[size][seed] = row
    sizes = sorted(runs)
    if len(sizes) != 18 or any(set(rows) != {74101, 74102, 74103} for rows in runs.values()):
        raise ValueError("Expected 18 MNIST bank sizes with three paired seeds each")
    series = [
        ("generalization_error", 1, r"$\Delta(U,S)$", "#1F5B91", "o", "-"),
        ("test_accuracy", 100, "Deployment accuracy (%)", "#A64915", "D", "--"),
    ]
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42}):
        figure, gap_axis = plt.subplots(figsize=(3.45, 1.85), layout="constrained")
        accuracy_axis = gap_axis.twinx()
        for axis, (key, scale, label, color, marker, line) in zip((gap_axis, accuracy_axis), series):
            axis.plot(sizes, [scale * mean(row[key] for row in runs[size].values()) for size in sizes],
                      color=color, marker=marker, linestyle=line, linewidth=1.25,
                      markersize=3.4, markerfacecolor="white", markeredgewidth=.85)
            axis.set_ylabel(label, color=color, labelpad=4)
            axis.tick_params(axis="y", colors=color, length=2.8, width=.65, pad=2.8)
            axis.spines["top"].set_visible(False)
        gap_axis.spines["right"].set_visible(False)
        accuracy_axis.spines[["bottom", "left"]].set_visible(False)
        gap_axis.set_axisbelow(True)
        gap_axis.grid(axis="y", color="#DCE0E4", linewidth=.55)
        gap_axis.set_xscale("log")
        gap_axis.xaxis.set_major_locator(FixedLocator([2, 5, 20, 100, 800, 3200, 25600]))
        gap_axis.xaxis.set_major_formatter(ScalarFormatter())
        gap_axis.xaxis.set_minor_locator(NullLocator())
        gap_axis.set_xlabel(r"Training channel-set size $m$", labelpad=4)
        save_figure(figure, output)


def plot_bound(fading, erasure, output):
    components = {
        "empirical": ("Empirical risk", "#777777"),
        "sample": ("Data complexity", "#009e73"),
        "channel": ("Channel complexity", "#0072b2"),
        "joint": ("Joint complexity (T2)", "#935d99"),
        "shift": ("Channel shift", "#e69f00"),
    }
    cases = ("perfect", "t1_mismatch", "t2_mismatch", "t1_matched", "t2_matched")
    positions = (0, .88, 1.46, 2.34, 2.92)
    figure, axes = plt.subplots(2, 1, figsize=(4.65, 3.15), sharex=True)
    figure.subplots_adjust(left=.14, right=.985, top=.82, bottom=.17, hspace=.37)
    for axis, panel, title, minimum_upper in zip(
            axes, (fading, erasure), ("(a) Fading", "(b) Erasure"), (1.05, .60)):
        reports = dict(panel)
        heldout, totals = [], []
        for case, x in zip(cases, positions):
            report = reports[case]
            parts = report["components"]
            running = 0.
            for field, (_, color) in components.items():
                value = parts.get(field, 0.)
                if value:
                    axis.bar(x, value, .49, bottom=running, color=color)
                running += value
            if not math.isclose(running + parts["mc"], report["uncapped_risk_upper"]):
                raise ValueError(f"Inconsistent risk decomposition for {report['family']}/{case}")
            axis.annotate(f"{running:.3f}", (x, running), xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=7,
                          bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"))
            totals.append(running)
            heldout.append(report["test"]["bounded_loss"])
        axis.scatter(positions, heldout, marker="D", s=16, color="black",
                     edgecolors="white", linewidths=.45, zorder=4)
        axis.set_ylim(0, max(minimum_upper, 1.10 * max(*totals, *heldout)))
        axis.set_title(title, fontsize=9, pad=2)
        axis.tick_params(axis="both", labelsize=7.5, length=2.5)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#ededed", linewidth=.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[0].axhline(1., color="#555555", linestyle=":", linewidth=1)
    axes[1].set_xticks(positions, ["C1", "T1", "T2", "T1", "T2"])
    axes[1].set_xlim(-.36, 3.28)
    for x, label in [(0, "Perfect"), (1.17, "Mismatch"), (2.63, "Matched")]:
        axes[1].text(x, -.19, label, transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=7.5)
    figure.supylabel("Population-risk bound components", x=.045, fontsize=8)
    figure.legend(handles=[Patch(color=color, label=label) for label, color in components.values()] + [
        Line2D([], [], marker="D", color="black", linestyle="none", markersize=4, label="Held-out Gibbs loss")],
        loc="upper center", bbox_to_anchor=(.54, .99), ncol=3, frameon=False, fontsize=7,
        labelspacing=.24, columnspacing=.8, handlelength=1.2)
    save_figure(figure, output)
