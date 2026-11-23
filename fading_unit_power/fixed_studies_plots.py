"""Render the MNIST bank sweep figure from the completed reports."""

from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
