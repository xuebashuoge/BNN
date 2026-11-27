"""Draw CIFAR-10 deployment accuracy from the m=2000 channel-mismatch sweep.

T1 uses the split-U data-dependent prior Q_{W|S,U_1} (split_u/cifar10_mismatch). With
--fresh, T1 uses the earlier fresh prior (data_dependent/only_fresh) and the figure is
written to data/cifar10_mismatch/cifar10_channel_mismatch_fresh.*. With --fixed-prior,
T1 uses the fixed prior Q_W and the figure is written to
data/cifar10_mismatch/cifar10_channel_mismatch_fixed_prior.*.
"""

import csv
import sys
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).resolve().parent
BANK_SIZE = 2000
SOURCE = OUTPUT_DIR / "split_u" / "cifar10_mismatch" / "observations.csv"
STEM = "cifar10_channel_mismatch_all_methods"
PROBABILITIES = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SEEDS = {75101, 75102, 75103}
METHOD_STYLES = {
    "ERM": ("#636871", "o", ":"),
    "L2": ("#377967", "^", "-."),
    "T1": ("#1F5B91", "o", "-"),
    "T2": ("#BC540F", "D", "--"),
}


def load_means():
    observations = defaultdict(dict)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment"] != "p_train_sweep" or not row["group"].startswith("cifar10_outage_"):
                continue
            if int(row["m"]) != BANK_SIZE:
                raise ValueError(f"Expected m={BANK_SIZE} for {row['task_id']}")
            probability = float(row["sweep_value"])
            if probability not in PROBABILITIES:
                continue
            key = (row["method"], probability)
            seed = int(row["seed"])
            if seed in observations[key]:
                raise ValueError(f"Duplicate sweep result for {key}, seed {seed}")
            observations[key][seed] = 100 * float(row["test_accuracy"])

    expected = {(method, p) for method in METHOD_STYLES for p in PROBABILITIES}
    if set(observations) != expected or any(set(runs) != SEEDS for runs in observations.values()):
        raise ValueError("Expected all four methods and all three paired seeds at each training probability")

    return {key: mean(runs.values()) for key, runs in observations.items()}


def draw(accuracies):
    figure, axis = plt.subplots(figsize=(3.45, 2.10))
    figure.subplots_adjust(left=0.177, right=0.968, bottom=0.19, top=0.97)
    axis.grid(axis="y", linewidth=0.55, color="#DCE0E4")
    axis.set_axisbelow(True)
    axis.axvline(0.5, ymin=0.19, color="#474A50", linestyle=":", linewidth=1.05)
    legend_handles = []
    for method, (color, marker, line) in METHOD_STYLES.items():
        averages = [accuracies[method, probability] for probability in PROBABILITIES]
        series, = axis.plot(
            PROBABILITIES, averages, label=method, color=color,
            marker=marker, linestyle=line, markerfacecolor="white",
            markeredgewidth=0.85, linewidth=1.25, markersize=3.6,
            markevery=slice(1, None),
        )
        axis.plot(
            0, averages[0], marker="*", markersize=5.6,
            markerfacecolor="white", markeredgewidth=0.85, color=color, zorder=4,
        )
        legend_handles.append(series)

    axis.set_ylabel("Deployment accuracy (%)")
    axis.set_xlabel(r"Training outage probability $p_{\mathrm{tr}}$")
    axis.set_xlim(-0.024, 0.923)
    # The archived figure tops out at 55%; extend in 5% steps only if a curve exceeds it.
    top = max(55, 5 * math.ceil((max(accuracies.values()) + 0.5) / 5))
    axis.set_ylim(29, top)
    axis.set_yticks(range(30, top + 1, 5))
    axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axis.spines[["top", "right"]].set_visible(False)
    legend_handles.append(Line2D([], [], color="#474A50", linestyle=":", label="Matched law"))
    legend_handles.append(
        Line2D([], [], color="#474A50", linestyle="none", marker="*", markersize=5.6,
               markerfacecolor="white", label="Perfect training")
    )
    axis.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.485, 30),
        bbox_transform=axis.get_yaxis_transform(), borderaxespad=0, ncol=3,
        framealpha=1, facecolor="white", edgecolor="none", columnspacing=0.75,
        handlelength=1.35, handletextpad=0.4,
        fontsize=6.8,
    )

    output = OUTPUT_DIR / STEM
    export_options = {"bbox_inches": "tight", "pad_inches": 0.025}
    figure.savefig(
        output.with_suffix(".pdf"),
        metadata={"Title": f"CIFAR-10 channel-mismatch sweep (m={BANK_SIZE})"},
        **export_options,
    )
    figure.savefig(output.with_suffix(".png"), dpi=450, **export_options)
    plt.close(figure)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.linewidth": 0.65,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    draw(load_means())


def use_fixed_prior():
    global SOURCE, OUTPUT_DIR, STEM
    OUTPUT_DIR = OUTPUT_DIR / "data" / "cifar10_mismatch"
    SOURCE = OUTPUT_DIR / "observations.csv"
    STEM = "cifar10_channel_mismatch_fixed_prior"


def use_fresh_prior():
    global SOURCE, OUTPUT_DIR, STEM
    SOURCE = OUTPUT_DIR / "data_dependent" / "only_fresh" / "cifar10_mismatch" / "observations.csv"
    OUTPUT_DIR = OUTPUT_DIR / "data" / "cifar10_mismatch"
    STEM = "cifar10_channel_mismatch_fresh"


if __name__ == "__main__":
    if sys.argv[1:] == ["--fixed-prior"]:
        use_fixed_prior()
    elif sys.argv[1:] == ["--fresh"]:
        use_fresh_prior()
    elif sys.argv[1:]:
        sys.exit("Usage: draw_cifar10_channel_mismatch.py [--fresh | --fixed-prior]")
    main()
