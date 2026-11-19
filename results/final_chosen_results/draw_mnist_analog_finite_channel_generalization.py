"""Draw the single-column MNIST figure from the spatially correlated diagonal fading sweep."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator, ScalarFormatter


def main():
    output_dir = Path(__file__).resolve().parent
    source = output_dir / "data" / "mnist_bank" / "observations.csv"
    output = output_dir / "mnist_analog_finite_channel_generalization"
    bank_sizes = [2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    ticks = [2, 5, 20, 100, 800, 3200, 25600]

    observations = defaultdict(dict)
    with source.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["method"] != "T1":
                continue
            m, seed = int(row["m"]), int(row["seed"])
            if seed in observations[m]:
                raise ValueError(f"Duplicate T1 sweep result at m={m}, seed {seed}")
            observations[m][seed] = (
                float(row["generalization_error"]),
                100 * float(row["test_accuracy"]),
            )

    seeds = {74101, 74102, 74103}
    if set(observations) != set(bank_sizes) or any(set(runs) != seeds for runs in observations.values()):
        raise ValueError("Expected all three T1 seeds at each bank size")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "mathtext.fontset": "dejavuserif",
    })
    fig, gap_axis = plt.subplots(figsize=(3.45, 1.85), layout="constrained")
    accuracy_axis = gap_axis.twinx()
    series = [
        (gap_axis, r"$\Delta(U,S)$", "#1F5B91", "o", "-"),
        (accuracy_axis, "Deployment accuracy (%)", "#A64915", "D", "--"),
    ]
    for index, (axis, label, color, marker, line) in enumerate(series):
        axis.plot(
            bank_sizes, [mean(v[index] for v in observations[m].values()) for m in bank_sizes],
            color=color, marker=marker, linestyle=line, linewidth=1.25,
            markersize=3.4, markerfacecolor="white", markeredgewidth=0.85,
        )
        axis.set_ylabel(label, color=color, labelpad=4)
        axis.tick_params(axis="y", colors=color, length=2.8, width=0.65, pad=2.8)
        axis.spines["top"].set_visible(False)
    gap_axis.spines["right"].set_visible(False)
    accuracy_axis.spines[["bottom", "left"]].set_visible(False)
    gap_axis.set_ylim(0, 0.52)
    gap_axis.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    gap_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    accuracy_axis.set_ylim(25, 82)
    accuracy_axis.set_yticks([30, 40, 50, 60, 70, 80])
    gap_axis.set_axisbelow(True)
    gap_axis.grid(axis="y", color="#DCE0E4", linewidth=0.55)
    gap_axis.set_xscale("log")
    gap_axis.set_xlim(1.79, bank_sizes[-1] * 1.15)
    gap_axis.xaxis.set_major_locator(FixedLocator(ticks))
    gap_axis.xaxis.set_major_formatter(ScalarFormatter())
    gap_axis.xaxis.set_minor_locator(NullLocator())
    gap_axis.set_xlabel(r"Training channel-set size $m$", labelpad=4)

    export_options = {"bbox_inches": "tight", "pad_inches": 0.035}
    fig.savefig(
        output.with_suffix(".pdf"), **export_options,
        metadata={"Title": "Finite-channel generalization on MNIST for T1 under spatially correlated diagonal fading"},
    )
    fig.savefig(output.with_suffix(".png"), dpi=450, **export_options)
    plt.close(fig)


if __name__ == "__main__":
    main()
