"""Plot saved d=8 results with held-out Gibbs loss markers and without the Monte Carlo correction.

Components and held-out losses come from the selected case reports under data/bounds/.
Bar labels show the sum of displayed components.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COMPONENTS = [
    ("Empirical risk", "#777777"),
    ("Data complexity", "#009e73"),
    ("Channel complexity", "#0072b2"),
    ("Joint complexity (T2)", "#935d99"),
    ("Channel shift", "#e69f00"),
]
STACK_ORDER = (0, 1, 2, 3, 4)

BOUNDS = Path(__file__).resolve().parent / "data" / "bounds"
STEM = Path(__file__).with_name("bound_decomposition_d8_t1_t2_both_channels")
CASES = ("perfect", "t1_matched", "t2_matched", "t1_mismatch", "t2_mismatch")
FIELDS = ("empirical", "sample", "channel", "joint", "shift")
YMAX = dict(fading=1.05, erasure=.60)


def read_panel(family):
    directory = BOUNDS / family
    rows, heldout = [], []
    parameter, matched, mismatch = (("rho_c", 0.35, 0.15) if family == "fading"
                                    else ("outage_prob", 0.50, 0.45))
    for case, certificate in zip(CASES, ("C1", "T1", "T2", "T1", "T2")):
        record = json.loads((directory / case / "bound.json").read_text())
        cfg = record["config"]
        if (record["family"] != family or record["certificate"] != certificate
                or cfg["width"] != 8 or cfg["m"] != 2000 or cfg["deployment_channel"][parameter] != matched):
            raise ValueError(f"Unexpected deployment or certificate in {family}/{case}")
        train = cfg["train_channel"]
        if case == "perfect":
            valid_train = train["kind"] == "perfect"
        else:
            valid_train = train[parameter] == (mismatch if case.endswith("_mismatch") else matched)
        if not valid_train:
            raise ValueError(f"Unexpected training channel in {family}/{case}")
        components = record["components"]
        values = [components.get(field, 0.) for field in FIELDS]
        if not math.isclose(sum(values) + components["mc"], record["uncapped_risk_upper"]):
            raise ValueError(f"Inconsistent risk decomposition in {family}/{case}")
        rows.append(values)
        heldout.append(record["test"]["bounded_loss"])
    return rows, heldout


def main():
    fig, axes = plt.subplots(2, 1, figsize=(4.65, 3.05), sharex=True)
    fig.subplots_adjust(left=.14, right=.985, top=.83, bottom=.17, hspace=.29)
    positions = [0, .88, 1.46, 2.34, 2.92]
    display_order = (0, 3, 4, 1, 2)
    panels = [
        (*read_panel("fading"), "(a)", YMAX["fading"]),
        (*read_panel("erasure"), "(b)", YMAX["erasure"]),
    ]
    for ax, (rows, test_risks, title, ymax) in zip(axes, panels):
        for x, index in zip(positions, display_order):
            values = rows[index]
            bottom = 0
            for index in STACK_ORDER:
                value = values[index]
                _, color = COMPONENTS[index]
                if value:
                    ax.bar(x, value, .49, bottom=bottom, color=color)
                bottom += value
            ax.text(x, bottom + .012, f"{bottom:.3f}", ha="center", fontsize=7)
        ax.scatter(positions, [test_risks[i] for i in display_order], marker="D", s=16, color="black",
                   edgecolors="white", linewidths=.45, zorder=4)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_ylim(0, ymax)
        ax.set_yticks([t for t in ((0, .25, .50, .75, 1) if ymax > 1 else (0, .2, .4, .6)) if t <= ymax])
        ax.tick_params(axis="both", labelsize=7.5, length=2.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#ededed")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[0].axhline(1, color="#555555", linestyle=":", linewidth=1)
    axes[1].set_xticks(positions, ["C1", "T1", "T2", "T1", "T2"])
    axes[1].set_xlim(-.36, 3.28)
    for x, label in [(0, "Perfect"), (1.17, "Mismatch"), (2.63, "Matched")]:
        axes[1].text(x, -.19, label, transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=7.5)
    fig.supylabel("Population-risk bound components", x=.045, fontsize=8)
    fig.legend(
        handles=[Patch(color=color, label=name) for name, color in (COMPONENTS[i] for i in STACK_ORDER)] + [
            Line2D([], [], marker="D", color="black", linestyle="none",
                   markersize=4, label="Population loss"),
        ],
        loc="upper center", bbox_to_anchor=(.54, .985), ncol=3,
        frameon=False, fontsize=7, labelspacing=.24, columnspacing=.8, handlelength=1.2,
    )
    stem = STEM
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
