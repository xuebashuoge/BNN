"""Bound vs accuracy trade-off of the tighter-bound candidates, one panel per case.

Each line is the Pareto front (smaller bound, higher validation accuracy) of the posteriors
trained for that case; markers show the certificates picked by each selection rule and the
archived ``only_fresh`` certificate. Bounds on the fronts use the selection Monte Carlo seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = ("perfect", "t1_matched", "t2_matched", "t1_mismatch", "t2_mismatch")
TITLES = dict(perfect="C1 perfect", t1_matched="T1 matched", t2_matched="T2 matched",
              t1_mismatch="T1 mismatch", t2_mismatch="T2 mismatch")
RULES = dict(tightest=("o", "#0072b2"), balanced=("s", "#009e73"), accurate=("^", "#d55e00"))


def main(bounds: Path):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4), sharey="row")
    for row, family in enumerate(("fading", "erasure")):
        for col, case in enumerate(CASES):
            ax = axes[row, col]
            records = {rule: json.loads((bounds / rule / family / case / "bound.json").read_text()) for rule in RULES}
            drawn = set()
            for rule, record in records.items():
                front = sorted(record["selection"]["frontier"], key=lambda p: p["validation_accuracy"])
                key = tuple((p["label"], round(p["upper"], 6)) for p in front)
                if key not in drawn:
                    drawn.add(key)
                    ax.plot([p["validation_accuracy"] for p in front], [p["upper"] for p in front],
                            color="#bbbbbb", lw=1, zorder=1)
                marker, color = RULES[rule]
                ax.scatter(record["selection"]["validation_accuracy"], record["uncapped_risk_upper"], marker=marker,
                           color=color, s=30, zorder=3, label=rule if row == col == 0 else None)
            previous = records["tightest"]["previous"]
            ax.scatter(previous["only_fresh_accuracy"], previous["only_fresh_upper"], marker="x", color="black",
                       s=30, zorder=3, label="archived only_fresh" if row == col == 0 else None)
            ax.axhline(records["tightest"]["test"]["bounded_loss"], color="#999999", ls=":", lw=.8)
            ax.set_title(f"{family}: {TITLES[case]}", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.grid(color="#eeeeee")
            if col == 0:
                ax.set_ylabel("certified bound", fontsize=8)
            if row == 1:
                ax.set_xlabel("validation accuracy (deployment)", fontsize=8)
    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .94))
    stem = bounds.parent / "bound_accuracy_tradeoff"
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=200, facecolor="white")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
