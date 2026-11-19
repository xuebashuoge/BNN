"""Plot and export the objective-specific two-moons correlated fading certificates."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .correlated_bound_study import CASES, FAMILY_SIZE, learner_config
from .diagonal_fading import DEPLOYMENT_RHO, diagonal_gain_kl
from .diagonal_fading_bounds import WIDTHS
from .run_experiment import file_fingerprint, save_json
from .theorem1 import joint_complexity, theorem1_terms


COMPONENTS = {
    "empirical": ("Empirical Gibbs loss", "#646B73"),
    "mc": ("Monte Carlo correction", "#C9CDD1"),
    "sample": ("Sample complexity", "#00856C"),
    "channel": ("Channel complexity", "#347ABB"),
    "joint": ("Joint complexity", "#9D65A4"),
    "shift": ("Channel shift", "#E5A02C"),
}
THEORIES = {"C1": "Corollary 1", "T1": "Theorem 1", "T2": "Theorem 2"}
STEM = "correlated_bound_decomposition"


def verify_case(report, width, case):
    """Recompute the component identities on the fitted posterior's declared KL."""
    cfg = report["config"]
    regime, method, certificate = CASES[case]
    if (cfg != learner_config(width, case, cfg["train_channel"].get("rho_c", DEPLOYMENT_RHO))
            or (report["dataset"], report["family"], report["regime"], cfg["method"],
                report["certificate"], report["theory"]) !=
            ("moons", "fading", regime, method, certificate, THEORIES[certificate])):
        raise ValueError("Unexpected training configuration or certificate")
    if regime == "mismatch" and not 0 < cfg["train_channel"]["rho_c"] < 1:
        raise ValueError("Expected positively correlated mismatch training")
    failure = cfg["delta"] / (2 * FAMILY_SIZE)
    if report["confidence"] != dict(family_failure=cfg["delta"], family_size=FAMILY_SIZE,
                                    failure_each_theorem_and_mc=failure):
        raise ValueError("Unexpected confidence allocation")
    test = report["test"]
    if (test["channel_source"] != "deployment" or not test["predictor"]["stochastic"]
            or not 0 <= test["accuracy"] <= 1 or not 0 <= test["bounded_loss"] <= 1):
        raise ValueError("Expected held-out deployment measurements for a Gibbs predictor")

    def check(label, actual, expected):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError(f"{label} differs: {actual} vs {expected}")

    empirical = report["empirical"]
    mc_radius = math.sqrt(math.log(1 / failure) / (2 * empirical["iid_observations"]))
    check("Empirical MC radius", report["empirical_mc_radius"], mc_radius)
    expected = {"empirical": empirical["mean"], "mc": min(mc_radius, 1 - empirical["mean"])}
    kl = torch.tensor(report["posterior_kl"], dtype=torch.float64)
    budget = dict(n=cfg["train_size"], m=cfg["m"], delta=failure)
    if certificate == "T2":
        expected["joint"] = joint_complexity(kl, **budget).item()
    else:
        terms = theorem1_terms(kl, **budget, perfect_training=regime == "perfect")
        expected.update(sample=terms["sample_complexity"].item(), channel=terms["channel_complexity"].item())

    shift = report["channel_shift"]
    deployment = cfg["deployment_channel"]
    if not shift["analytic"]:
        raise ValueError("Expected an analytic channel-shift bound")
    if regime == "perfect":
        if (shift["method"], shift["reference"]) != ("C1 second", "P_iid"):
            raise ValueError("Expected Corollary 1(ii) with the independent-gain reference")
        reference_kl = diagonal_gain_kl(deployment, dict(deployment, rho_c=0.), width)
        check("Deployment-to-reference KL", shift["reference_kl"], reference_kl)
        expected_shift = reference_kl + shift["reference_log_mgf_upper"]
    else:
        channel_kl = diagonal_gain_kl(deployment, cfg["train_channel"], width)
        check("Deployment-to-training KL", shift["channel_kl"], channel_kl)
        expected_shift = math.sqrt(channel_kl / 2)
    check("Channel shift", shift["upper"], expected_shift)
    expected["shift"] = expected_shift
    components = report["components"]
    if expected.keys() != components.keys():
        raise ValueError("Unexpected bound components for the reported certificate")
    for term, value in expected.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {term}: {value}")
        check(term, components[term], value)
    check("Uncapped bound", report["uncapped_risk_upper"], sum(components.values()))
    if report["nonvacuous"] != (report["uncapped_risk_upper"] < 1.):
        raise ValueError("Nonvacuity flag disagrees with the bound")


def read_case(root, width, case):
    path = root / f"d{width}" / case / "bound.json"
    report = json.loads(path.read_text())
    try:
        verify_case(report, width, case)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    cfg, test, shift = report["config"], report["test"], report["channel_shift"]
    train, deploy = cfg["train_channel"], cfg["deployment_channel"]
    metadata_path = path.parent / "checkpoint" / "metadata.json"
    training_revision = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        weights = file_fingerprint(metadata_path.with_name("weights.pt"))
        if (metadata["config"] != cfg or weights != report["checkpoint_weights"]
                or metadata["training_channel_bank_hash"] != report["saved_bank_hash"]
                or metadata["selected_epoch"] != report["selected_epoch"]):
            raise ValueError(f"Certificate and checkpoint disagree: {path}")
        training_revision = metadata["git_sha"]
    row = dict(width=width, case=case, dataset=report["dataset"], family=report["family"],
               regime=report["regime"], training_method=cfg["method"], certificate=report["certificate"],
               n=cfg["train_size"], m=cfg["m"], train_rho_c=train.get("rho_c"),
               test_rho_c=deploy["rho_c"], test_noise_std=deploy.get("noise_std", 1.),
               **{name: report["components"].get(name) for name in COMPONENTS},
               uncapped_risk_upper=report["uncapped_risk_upper"], nonvacuous=report["nonvacuous"],
               heldout_gibbs_loss=test["bounded_loss"], heldout_accuracy_pct=100 * test["accuracy"],
               posterior_kl=report["posterior_kl"], deployment_to_training_kl=shift.get("channel_kl"),
               deployment_to_reference_kl=shift.get("reference_kl"),
               reference_log_mgf_upper=shift.get("reference_log_mgf_upper"), shift_method=shift["method"],
               selected_epoch=report["selected_epoch"], test_draws=test["draws"],
               test_examples=test["examples"], empirical_draws=report["empirical"]["iid_observations"],
               failure_each_theorem_and_mc=report["confidence"]["failure_each_theorem_and_mc"],
               family_failure=report["confidence"]["family_failure"], seed=cfg["seed"],
               source_report=str(path.relative_to(root)), source_sha256=file_fingerprint(path)["sha256"],
               training_git_sha=training_revision,
               weights_sha256=report["checkpoint_weights"]["sha256"], bank_sha256=report["saved_bank_hash"])
    return row


def draw_figure(rows, output, mismatch_rho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    style = {"font.size": 8.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.45, 4.0))
        fig.subplots_adjust(left=.095, right=.992, bottom=.32, top=.85, wspace=.12)
        positions = [0., 1.08, 1.75, 2.86, 3.53]
        ymax = max(1., max(row["uncapped_risk_upper"] for row in rows)) * 1.13
        groups = [(positions[0], "Perfect"), (1.415, "Matched"), (3.195, "Mismatch")]
        for ax, width in zip(axes, WIDTHS):
            cells = [row for row in rows if row["width"] == width]
            bottom = [0.] * len(cells)
            for field, (_, color) in COMPONENTS.items():
                heights = [row[field] or 0. for row in cells]
                ax.bar(positions, heights, bottom=bottom, width=.54, color=color, zorder=2)
                bottom = [base + height for base, height in zip(bottom, heights)]
            ax.axhline(1, ls=(0, (4, 2)), color="#333333", linewidth=.85, zorder=1.8)
            ax.scatter(positions, [r["heldout_gibbs_loss"] for r in cells], marker="D", s=20,
                       color="#111111", edgecolors="white", linewidths=.55, zorder=4)
            for x, total in zip(positions, bottom):
                ax.annotate(f"{total:.3f}", (x, total), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.6, zorder=5,
                            bbox=dict(facecolor="white", edgecolor="none", pad=.4))
            ax.set_xticks(positions, [r["certificate"] for r in cells], fontsize=8)
            for x, label in groups:
                ax.text(x, -.13, label, ha="center", va="top", transform=ax.get_xaxis_transform())
            ax.set(xlim=(-.50, 4.03), ylim=(0., ymax), title=rf"$d={width}$")
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", linewidth=.45, color="#DADDE0")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Bounded Gibbs risk")
        fig.suptitle(rf"Two moons   $n={rows[0]['n']}$   $m={rows[0]['m']}$   "
                     rf"$\rho_{{\rm test}}={DEPLOYMENT_RHO}$   $\rho_{{\rm mismatch}}={mismatch_rho:g}$",
                     y=.985, fontsize=10)
        handles = [Patch(facecolor=color, label=label) for label, color in COMPONENTS.values()]
        handles += [Line2D([], [], marker="D", color="#111111", ls="none", markersize=4,
                           label="Held-out Gibbs loss"),
                    Line2D([], [], ls=(0, (4, 2)), color="#333333", lw=.85, label="Risk ceiling (1)")]
        fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, fontsize=7.6,
                   handlelength=1.3, columnspacing=1.3, bbox_to_anchor=(.5, .025))
        for suffix in (".png", ".pdf"):
            fig.savefig(output / (STEM + suffix), dpi=300, facecolor="white")
        plt.close(fig)


def export(root, output):
    rows = [read_case(root, width, case) for width in WIDTHS for case in CASES]
    mismatch_correlations = {row["train_rho_c"] for row in rows if row["regime"] == "mismatch"}
    if len(mismatch_correlations) != 1:
        raise ValueError("All mismatch training cases must share the same gain correlation")
    mismatch_rho, = mismatch_correlations
    output.mkdir(parents=True, exist_ok=True)
    draw_figure(rows, output, mismatch_rho)
    with (output / (STEM + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(source_root=str(root.resolve()), family_size=FAMILY_SIZE, family_confidence=1. - rows[0]["family_failure"],
                   mismatch_train_rho_c=mismatch_rho, test_rho_c=DEPLOYMENT_RHO,
                   nonvacuous=sum(row["nonvacuous"] for row in rows), rows=rows)
    save_json(output / (STEM + ".json"), summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.root, args.output)
    print(f"{summary['nonvacuous']} nonvacuous certificates out of {summary['family_size']}")
