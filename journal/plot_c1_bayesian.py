"""Plot the complete frozen-posterior C1 reference-channel diagnostic.

This is a separate deployment split: the transmitter sends two output logits
through centered-lognormal noise, followed by a fixed softmax receiver. The
erasure comparison retains the hidden-feature link and its discrete mask
metric. These are different channels and their severity axes are not compared.
All 12 declared perfect-trained posteriors are retained; no training occurs.

Only the C1 second-inequality *channel term* is closed form. The empirical
Gibbs risk is estimated with its recorded confidence correction. The direct
lognormal distance is an independent diagnostic and never enters the guarantee.
Every figure is exported as PNG, PDF and EPS, with no confidence shading.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bnn-c1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .c1_reference import erasure_first_shift, lognormal_logit_channel_terms, smooth_loss_lipschitz


METHODS = ("ERM", "L2", "T1", "T2")
SEEDS = (76101, 76102, 76103)
ERASURE_GRID = (0., .01, .05, .1, .25, .5)
LOGNORMAL_GRID = (0., .1, .25, .5, 1., 2.)
COLORS = {"risk": "#0072B2", "bound": "#D55E00", "direct": "#222222",
          "jensen": "#009E73", "moment": "#8064A2", "kl": "#A78C62"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, message):
    if not finite(actual) or not finite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(message)


def fingerprint(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def describe(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


class Evidence:
    """Validate the complete declared family before exporting any figures."""

    def __init__(self, source):
        self.path = Path(source)
        self.source_fingerprint = fingerprint(self.path)
        data = json.loads(self.path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError("Require C1 Bayesian report schema version 1")
        self.confidence = data.get("confidence", {})
        conf = self.confidence
        if conf.get("runs") != 12 or conf.get("total_failure") != .05:
            raise ValueError("Confidence must cover the complete 12-posterior family at failure .05")
        if conf.get("empirical_draws") != 1000 or conf.get("test_draws") != 100:
            raise ValueError("Require fresh 1000-draw empirical and 100-draw held-out Gibbs estimates")
        data_delta, mc_delta = conf.get("data_failure_per_run"), conf.get("empirical_mc_failure_per_run")
        if not all(finite(value) and 0 < value < 1 for value in (data_delta, mc_delta)):
            raise ValueError("Confidence allocations must be positive and finite")
        close(12 * data_delta, conf.get("data_failure_total"), "Data confidence allocation must cover all 12 posteriors")
        close(12 * mc_delta, conf.get("empirical_mc_failure_total"), "MC confidence allocation must cover all 12 posteriors")
        close(12 * (data_delta + mc_delta), .05, "Confidence union must sum to .05")
        self.radius = math.sqrt(math.log(1 / mc_delta) / (2 * conf["empirical_draws"]))
        self.analytic = data.get("analytic", {})
        if self.analytic.get("epsilon") != .01 or self.analytic.get("tau") != .1:
            raise ValueError("Require the declared epsilon=.01, lognormal shape tau=.1 example")
        close(self.analytic.get("K"), smooth_loss_lipschitz(.01), "K must be the uniform two-logit loss Lipschitz constant")
        self.reports = data.get("reports", [])
        if len(self.reports) != 12:
            raise ValueError("Require all 12 frozen perfect-trained posteriors")
        self.groups = defaultdict(list)
        seen = set()
        for row in self.reports:
            key = (row.get("method_label"), row.get("seed"))
            if key in seen or key[0] not in {f"{m}-Perfect" for m in METHODS} or key[1] not in SEEDS:
                raise ValueError("Require distinct ERM/L2/T1/T2-Perfect posteriors with all three paired seeds")
            seen.add(key)
            self.validate_row(row)
            self.groups[key[0].split("-")[0]].append(row)
        if any({row["seed"] for row in self.groups[method]} != set(SEEDS) for method in METHODS):
            raise ValueError("Every method must retain all three paired seeds")
        # Channel-only quantities must be identical across the 12 weight laws.
        for law, keys in (("erasure", ("shift_first_exact",)),
                          ("lognormal", ("shift_first_mc_diagnostic", "shift_first_mc_se", "shift_second_closed",
                                         "log_reference_moment", "reference_kl", "jensen_shift_upper"))):
            reference = self.reports[0][law]
            for row in self.reports[1:]:
                for case, expected in zip(row[law], reference):
                    for key in keys:
                        close(case[key], expected[key], "Channel-only terms must share one independent benchmark across the family")
        hashes = data.get("source_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Retain source fingerprints for frozen-checkpoint provenance")
        self.source_hashes, self.unavailable_sources = hashes, []
        for name, expected in hashes.items():
            if (not isinstance(name, str) or not isinstance(expected, dict)
                    or not isinstance(expected.get("sha256"), str)
                    or len(expected["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in expected["sha256"])
                    or not isinstance(expected.get("bytes"), int) or expected["bytes"] < 1):
                raise ValueError("Source fingerprints must have a SHA256 and positive byte count")
            path = Path(name)
            if path.is_file():
                if fingerprint(path) != expected:
                    raise ValueError(f"Source fingerprint differs: {name}")
            else:
                self.unavailable_sources.append(name)
        if fingerprint(self.path) != self.source_fingerprint:
            raise ValueError("The C1 report changed during validation")

    def validate_row(self, row):
        for key in ("empirical_bounded_risk_mc", "empirical_mc_radius", "data_complexity", "baseline_risk_upper_uncapped"):
            if not finite(row.get(key)) or row[key] < 0:
                raise ValueError("Every empirical-risk and complexity term must be finite and nonnegative")
        if row["empirical_bounded_risk_mc"] > 1:
            raise ValueError("The bounded empirical risk must lie in [0,1]")
        close(row["empirical_mc_radius"], self.radius, "Empirical risk must use the declared simultaneous-family MC correction")
        baseline = min(1., row["empirical_bounded_risk_mc"] + row["empirical_mc_radius"]) + row["data_complexity"]
        close(row["baseline_risk_upper_uncapped"], baseline, "Baseline must include empirical MC upper risk and data complexity")
        for law, grid in (("erasure", ERASURE_GRID), ("lognormal", LOGNORMAL_GRID)):
            cases = row.get(law, [])
            if tuple(case.get("severity") for case in cases) != grid:
                raise ValueError(f"Require the complete ordered {law} severity grid")
            for case in cases:
                for key in ("test_accuracy", "test_bounded_risk"):
                    if not finite(case.get(key)) or not 0 <= case[key] <= 1:
                        raise ValueError("Held-out accuracy and bounded risk must lie in [0,1]")
                if law == "erasure":
                    shift = erasure_first_shift(8, case["severity"])
                    close(case.get("shift_first_exact"), shift, "Erasure first-distance term must equal 1-(1-p)^8")
                else:
                    terms = lognormal_logit_channel_terms(case["severity"], self.analytic["tau"], self.analytic["epsilon"])
                    for field, key in (("shift_second_closed", "c1_second_shift_upper"),
                                       ("log_reference_moment", "log_exponential_moment"),
                                       ("reference_kl", "channel_kl"), ("jensen_shift_upper", "jensen_shift_upper")):
                        close(case.get(field), terms[key], "Lognormal second-inequality term must use the exact fixed-Gaussian reference")
                    for key in ("shift_first_mc_diagnostic", "shift_first_mc_se"):
                        if not finite(case.get(key)) or case[key] < 0:
                            raise ValueError("Direct-distance diagnostic mean and standard error must be finite and nonnegative")
                    if case["severity"] == 0 and (case["shift_first_mc_diagnostic"] or case["shift_first_mc_se"]):
                        raise ValueError("The zero-noise diagnostic distance must be zero")
                    shift = terms["c1_second_shift_upper"]
                close(case.get("risk_upper_uncapped"), baseline + shift,
                      "Population-risk RHS must add the exact/analytic channel shift to the full baseline; diagnostic MC never enters it")


def style(ax):
    ax.grid(color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def finish(fig, output, name, data, index):
    paths = []
    for extension in ("png", "pdf", "eps"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None, bbox_inches="tight", pad_inches=.07)
        paths.append(str(path.resolve()))
    plt.close(fig)
    index["figures"].append({"name": name, "files": paths, "data": data,
        "uncertainty_display": "No confidence shading or error bars; risk curves average all three declared seeds"})


def channel_terms(evidence, output, index):
    row = evidence.reports[0]
    erasure, lognormal = row["erasure"], row["lognormal"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    axes[0].plot(ERASURE_GRID, [case["shift_first_exact"] for case in erasure], "o-", color=COLORS["risk"],
                 markersize=4, label=r"Exact first term $1-(1-p)^8$")
    axes[0].set(title="(a) Erasure: exact first inequality", xlabel="Hidden-feature erasure probability p",
                ylabel="C1 channel-shift term", ylim=(0, 1.05))
    axes[0].legend(loc="lower right")
    for field, label, color, marker, linestyle in (
            ("shift_first_mc_diagnostic", "First term: independent MC diagnostic", "direct", "o", "-"),
            ("shift_second_closed", "Second term: closed-form guarantee", "bound", "s", "-"),
            ("jensen_shift_upper", "Jensen analytic comparison", "jensen", "^", "--"),
            ("log_reference_moment", "Exact Gaussian log moment", "moment", None, ":"),
            ("reference_kl", "Exact lognormal-to-Gaussian KL", "kl", None, "-.")):
        axes[1].plot(LOGNORMAL_GRID, [case[field] for case in lognormal], label=label,
                     color=COLORS[color], marker=marker, markersize=4, linestyle=linestyle, linewidth=1.3)
    axes[1].set(title="(b) Output logits: centered-lognormal noise", xlabel=r"Noise standard deviation $\sigma$ per logit")
    axes[1].set_ylim(0, max(case["shift_second_closed"] for case in lognormal) * 1.08)
    axes[1].legend(loc="upper left", fontsize=7.3)
    for ax in axes:
        style(ax)
    fig.suptitle(r"C1 channel terms: smooth bounded loss, $\epsilon=0.01$, $\tau=0.1$")
    fig.text(.5, .004, "The panels use different splits and metrics. The first lognormal distance has no simple known closed form; its MC is diagnostic only.\n"
             "Fixed reference Q = N(0, sigma^2 I_2): second term = exact log moment + KL; at sigma=0 the shift is exactly zero.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .105, 1, .96))
    finish(fig, output, "c1_channel_terms", {"erasure": erasure, "lognormal": lognormal}, index)


def risk_curves(evidence, output, index, law):
    grid = ERASURE_GRID if law == "erasure" else LOGNORMAL_GRID
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.5), sharex=True, sharey=True)
    data, tallest = [], 1.
    for method, ax in zip(METHODS, axes.flat):
        rows = sorted(evidence.groups[method], key=lambda row: row["seed"])
        cells = []
        for position, severity in enumerate(grid):
            stats = {key: describe(row[law][position][key] for row in rows)
                     for key in ("test_bounded_risk", "risk_upper_uncapped", "test_accuracy")}
            cells.append({"severity": severity, "statistics": stats})
        risk = [cell["statistics"]["test_bounded_risk"]["mean"] for cell in cells]
        bound = [cell["statistics"]["risk_upper_uncapped"]["mean"] for cell in cells]
        tallest = max(tallest, *bound, *risk)
        ax.plot(grid, bound, "s-", color=COLORS["bound"], markersize=4, label="Population-risk RHS (uncapped)")
        ax.plot(grid, risk, "o-", color=COLORS["risk"], markersize=4, label="Held-out Gibbs risk estimate")
        ax.axhline(1., color="#777777", linestyle=":", linewidth=.8, label="Vacuity threshold")
        ax.set_title(f"{method}-Perfect (all three seeds)")
        style(ax)
        data.append({"method_label": f"{method}-Perfect", "seeds": [row["seed"] for row in rows], "points": cells})
    for ax in axes.flat:
        ax.set_ylim(0, 1.08 * tallest)
    for ax in axes[:, 0]:
        ax.set_ylabel("Smooth loss / uncapped risk RHS")
    xlabel = "Hidden-feature erasure probability p" if law == "erasure" else r"Output-logit noise standard deviation $\sigma$"
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .947), ncol=3, fontsize=7.6)
    channel_description = "exact C1 first term; hidden-feature erasure" if law == "erasure" else "closed-form C1 second term; additive two-logit distortion"
    fig.suptitle("C1 population-risk guarantee: " + channel_description, y=.994, fontsize=11)
    fig.text(.5, .007, "All 12 frozen perfect-trained Gaussian posteriors; each curve averages the same three paired seeds.\n"
             "RHS = empirical Gibbs risk + MC correction + data complexity + channel term (with empirical upper risk capped at 1).\n"
             "Only the channel term is closed form. Held-out estimates are diagnostics; values above 1 are vacuous.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .103, 1, .91))
    finish(fig, output, f"c1_{law}_population_risk", data, index)


def export(source, output):
    evidence = Evidence(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 1, "source_report": str(evidence.path.resolve()),
             "source_report_fingerprint": evidence.source_fingerprint, "source_hashes": evidence.source_hashes,
             "unavailable_original_source_paths": evidence.unavailable_sources,
             "confidence": evidence.confidence, "analytic": evidence.analytic,
             "scope": "C1 channel tractability diagnostic; complete frozen 12-posterior family; separate output-logit deployment split",
             "closed_form_claim": "Only the second-inequality channel term is analytic; empirical risk has its simultaneous-family MC correction",
             "figures": []}
    channel_terms(evidence, output, index)
    for law in ("erasure", "lognormal"):
        risk_curves(evidence, output, index, law)
    if fingerprint(evidence.path) != evidence.source_fingerprint:
        raise ValueError("The C1 report changed during export")
    (output / "figure_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="C1 Bayesian risk report JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for PNG/PDF/EPS and provenance index")
    args = parser.parse_args()
    result = export(args.source, args.output)
    print(json.dumps({"figures": len(result["figures"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
