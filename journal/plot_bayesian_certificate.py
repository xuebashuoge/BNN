"""Plot the separate 48-checkpoint small-model population-risk demonstration.

Consumes the fresh-bank evaluator's risk_bounds.json and hash-linked per-row
reports. No training, checkpoint loading, or inference occurs. This small
two-moons model is explicitly separated from the main accuracy experiments.
Every figure is saved as PNG, PDF, and EPS without confidence shading.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bnn-bayesian-certificate-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .run_experiment import bounded_loss_description, normalize_config


METHODS = ("ERM", "L2", "T1", "T2")
REGIMES = ("matched", "perfect")
LAWS = ("analog", "outage")
RUNS, DRAWS, TEST_DRAWS, DELTA = 48, 1000, 20, .025
COLORS = dict(ERM="#666666", L2="#B89256", T1="#0072B2", T2="#D55E00",
              empirical="#696969", mc="#CBD0D5", shift="#E69F00",
              channel="#0072B2", data="#009E73", joint="#CC79A7")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values):
    values = list(values)
    if not values or not all(finite(value) for value in values):
        raise ValueError("Every declared seed observation must be finite")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None, values=values)


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def close(actual, expected, message):
    if not finite(actual) or abs(actual - expected) > 1e-10:
        raise ValueError(message)


def checked_fingerprint(value):
    return (isinstance(value, dict) and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in value["sha256"])
            and isinstance(value.get("bytes"), int) and value["bytes"] > 0)


class Evidence:
    def __init__(self, source):
        self.path = Path(source)
        self.source_fingerprint = fingerprint(self.path)
        data = json.loads(self.path.read_text())
        if (data.get("schema_version") != 2 or data.get("total_runs") != RUNS
                or data.get("delta") != DELTA or data.get("draws") != DRAWS):
            raise ValueError("Require the declared 48-run, delta=.025, 1000-draw certificate family")
        self.rows, self.definitions = data["rows"], data.get("definitions", {})
        if len(self.rows) != RUNS or len({row["task_id"] for row in self.rows}) != RUNS:
            raise ValueError("The small-model report needs exactly 48 distinct checkpoints")
        self.sources = {str(self.path.resolve()): self.source_fingerprint}
        self.groups = defaultdict(list)
        data_signatures = set()
        for row in self.rows:
            self.validate_row(row)
            method, regime, law = row["method_label"].split("-", 1)[0], row["training_regime"], row["channel_name"]
            self.groups[(law, method, regime)].append(row)
            cfg = normalize_config(row["config"])
            data_signatures.add((cfg["data_seed"], cfg["moon_noise"], cfg["prior_std"],
                                 cfg["training_loss"], cfg["loss_epsilon"]))
        expected = {(law, method, regime) for law in LAWS for method in METHODS for regime in REGIMES}
        if set(self.groups) != expected or any(len(rows) != 3 for rows in self.groups.values()):
            raise ValueError("Retain all four methods, both laws/regimes, and all three seeds")
        paired_seeds = {tuple(sorted((row["seed"], row["config"]["channel_seed"]) for row in rows))
                        for rows in self.groups.values()}
        if len(paired_seeds) != 1 or len({seed for seed, _ in next(iter(paired_seeds))}) != 3:
            raise ValueError("All 16 comparisons require the same three distinct paired seeds")
        if len(data_signatures) != 1:
            raise ValueError("The demonstration must share one data split, prior, noise, and loss definition")
        if fingerprint(self.path) != self.source_fingerprint:
            raise ValueError("The aggregate changed while its evidence was validated")

    def validate_row(self, row):
        cfg = normalize_config(row["config"])
        if (row.get("stage") != "bayes_certificate" or row.get("dataset") != "moons"
                or cfg["dataset"] != "moons" or cfg["architecture"] != "mlp"
                or cfg["width"] != 8 or cfg["train_size"] != 512 or cfg["val_size"] != 512
                or cfg["m"] != 2000 or row.get("n") != 512 or row.get("m") != 2000
                or row.get("width") != 8 or cfg["moons_sampler"] != "iid"
                or cfg["predictor"] != "gaussian" or cfg["training_loss"] != "bounded_ce_smooth"
                or cfg["loss_epsilon"] != .01 or row.get("seed") != cfg["seed"]):
            raise ValueError("This report is exclusively the IID moons MLP d=8, n=512, m=2000 demonstration")
        law, regime = row.get("channel_name"), row.get("training_regime")
        if law not in LAWS or regime not in REGIMES:
            raise ValueError("Unknown deployment law or training regime")
        method = row["method_label"].split("-", 1)[0]
        if row["method_label"] != f"{method}-{regime.title()}" or method not in METHODS:
            raise ValueError("The small-model report needs the standard eight method labels")
        algorithms = dict(ERM="erm", L2="l2", T1="proposed", T2="joint")
        if cfg["method"] != algorithms[method]:
            raise ValueError("Method label differs from the trained objective")
        deployment = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.) if law == "analog" else dict(kind="outage", outage_prob=.5)
        if cfg["deployment_channel"] != deployment or row["deployment_channel"] != deployment:
            raise ValueError("Deployment channel differs from the declared small-model law")
        training = dict(kind="perfect") if regime == "perfect" else deployment
        if cfg["train_channel"] != training or row["train_channel"] != training:
            raise ValueError("Training channel differs from its matched/perfect label")
        if row.get("certificate_draws") != DRAWS or row.get("full_test_draws") != TEST_DRAWS:
            raise ValueError("Use fresh 1000-draw bank inference and the saved 20-draw full test")
        for metric in ("accuracy", "test_bounded_risk", "empirical_bounded_risk"):
            if not finite(row.get(metric)) or not 0 <= row[metric] <= 1:
                raise ValueError("Accuracy and bounded Gibbs risks must lie in [0,1]")
        for metric in ("posterior_kl_float64", "channel_complexity", "sample_complexity", "joint_complexity"):
            if not finite(row.get(metric)) or row[metric] < 0:
                raise ValueError("Every posterior must have finite nonnegative Gaussian complexities")
        shift = row["shift"]["channel_shift_complexity"]
        if not finite(shift) or not 0 <= shift <= 1:
            raise ValueError("Require the recorded uniform bounded-loss channel shift")
        if regime == "matched" and shift != 0:
            raise ValueError("Matched channel laws must have zero shift")
        if regime == "perfect" and row["channel_complexity"] != 0:
            raise ValueError("Perfect training must have zero channel complexity")
        confidence = row["confidence_allocation"]
        if (confidence.get("total_failure") != DELTA or confidence.get("simultaneous_runs") != RUNS
                or confidence.get("simultaneous_theorems") != 2):
            raise ValueError("Certificate confidence must cover the complete declared family")
        theorem_delta, mc_delta = confidence["failure_per_theorem_run"], confidence["mc_failure_per_run"]
        if not finite(theorem_delta) or not finite(mc_delta) or min(theorem_delta, mc_delta) <= 0:
            raise ValueError("Require positive theorem and Monte Carlo confidence allocations")
        close(2 * RUNS * theorem_delta + RUNS * mc_delta, DELTA, "Confidence union does not reproduce delta=.025")
        radius = math.sqrt(math.log(1 / mc_delta) / (2 * DRAWS))
        close(confidence["empirical_mc_upper_radius"], radius, "Monte Carlo radius must use 1000 full-loader draws")
        upper = min(1., row["empirical_bounded_risk"] + radius)
        close(row["empirical_bounded_risk_upper"], upper, "Empirical upper risk omits or misapplies the fresh MC correction")
        for key, complexity in (("theorem_1", row["channel_complexity"] + row["sample_complexity"]),
                                ("theorem_2", row["joint_complexity"])):
            bound = row[key]
            rhs = upper + shift + complexity
            close(bound["rhs_uncapped"], rhs, "Population-risk RHS must include empirical upper risk, shift, and unweighted complexities")
            close(bound["population_risk_upper"], min(1., rhs), "Clipped risk upper bound disagrees with its uncapped RHS")
            if bound["nonvacuous"] is not (rhs < 1):
                raise ValueError("Nonvacuity label disagrees with the population-risk RHS")
        bank = row["certificate_empirical_gibbs_mc"]
        if (bank.get("draws") != DRAWS or bank.get("examples") != 512
                or bank.get("channel_source") != "bank"
                or bank.get("bounded_loss_definition") != bounded_loss_description(cfg)
                or bank.get("predictor", {}).get("family") != "diagonal_gaussian"
                or bank.get("predictor", {}).get("stochastic") is not True
                or bank.get("posterior_rule") != "one weight draw fixed across entire loader per replicate"):
            raise ValueError("Fresh empirical inference must evaluate the declared Gaussian Gibbs loss on its saved bank")
        close(bank["bounded_loss"], row["empirical_bounded_risk"], "Aggregate empirical mean differs from its bank report")
        replicates = bank.get("replicate_statistics", [])
        if len(replicates) != DRAWS or not all(finite(rep.get("bounded_loss")) and 0 <= rep["bounded_loss"] <= 1 for rep in replicates):
            raise ValueError("Fresh empirical Monte Carlo requires all 1000 bounded full-loader replicate observations")
        close(statistics.fmean(rep["bounded_loss"] for rep in replicates), bank["bounded_loss"], "Fresh empirical mean disagrees with saved replicate observations")
        for key in ("metadata", "weights", "evaluation", "history"):
            if not checked_fingerprint(row.get("source_fingerprints", {}).get(key)):
                raise ValueError("Every certificate must retain checkpoint/test/history source fingerprints")
        expected_fingerprint = row.get("certificate_report_fingerprint")
        if not checked_fingerprint(expected_fingerprint):
            raise ValueError("The aggregate must hash-link its fresh certificate report")
        candidates = [self.path.parent / "rows" / f"{row['task_id']}.json", Path(row["certificate_report"])]
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing hash-linked per-checkpoint certificate report: {row['task_id']}")
        if fingerprint(path) != expected_fingerprint:
            raise ValueError("Fresh certificate report fingerprint differs from the aggregate")
        saved = json.loads(path.read_text())
        for key in ("task_id", "config", "empirical_bounded_risk", "theorem_1", "theorem_2", "source_fingerprints"):
            if saved.get(key) != row.get(key):
                raise ValueError("Fresh certificate report content differs from aggregate evidence")
        self.sources[str(path.resolve())] = expected_fingerprint


def style(ax):
    ax.grid(color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def finish(fig, output, name, data, index):
    files = []
    for extension in ("png", "pdf", "eps"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None, bbox_inches="tight", pad_inches=.06)
        files.append(str(path.resolve()))
    plt.close(fig)
    index["figures"].append(dict(name=name, files=files, data=data,
        uncertainty_display="No confidence shading; every displayed point/bar averages all three declared paired seeds"))


def title(law):
    return f"Two moons, small model: hidden width d=8, n=512, m=2000; {law} deployment"


def comparisons(evidence, output, index):
    keys = [(method, regime) for method in METHODS for regime in REGIMES]
    labels = [f"{method}\n{regime}" for method, regime in keys]
    for law in LAWS:
        fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.6))
        data = []
        for position, (method, regime) in enumerate(keys):
            rows = sorted(evidence.groups[(law, method, regime)], key=lambda row: row["seed"])
            stats = dict(accuracy_percent=describe(row["accuracy"]*100 for row in rows),
                         test_bounded_risk=describe(row["test_bounded_risk"] for row in rows),
                         theorem_1_rhs=describe(row["theorem_1"]["rhs_uncapped"] for row in rows),
                         theorem_2_rhs=describe(row["theorem_2"]["rhs_uncapped"] for row in rows))
            axes[0].bar(position, stats["accuracy_percent"]["mean"], width=.72, color=COLORS[method],
                        hatch="//" if regime == "perfect" else None)
            axes[1].plot(position-.16, stats["test_bounded_risk"]["mean"], "D", color="black",
                         label="Held-out Gibbs risk estimate" if position == 0 else None, markersize=4)
            axes[1].plot(position, stats["theorem_1_rhs"]["mean"], "o", color=COLORS["T1"],
                         label="T1 population-risk RHS" if position == 0 else None, markersize=5)
            axes[1].plot(position+.16, stats["theorem_2_rhs"]["mean"], "s", color=COLORS["T2"],
                         markerfacecolor="white", label="T2 population-risk RHS" if position == 0 else None, markersize=5)
            data.append(dict(method=method, training_regime=regime, statistics=stats,
                             seeds=[row["seed"] for row in rows], task_ids=[row["task_id"] for row in rows]))
        for ax in axes:
            ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
            style(ax)
        axes[0].set_ylabel("Deployment test accuracy (%)")
        axes[0].set_ylim(0, 100)
        axes[1].set_ylabel("Smooth loss / uncapped risk RHS")
        axes[1].axhline(1, color="#777777", linestyle=":", linewidth=.8)
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="best", fontsize=7)
        fig.suptitle(title(law))
        fig.text(.5, -.012, "Means over three paired seeds. Test: 20 Gibbs draws; fresh empirical bank risk: 1000 draws.\n"
                 "RHS includes empirical MC upper risk, unweighted complexity, and channel shift; values above 1 are vacuous.",
                 ha="center", fontsize=7.3)
        fig.tight_layout(rect=(0, .065, 1, .96))
        finish(fig, output, f"small_model_{law}_accuracy_population_bounds", data, index)


def decompositions(evidence, output, index):
    components = ("empirical", "mc", "shift", "channel", "data", "joint")
    for law in LAWS:
        fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
        data = []
        for ax, regime in zip(axes, REGIMES):
            heights = {key: [] for key in components}
            risks, labels = [], []
            for method in METHODS:
                rows = sorted(evidence.groups[(law, method, regime)], key=lambda row: row["seed"])
                for theorem in (1, 2):
                    stacks = dict(
                        empirical=describe(row["empirical_bounded_risk"] for row in rows),
                        mc=describe(row["empirical_bounded_risk_upper"]-row["empirical_bounded_risk"] for row in rows),
                        shift=describe(row["shift"]["channel_shift_complexity"] for row in rows),
                        channel=describe(row["channel_complexity"] if theorem == 1 else 0. for row in rows),
                        data=describe(row["sample_complexity"] if theorem == 1 else 0. for row in rows),
                        joint=describe(row["joint_complexity"] if theorem == 2 else 0. for row in rows))
                    rhs = describe(row[f"theorem_{theorem}"]["rhs_uncapped"] for row in rows)
                    close(sum(item["mean"] for item in stacks.values()), rhs["mean"], "Risk stack does not reproduce the population-risk RHS")
                    for component in components:
                        heights[component].append(stacks[component]["mean"])
                    test = describe(row["test_bounded_risk"] for row in rows)
                    risks.append(test["mean"])
                    labels.append(f"{method}\nT{theorem}")
                    data.append(dict(method=method, regime=regime, theorem=theorem,
                        components=stacks, population_risk_rhs=rhs, test_bounded_risk=test,
                        task_ids=[row["task_id"] for row in rows]))
            xs, bottom = np.arange(len(labels)), np.zeros(len(labels))
            for component in components:
                ax.bar(xs, heights[component], bottom=bottom, width=.7, color=COLORS[component])
                bottom += np.array(heights[component])
            ax.plot(xs, risks, "D", color="black", markersize=3.5)
            ax.axhline(1, color="#777777", linestyle=":", linewidth=.8)
            ax.set_xticks(xs, labels)
            ax.set_title(f"{regime.title()} training")
            ax.set_ylabel("Uncapped population-risk RHS")
            # Empty final stack segments can make the bar total a sticky
            # upper edge. Set explicit headroom so every full RHS is visible.
            ax.set_ylim(0., 1.08 * max(1., float(bottom.max()), max(risks)))
            style(ax)
        handles = [Patch(color=COLORS[key], label=label) for key, label in (
            ("empirical", "Fresh empirical risk"), ("mc", "MC correction"), ("shift", "Channel shift"),
            ("channel", "Channel term"), ("data", "Data term"), ("joint", "Joint term"))]
        handles.append(plt.Line2D([], [], color="black", marker="D", linestyle="none", markersize=4,
                                  label="Held-out risk estimate"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4,
                   columnspacing=1.1, handlelength=1.5, fontsize=7.5)
        fig.suptitle(title(law))
        fig.text(.5, -.01, "Every frozen posterior is evaluated with both theorem RHSs; bars average all three paired seeds.\n"
                 r"MC/theorem confidence is allocated jointly over 48 checkpoints and both theorems ($\delta=0.025$). Bounds concern smooth Gibbs loss, not classification error.",
                 ha="center", fontsize=7.1)
        fig.tight_layout(rect=(0, .07, 1, .79), w_pad=1.7)
        finish(fig, output, f"small_model_{law}_risk_bound_decomposition", data, index)


def export(source, output):
    evidence = Evidence(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = dict(schema_version=1, figures=[], scientific_setup="Separate small-model bound demonstration; never pooled with main accuracy runs",
        total_runs=RUNS, bank_draws=DRAWS, test_draws=TEST_DRAWS, delta=DELTA,
        source_fingerprints=evidence.sources, definitions=evidence.definitions,
        scope="Population-risk RHS = corrected empirical bounded risk + uniform shift + unweighted theorem complexity; plots retain uncapped RHSs")
    comparisons(evidence, output, index)
    decompositions(evidence, output, index)
    (output / "figure_index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Fresh-bank evaluator's risk_bounds.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = export(args.source, args.output)
    print(json.dumps(dict(figures=len(index["figures"]), runs=RUNS, output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
