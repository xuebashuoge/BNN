"""Plot complete, declared all-Gaussian experiments from saved JSON only.

Use a final confirmation or sweep manifest, not a tuning manifest. Every declared
task and seed must have a matching full test evaluation before figures are
written. Each source stays separate. Figures contain means and point markers,
without confidence bands, and are exported as PNG, PDF, and EPS.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bnn-bayesian-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .run_experiment import bounded_loss_description, normalize_config


METHODS = ("ERM", "L2", "T1", "T2")
COLORS = dict(ERM="#666666", L2="#B89256", T1="#0072B2", T2="#D55E00")
MARKERS = dict(ERM="o", L2="s", T1="^", T2="D")
EXECUTION_SETTINGS = {"device", "threads", "download"}
EVALUATION_SETTINGS = EXECUTION_SETTINGS | {"batch_size", "eval_seed", "eval_draws"}
EXPERIMENTS = {"main", "m_sweep", "p_train_sweep"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def summary(values):
    values = list(values)
    if not values or not all(finite(value) for value in values):
        raise ValueError("Every declared observation must be finite")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None,
                values=values)


def matching_config(actual, expected, excluded):
    actual, expected = normalize_config(actual), normalize_config(expected)
    return {k: v for k, v in actual.items() if k not in excluded} == {
        k: v for k, v in expected.items() if k not in excluded}


def base_method(label):
    method = label.split("-", 1)[0]
    if method not in METHODS:
        raise ValueError(f"Unexpected method label: {label}")
    return method


class Evidence:
    """Load exactly one manifest's results; never silently drop missing seeds."""

    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), Path(manifest)
        self.files, self.rows = {}, []
        raw = self.manifest.read_bytes()
        self.files[str(self.manifest.resolve())] = hashlib.sha256(raw).hexdigest()
        entries = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not entries or any(entry.get("experiment") not in EXPERIMENTS for entry in entries):
            raise ValueError("Plot a final main/m_sweep/p_train_sweep manifest with experiment metadata")
        declared = {entry["task_id"]: entry for entry in entries}
        if len(declared) != len(entries):
            raise ValueError("Duplicate manifest task IDs")
        results = self.read(self.root / "test_results.json")
        if not isinstance(results, list):
            raise ValueError("test_results.json must contain a list")
        by_id = {row["task_id"]: row for row in results}
        if len(by_id) != len(results):
            raise ValueError("Duplicate test task IDs")
        missing = set(declared) - set(by_id)
        if missing:
            raise ValueError(f"Incomplete declared test coverage: {len(missing)} missing tasks; first={sorted(missing)[0]}")
        # Extra rows are excluded explicitly, never pooled with declared evidence.
        self.excluded_task_ids = sorted(set(by_id) - set(declared))
        for task_id, entry in declared.items():
            row = by_id[task_id]
            if row.get("test_status") != "complete":
                raise ValueError(f"Incomplete test evaluation: {task_id}")
            for key in ("group", "method_label", "variant_id", "family"):
                if row.get(key) != entry.get(key):
                    raise ValueError(f"Manifest/result {key} mismatch: {task_id}")
            if not matching_config(row["config"], entry["config"], EXECUTION_SETTINGS):
                raise ValueError(f"Manifest/result configuration mismatch: {task_id}")
            config = normalize_config(row["config"])
            if config.get("predictor") != "gaussian":
                raise ValueError(f"All-Bayesian figures require predictor='gaussian': {task_id}")
            if row.get("seed") != config["seed"]:
                raise ValueError(f"Result seed differs from its checkpoint configuration: {task_id}")
            train, deployment = config["train_channel"], config["deployment_channel"]
            expected_channel = "outage" if deployment["kind"] == "outage" else "analog"
            if entry["channel_name"] != expected_channel:
                raise ValueError(f"Channel metadata differs from deployment law: {task_id}")
            regime = "perfect" if train["kind"] == "perfect" else "matched" if train == deployment else "mismatch"
            if entry["training_regime"] != regime:
                raise ValueError(f"Training-regime metadata differs from channel laws: {task_id}")
            axis, point = entry["sweep_axis"], entry["sweep_value"]
            if entry["experiment"] == "m_sweep" and (axis != "m" or point != config["m"]):
                raise ValueError(f"m-sweep metadata differs from checkpoint configuration: {task_id}")
            if entry["experiment"] == "p_train_sweep" and (
                    axis != "p_train" or train["kind"] not in {"outage", "perfect"}
                    or point != train.get("outage_prob", 0.)):
                raise ValueError(f"p_train-sweep metadata differs from checkpoint configuration: {task_id}")
            if entry["experiment"] == "main" and (axis is not None or point is not None):
                raise ValueError(f"Main comparison unexpectedly carries sweep metadata: {task_id}")
            candidates = [self.root / task_id / "test_full" / "evaluation.json",
                          self.root / task_id / "evaluation.json"]
            if row.get("test_report"):
                candidates.append(Path(row["test_report"]))
            path = next((path for path in candidates if path.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing full test evaluation: {task_id}")
            report = self.read(path)
            self.validate_report(report, row)
            self.rows.append({**row, **{key: entry[key] for key in (
                "experiment", "channel_name", "training_regime", "sweep_axis", "sweep_value")},
                "_report": report, "_report_path": str(path.resolve())})
        self.validate_design()

    def read(self, path):
        raw = Path(path).read_bytes()
        self.files[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    @staticmethod
    def validate_report(report, row):
        task_id = row["task_id"]
        if (report.get("split") != "test" or report.get("risk_scope") != "full"
                or Path(report["checkpoint"]).name != task_id
                or not matching_config(report["evaluation_config"], row["config"], EVALUATION_SETTINGS)):
            raise ValueError(f"Full test evaluation does not match declared checkpoint: {task_id}")
        config = normalize_config(report["evaluation_config"])
        heldout = report["heldout_gibbs_mc"]
        if not finite(row.get("test_accuracy")) or abs(heldout["accuracy"] - row["test_accuracy"]) > 1e-10:
            raise ValueError(f"Saved accuracy summary/report mismatch: {task_id}")
        risks = (
            (heldout, "deployment", 10000),
            (report["empirical_gibbs_mc"], "bank", config["train_size"]),
            (report["same_training_data_fresh_training_channel_gibbs_mc"], "deployment", config["train_size"]),
            (report["heldout_fresh_training_channel_gibbs_mc"], "deployment", 10000))
        for risk, source, examples in risks:
            predictor = risk.get("predictor", {})
            if predictor.get("stochastic") is not True or predictor.get("family") != "diagonal_gaussian":
                raise ValueError(f"Deterministic or unspecified predictor in all-Bayesian figure: {task_id}")
            if risk.get("posterior_rule") != "one weight draw fixed across entire loader per replicate":
                raise ValueError(f"Evaluation must hold one Gaussian draw across the entire loader: {task_id}")
            if risk.get("channel_source") != source or risk.get("examples") != examples:
                raise ValueError(f"Evaluation channel source or example count differs from declared risk: {task_id}")
            if risk.get("draws") != config["eval_draws"] or len(risk.get("replicate_statistics", [])) != config["eval_draws"]:
                raise ValueError(f"Evaluation draw count differs from its saved replicates: {task_id}")
            if risk.get("bounded_loss_definition") != bounded_loss_description(config):
                raise ValueError(f"Evaluation loss definition differs from checkpoint configuration: {task_id}")
            if not all(finite(risk.get(key)) and 0 <= risk[key] <= 1 for key in ("accuracy", "bounded_loss")):
                raise ValueError(f"Invalid saved bounded risk: {task_id}")
            for metric in ("accuracy", "bounded_loss", "cross_entropy"):
                values = [replicate.get(metric) for replicate in risk["replicate_statistics"]]
                if not finite(risk.get(metric)) or not all(finite(item) for item in values) or abs(statistics.fmean(values) - risk[metric]) > 1e-10:
                    raise ValueError(f"Evaluation mean differs from its saved replicates: {task_id}")
        terms = report["final_posterior_complexities"]
        if not all(finite(terms.get(key)) and terms[key] >= 0 for key in (
                "kl", "channel_complexity", "sample_complexity", "joint_complexity")):
            raise ValueError(f"Missing finite Gaussian complexity: {task_id}")
        shift = report["channel_shift_diagnostic"].get("channel_shift_complexity")
        if not finite(shift) or not 0 <= shift <= 1:
            raise ValueError(f"Missing bounded-loss channel shift term: {task_id}")
        for metric in ("bounded_loss", "cross_entropy"):
            signed = report["signed_gap_decomposition_mc"][metric]
            if not all(finite(signed.get(key)) for key in (
                    "channel_generalization", "data_generalization", "channel_shift", "observed_gap", "sum")):
                raise ValueError(f"Incomplete risk decomposition: {task_id}")
            summed = sum(signed[key] for key in ("channel_generalization", "data_generalization", "channel_shift"))
            if abs(summed - signed["observed_gap"]) > 1e-10 or abs(summed - signed["sum"]) > 1e-10:
                raise ValueError(f"Risk decomposition does not telescope: {task_id}")
            if abs(signed["observed_gap"] - (heldout[metric] - report["empirical_gibbs_mc"][metric])) > 1e-10:
                raise ValueError(f"Risk decomposition disagrees with saved risks: {task_id}")
            empirical, fresh_train, fresh_heldout = (risk[metric] for risk, _, _ in risks[1:])
            components = dict(channel_generalization=fresh_train-empirical,
                              data_generalization=fresh_heldout-fresh_train,
                              channel_shift=heldout[metric]-fresh_heldout)
            if any(abs(signed[key]-expected) > 1e-10 for key, expected in components.items()):
                raise ValueError(f"Risk decomposition components disagree with saved risks: {task_id}")

    def validate_design(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[(row["experiment"], row["group"])].append(row)
        for (experiment, group), rows in groups.items():
            expected = {(method, regime) for method in METHODS for regime in ("matched", "perfect")} if experiment == "main" else None
            actual = {(base_method(row["method_label"]), row["training_regime"]) for row in rows}
            if expected is not None and expected != actual:
                raise ValueError(f"Main comparison requires all four methods and both training regimes: {group}")
            if {base_method(row["method_label"]) for row in rows} != set(METHODS):
                raise ValueError(f"Comparison must retain ERM, L2, T1, and T2: {group}")
            variants, seeds, signatures = defaultdict(set), defaultdict(list), set()
            for row in rows:
                key = (base_method(row["method_label"]), row["training_regime"])
                variants[key].add(row["variant_id"])
                seeds[key].append((row["config"]["seed"], row["config"]["channel_seed"]))
                cfg = row["config"]
                signatures.add(json.dumps({key: cfg[key] for key in (
                    "dataset", "architecture", "width", "train_size", "val_size", "m", "data_seed",
                    "deployment_channel", "prior_std", "training_loss", "loss_epsilon", "clip_ce")}, sort_keys=True))
            if len(signatures) != 1:
                raise ValueError(f"Comparison mixes data/model/channel/prior/loss settings: {group}")
            if any(len(value) != 1 for value in variants.values()):
                raise ValueError(f"Multiple variants per method; plot a frozen confirmation manifest: {group}")
            if any(len(set(value)) != len(value) for value in seeds.values()):
                raise ValueError(f"Duplicate initialization/bank seed pair: {group}")
            if len({tuple(sorted(value)) for value in seeds.values()}) != 1:
                raise ValueError(f"All methods/regimes require identical declared paired seeds: {group}")
        # A curve varies its named channel parameter while retaining each
        # method's frozen validation-selected optimizer settings.
        curves = defaultdict(list)
        for row in self.rows:
            if row["experiment"] != "main":
                cfg = row["config"]
                key = (row["experiment"], cfg["dataset"], row["channel_name"],
                       base_method(row["method_label"]),
                       row["training_regime"] if row["experiment"] == "m_sweep" else "varying")
                curves[key].append(row)
        for key, rows in curves.items():
            excluded = EXECUTION_SETTINGS | {"seed", "channel_seed", "eval_seed", "m"}
            if key[0] == "p_train_sweep":
                excluded = (excluded - {"m"}) | {"train_channel"}
            signatures = {json.dumps({k: v for k, v in normalize_config(row["config"]).items()
                                      if k not in excluded}, sort_keys=True) for row in rows}
            if len(signatures) != 1:
                raise ValueError(f"Sweep changes settings beyond the declared axis: {key}")


def finish(fig, output, name, data, index):
    files = []
    for extension in ("png", "pdf", "eps"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None, bbox_inches="tight", pad_inches=.06)
        files.append(str(path.resolve()))
    plt.close(fig)
    index["figures"].append(dict(name=name, files=files, data=data,
                                 uncertainty_display="none; means across all declared paired seeds"))


def style(ax):
    ax.grid(color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def value(row, metric):
    report = row["_report"]
    if metric == "accuracy":
        return row["test_accuracy"] * 100
    return report["signed_gap_decomposition_mc"]["bounded_loss"][metric]


def main_figures(rows, output, prefix, index):
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] == "main":
            groups[row["group"]].append(row)
    for group, runs in sorted(groups.items()):
        methods = defaultdict(list)
        for row in runs:
            methods[(base_method(row["method_label"]), row["training_regime"])].append(row)
        keys = [(method, regime) for method in METHODS for regime in ("matched", "perfect")]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
        data = []
        for position, (method, regime) in enumerate(keys):
            selected = methods[(method, regime)]
            acc = summary(value(row, "accuracy") for row in selected)
            gap = summary(value(row, "observed_gap") for row in selected)
            axes[0].bar(position, acc["mean"], width=.72, color=COLORS[method],
                        hatch="//" if regime == "perfect" else None)
            axes[1].plot(position, gap["mean"], marker=MARKERS[method], color=COLORS[method],
                         markerfacecolor="white" if regime == "perfect" else COLORS[method])
            data.append(dict(method=method, regime=regime, accuracy_percent=acc, observed_gap=gap,
                             task_ids=[row["task_id"] for row in selected]))
        labels = [f"{method}\n{regime}" for method, regime in keys]
        for ax in axes:
            ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
            style(ax)
        axes[0].set_ylabel("Deployment test accuracy (%)")
        axes[0].set_ylim(0, 100)
        axes[1].set_ylabel("Observed bounded-loss total gap")
        axes[1].axhline(0, color="black", linewidth=.65)
        fig.suptitle(f"{runs[0]['config']['dataset'].upper()} / {runs[0]['channel_name']}")
        fig.tight_layout()
        finish(fig, output, f"{prefix}_{slug(group)}_accuracy_gap", data, index)

        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        data = []
        for position, (method, regime) in enumerate(keys):
            selected = methods[(method, regime)]
            separate, joint = [], []
            for row in selected:
                report = row["_report"]
                terms = report["final_posterior_complexities"]
                shift = report["channel_shift_diagnostic"]["channel_shift_complexity"]
                separate.append(shift + terms["channel_complexity"] + terms["sample_complexity"])
                joint.append(shift + terms["joint_complexity"])
            first, second = summary(separate), summary(joint)
            ax.plot(position-.1, first["mean"], marker="o", color=COLORS[method])
            ax.plot(position+.1, second["mean"], marker="s", markerfacecolor="white", color=COLORS[method])
            data.append(dict(method=method, regime=regime, theorem_1_gap_diagnostic=first,
                             theorem_2_gap_diagnostic=second, task_ids=[row["task_id"] for row in selected]))
        ax.plot([], [], "o", color="black", label="T1: shift + channel + data")
        ax.plot([], [], "s", markerfacecolor="white", color="black", label="T2: shift + joint")
        ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
        ax.set_ylabel("Unweighted bounded-loss gap diagnostic")
        ax.set_title("Frozen posterior diagnostics; empirical MC correction excluded")
        ax.axhline(1, color="#777777", linestyle=":", linewidth=.8)
        ax.legend(loc="best")
        style(ax)
        fig.tight_layout()
        finish(fig, output, f"{prefix}_{slug(group)}_bound_diagnostics", data, index)


def sweep_figures(rows, output, prefix, index):
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] in {"m_sweep", "p_train_sweep"}:
            cfg = row["config"]
            key = (row["experiment"], cfg["dataset"], row["channel_name"], cfg["architecture"],
                   cfg["width"], cfg["train_size"], json.dumps(cfg["deployment_channel"], sort_keys=True))
            groups[key].append(row)
    for number, (key, runs) in enumerate(sorted(groups.items())):
        experiment, dataset, channel, *_ = key
        axis = "m" if experiment == "m_sweep" else "p_train"
        metrics = ("accuracy", "channel_generalization", "observed_gap")
        fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.2))
        data, series = [], defaultdict(list)
        for row in runs:
            series[(base_method(row["method_label"]),
                    row["training_regime"] if axis == "m" else "varying training law")].append(row)
        for (method, regime), selected in sorted(series.items()):
            grouped = defaultdict(list)
            for row in selected:
                if row["sweep_axis"] != axis or not finite(row["sweep_value"]):
                    raise ValueError("Sweep metadata does not match its declared axis")
                grouped[row["sweep_value"]].append(row)
            xs = sorted(grouped)
            if len(xs) < 2:
                raise ValueError("A trend requires at least two declared sweep values")
            label = method + (f" ({regime})" if axis == "m" else "")
            stats = {}
            for ax, metric in zip(axes, metrics):
                stats[metric] = [summary(value(row, metric) for row in grouped[x]) for x in xs]
                ax.plot(xs, [item["mean"] for item in stats[metric]], color=COLORS[method],
                        marker=MARKERS[method], markersize=4, linestyle="--" if regime == "perfect" else "-",
                        markerfacecolor="white" if regime == "perfect" else COLORS[method], label=label)
            data.append(dict(method=method, regime=regime, x=xs, statistics=stats,
                             task_ids=[[row["task_id"] for row in grouped[x]] for x in xs]))
        for ax in axes:
            ax.set_xlabel(r"Channel-bank size $m$" if axis == "m" else r"Training erasure probability $p_{\rm tr}$")
            if axis == "m":
                ax.set_xscale("log")
                xs = sorted({row["sweep_value"] for row in runs})
                ax.set_xticks(xs, [str(x) for x in xs], rotation=45, ha="right")
            else:
                matched = runs[0]["config"]["deployment_channel"]["outage_prob"]
                ax.axvline(matched, color="#777777", linestyle=":", linewidth=.8)
            style(ax)
        axes[0].set_ylabel("Deployment test accuracy (%)")
        axes[1].set_ylabel("Observed bounded-loss channel gap")
        axes[2].set_ylabel("Observed bounded-loss total gap")
        for ax in axes[1:]:
            ax.axhline(0, color="black", linewidth=.65)
        axes[0].legend(ncol=2)
        fig.suptitle(f"{dataset.upper()} / {channel}; all declared paired seeds")
        fig.tight_layout()
        finish(fig, output, f"{prefix}_{dataset}_{channel}_{axis}_sweep_{number}", data, index)


def export(sources, output):
    # Validate all sources before publishing any figure.
    evidence = [Evidence(root, manifest) for root, manifest in sources]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = dict(figures=[], sources=[], policy="Saved JSON only; all declared seeds; no test or seed selection; sources not pooled",
                 bound_scope="Unweighted bounded-loss gap diagnostics, not population-risk certificates; no empirical MC correction")
    table = []
    for source in evidence:
        prefix = slug(source.root.name) + "_" + hashlib.sha256(str(source.root.resolve()).encode()).hexdigest()[:6]
        main_figures(source.rows, output, prefix, index)
        sweep_figures(source.rows, output, prefix, index)
        index["sources"].append(dict(root=str(source.root.resolve()), manifest=str(source.manifest.resolve()),
                                      declared_tasks=len(source.rows), excluded_task_ids=source.excluded_task_ids,
                                      source_sha256=source.files))
        for row in source.rows:
            report = row["_report"]
            terms = report["final_posterior_complexities"]
            shift = report["channel_shift_diagnostic"]["channel_shift_complexity"]
            table.append(dict(source=str(source.root.resolve()), task_id=row["task_id"], group=row["group"],
                              experiment=row["experiment"], method=row["method_label"], regime=row["training_regime"],
                              seed=row["seed"], channel_seed=row["config"]["channel_seed"],
                              sweep_axis=row["sweep_axis"], sweep_value=row["sweep_value"],
                              test_accuracy=row["test_accuracy"], bounded_test_risk=report["heldout_gibbs_mc"]["bounded_loss"],
                              observed_channel_gap=value(row, "channel_generalization"), observed_total_gap=value(row, "observed_gap"),
                              kl=terms["kl"], theorem_1_gap_diagnostic=shift+terms["channel_complexity"]+terms["sample_complexity"],
                              theorem_2_gap_diagnostic=shift+terms["joint_complexity"]))
    (output / "figure_index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    with (output / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("ROOT", "MANIFEST"),
                        help="Repeat for confirmation and sweep sources; distinct sources are never pooled")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = export(args.source, args.output)
    print(json.dumps(dict(figures=len(index["figures"]), sources=len(index["sources"]), output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
