#!/usr/bin/env python3
"""Audit frozen experiment metadata without importing or running training code.

Run with the torch28 environment. Seed intervals describe training-seed variation
conditional on the saved dataset, channel-bank protocol, and evaluated test set.
They are descriptive, pointwise intervals, not post-selection guarantees.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SHARED = (
    "dataset", "hidden_dim", "batch_size", "n_samples", "epochs",
    "m_artificial_channels", "lr", "lr_decay_step", "lr_decay_gamma", "mode",
    "mu_m_te", "std_m_te", "mu_b_te", "std_b_te", "mu_m_tr", "std_m_tr",
    "mu_b_tr", "std_b_tr", "p_outage_te", "p_outage_tr", "moon_noise",
)
SPECIFIC = ("objective", "n_u_sets", "mi_mc_samples", "alpha_coeff", "beta_coeff", "gamma_coeff")
METRICS = ("population_accuracy", "population_loss", "empirical_accuracy", "empirical_loss", "bound")
FIXED_MNIST = (
    "test17_ideal", "test17_low_SNR", "test17_severe_fading", "test17_severe_fading_low_SNR",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def describe(values):
    """Student t interval; no interval is manufactured for a single seed."""
    from scipy.stats import t
    vals = [float(x) for x in values if finite(x)]
    n = len(vals)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else None
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def normalize(record, source, line):
    record = dict(record)
    aliases = {"acc": "population_accuracy", "loss": "population_loss",
               "test_acc": "population_accuracy", "test_loss": "population_loss",
               "beta": "beta_coeff", "gamma": "gamma_coeff", "alpha": "alpha_coeff"}
    for old, new in aliases.items():
        if new not in record and old in record:
            record[new] = record[old]
    record["source"] = source
    record["source_line"] = line
    record["method"] = record.get("scenario", record.get("scenario_name", "unknown"))
    if "dataset" not in record:
        record["dataset"] = "MNIST (source inferred)" if source.startswith("test17") else "two_moons (source inferred)"
    return record


def config(record, specific=True):
    fields = SHARED + (SPECIFIC if specific else ())
    return {field: record.get(field) for field in fields}


def read_snapshot(root):
    runs, inventory, errors, duplicates = [], [], [], []
    cross_source = defaultdict(list)
    for path in sorted(root.rglob("*.json*")):
        if path.suffix not in (".json", ".jsonl"):
            continue
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        item = {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        inventory.append(item)
        try:
            if path.suffix == ".jsonl":
                parsed = []
                for line, text in enumerate(raw.decode("utf-8").splitlines(), 1):
                    if not text.strip():
                        continue
                    try:
                        parsed.append((line, json.loads(text)))
                    except (ValueError, TypeError) as exc:
                        errors.append({"path": relative, "line": line, "error": str(exc)})
                item["records"] = len(parsed)
                # Comparisons/aggregate summaries are derived data: never count
                # those records as additional independent experimental runs.
                if path.name != "sweep_results.jsonl":
                    continue
            else:
                payload = json.loads(raw)
                item["shape"] = type(payload).__name__
                if path.name != "sweep_results.json" or not isinstance(payload, list):
                    continue
                parsed = list(enumerate(payload, 1))
            seen = {}
            for line, record in parsed:
                if not isinstance(record, dict):
                    errors.append({"path": relative, "line": line, "error": "run is not an object"})
                    continue
                try:
                    fingerprint = digest(record)
                except ValueError:
                    errors.append({"path": relative, "line": line, "error": "nonfinite JSON run"})
                    continue
                cross_source[fingerprint].append({"path": relative, "line": line})
                if fingerprint in seen:
                    duplicates.append({"path": relative, "line": line, "first_line": seen[fingerprint], "kind": "exact"})
                    continue
                seen[fingerprint] = line
                normalized = normalize(record, path.parent.relative_to(root).as_posix(), line)
                if not finite(normalized.get("population_accuracy")):
                    errors.append({"path": relative, "line": line, "error": "missing/nonfinite population accuracy"})
                    continue
                normalized["raw_sha256_prefix"] = fingerprint
                runs.append(normalized)
        except (ValueError, UnicodeError, OSError) as exc:
            errors.append({"path": relative, "error": str(exc)})
    archive_overlaps = [locations for locations in cross_source.values()
                        if len({x["path"] for x in locations}) > 1]
    return runs, inventory, errors, duplicates, archive_overlaps


def aggregate(runs):
    """Keep source folders separate so archived sweeps cannot inflate seed n."""
    groups = defaultdict(list)
    for run in runs:
        groups[(run["source"], run["method"], canonical(config(run)))].append(run)
    summaries, conflicts = [], []
    for (source, method, serialized), group in sorted(groups.items()):
        by_seed = defaultdict(list)
        for run in group:
            by_seed[run.get("seed")].append(run)
        clean = []
        for seed, records in by_seed.items():
            if seed is None or len(records) != 1:
                conflicts.append({"source": source, "method": method, "seed": seed,
                                  "config": json.loads(serialized), "lines": [r["source_line"] for r in records]})
            else:
                clean.append(records[0])
        row = {"source": source, "method": method, "config_id": digest(json.loads(serialized)),
               "config": json.loads(serialized), "num_seeds": len(clean),
               "seeds": sorted(r["seed"] for r in clean),
               "selection_status": "exploratory legacy results; test-based tuning/selection not excluded"}
        for metric in METRICS:
            row[metric] = describe(r.get(metric) for r in clean)
        summaries.append(row)
    return summaries, conflicts


def paired_analysis(runs):
    baselines = defaultdict(list)
    for run in runs:
        if run["method"] == "erm":
            key = (run["source"], run.get("seed"), canonical(config(run, False)))
            baselines[key].append(run)
    groups = defaultdict(list)
    unmatched, paired_rows = [], []
    for run in runs:
        if run["method"] != "proposed":
            continue
        key = (run["source"], run.get("seed"), canonical(config(run, False)))
        matches = baselines.get(key, [])
        if len(matches) != 1 or run.get("seed") is None:
            unmatched.append({"source": run["source"], "line": run["source_line"],
                              "seed": run.get("seed"), "erm_matches": len(matches)})
            continue
        baseline = matches[0]
        row = {"source": run["source"], "seed": run["seed"], "config_id": digest(config(run)),
               "config": config(run), "proposed_line": run["source_line"], "erm_line": baseline["source_line"]}
        for metric in METRICS:
            p, e = run.get(metric), baseline.get(metric)
            row["proposed_" + metric] = p
            row["erm_" + metric] = e
            row["difference_" + metric] = p - e if finite(p) and finite(e) else None
        groups[(run["source"], canonical(config(run)))].append(row)
    summaries = []
    for (source, serialized), group in sorted(groups.items()):
        seed_counts = Counter(r["seed"] for r in group)
        clean = [row for row in group if seed_counts[row["seed"]] == 1]
        for row in group:
            if seed_counts[row["seed"]] != 1:
                unmatched.append({"source": source, "line": row["proposed_line"], "seed": row["seed"],
                                  "reason": "multiple proposed records for same config/seed; excluded"})
        paired_rows.extend(clean)
        item = {"source": source, "config_id": digest(json.loads(serialized)), "config": json.loads(serialized),
                "num_paired_seeds": len(clean), "seeds": sorted(r["seed"] for r in clean),
                "interval_method": "pointwise paired Student t, 95%; no multiplicity/selection adjustment",
                "selection_status": "exploratory; fixed config across all available paired seeds; no seed selection"}
        for metric in METRICS:
            item["proposed_" + metric] = describe(r["proposed_" + metric] for r in clean)
            item["erm_" + metric] = describe(r["erm_" + metric] for r in clean)
            item["difference_" + metric] = describe(r["difference_" + metric] for r in clean)
        item["accuracy_win_fraction"] = (sum(r["difference_population_accuracy"] > 0 for r in clean) / len(clean)) if clean else None
        summaries.append(item)
    return summaries, paired_rows, unmatched


def flatten_summary(row):
    result = {}
    for key, value in row.items():
        if isinstance(value, dict) and "mean" in value and "n" in value:
            result.update({key + "_" + subkey: subvalue for subkey, subvalue in value.items()})
        else:
            result[key] = value
    return result


def save_plot(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")


def plot_mnist(summaries, pairs, output):
    import matplotlib.pyplot as plt
    selected = []
    for source in FIXED_MNIST:
        candidates = [r for r in summaries if r["source"] == source and r["num_paired_seeds"]]
        # Never choose the most favorable parameter setting for a headline plot.
        if len(candidates) == 1:
            selected.append(candidates[0])
    if not selected:
        return []
    labels = {"test17_ideal": "Near ideal", "test17_low_SNR": "Low SNR",
              "test17_severe_fading": "Severe fading", "test17_severe_fading_low_SNR": "Fading + low SNR"}
    fig, (ax, diff) = plt.subplots(1, 2, figsize=(10.3, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = ("#527a9b", "#cf804b")
    for idx, row in enumerate(selected):
        for offset, method, color in zip((-0.17, 0.17), ("erm", "proposed"), colors):
            stat = row[method + "_population_accuracy"]
            half = (stat["ci95_high"] - stat["mean"]) * 100 if stat["ci95_high"] is not None else 0
            ax.bar(idx + offset, stat["mean"] * 100, width=0.31, color=color,
                   yerr=half, capsize=3, label=("ERM" if method == "erm" else "Proposed") if idx == 0 else None)
        stat = row["difference_population_accuracy"]
        points = [p["difference_population_accuracy"] * 100 for p in pairs
                  if p["source"] == row["source"] and p["config_id"] == row["config_id"]]
        jitter = [idx + 0.11 * math.sin(i * 2.39996) for i in range(len(points))]
        diff.scatter(jitter, points, s=10, alpha=0.25, color="#777777", zorder=1)
        half = (stat["ci95_high"] - stat["mean"]) * 100 if stat["ci95_high"] is not None else 0
        diff.errorbar(idx, stat["mean"] * 100, yerr=half, fmt="o", capsize=5,
                      color="#222222", markersize=5, zorder=3)
    for axis in (ax, diff):
        axis.set_xticks(range(len(selected)), [labels[r["source"]] for r in selected], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    ax.set_ylabel("Deployment accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    diff.axhline(0, color="#999999", lw=1)
    diff.set_ylabel("Proposed − ERM (percentage points)")
    fig.suptitle("Frozen MNIST results: every seed, one saved parameter setting per channel", fontsize=11)
    fig.text(0.5, -0.02, "Exploratory legacy results • bars/black dots: means with 95% seed t intervals • gray dots: all paired seeds",
             ha="center", fontsize=8)
    fig.tight_layout()
    save_plot(fig, output, "mnist_fixed_parameters")
    plt.close(fig)
    return [{"source": r["source"], "config_id": r["config_id"], "config": r["config"]} for r in selected]


def mismatch_curves(root, output):
    import matplotlib.pyplot as plt
    curves, errors = [], []
    for path in sorted(root.glob("test14*/sweep_summary.json")):
        try:
            payload = json.loads(path.read_text())
            if "sweep_data" not in payload:
                continue
            for record in payload["sweep_data"]:
                curves.append({"source": path.parent.name, "p_te_fixed": payload.get("p_te_fixed"), **record})
        except (ValueError, TypeError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
        for source in sorted({r["source"] for r in curves}):
            group = sorted((r for r in curves if r["source"] == source), key=lambda r: r["p_tr"])
            for axis, metric in zip(axes, ("test_acc", "test_loss")):
                scale = 100 if metric == "test_acc" else 1
                axis.plot([r["p_tr"] for r in group], [r[metric] * scale for r in group], lw=1.5, label=source.replace("test14_", ""))
                if finite(group[0].get("p_te_fixed")):
                    axis.axvline(group[0]["p_te_fixed"], color="#888888", ls=":", lw=0.8)
        for axis in axes:
            axis.set_xlabel("Training outage probability")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Deployment accuracy (%)")
        axes[1].set_ylabel("Deployment cross-entropy")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Fixed deployment channel, varying training channel", fontsize=11)
        fig.text(0.5, -0.015, "Exploratory legacy Theorem 3 prototype • all saved curves • no seed or test-peak selection • dotted: matched distribution",
                 ha="center", fontsize=8)
        fig.tight_layout()
        save_plot(fig, output, "legacy_mismatch")
        plt.close(fig)
    return curves, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/journal_legacy_snapshot"))
    parser.add_argument("--output", type=Path, default=Path("results/journal_audit"))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"snapshot root does not exist: {root}")
    if output == root or root in output.parents:
        parser.error("output must be outside the immutable snapshot root")
    output.mkdir(parents=True, exist_ok=True)
    # Use a headless backend and a writable cache without importing training code.
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    runs, inventory, errors, duplicates, overlaps = read_snapshot(root)
    summaries, conflicts = aggregate(runs)
    paired, pair_rows, unmatched = paired_analysis(runs)
    plot_configs = plot_mnist(paired, pair_rows, output)
    curves, curve_errors = mismatch_curves(root, output)
    errors.extend(curve_errors)
    warnings = [
        "No training modules are imported, no checkpoints are executed, and no training is performed.",
        "Saved test metrics are exploratory: hyperparameters/earlier scenario seeds may have been test-selected.",
        "No best seed or best parameter setting is selected. Every available seed is retained within each fixed configuration.",
        "Student t intervals measure seed variation conditional on this test set and channel-bank protocol; they do not certify population accuracy.",
        "Grid intervals are pointwise and unadjusted for multiple comparisons or parameter selection.",
        "Source folders are analyzed separately; overlapping archived runs are never pooled to increase sample size.",
        "Missing channel metadata remains unknown rather than inferred from current script globals.",
        "Legacy bound values and posterior-mean empirical risks are metadata only; they are not recomputed or treated as valid certificates.",
        "A JSON-only snapshot does not establish availability or identity of its original weight checkpoints.",
    ]
    report = {"root": str(root), "num_run_records": len(runs), "num_source_files": len(inventory),
              "run_counts_by_source": dict(sorted(Counter(r["source"] for r in runs).items())),
              "warnings": warnings, "method_config_summaries": summaries,
              "paired_config_summaries": paired, "mnist_figure_configs": plot_configs,
              "mismatch_curve_sources": sorted({r["source"] for r in curves}),
              "num_exact_within_file_duplicates_excluded": len(duplicates),
              "num_cross_file_duplicate_fingerprints": len(overlaps),
              "num_ambiguous_method_config_seeds": len(conflicts), "num_unmatched_proposed_records": len(unmatched),
              "num_parse_or_metric_errors": len(errors),
              "local_checkpoint_files": [str(p.relative_to(root)) for p in sorted(root.rglob("*.pth"))]}
    write_json(output / "audit.json", report)
    write_json(output / "source_inventory.json", inventory)
    write_json(output / "diagnostics.json", {"parse_or_metric_errors": errors, "within_file_duplicates": duplicates,
               "cross_file_exact_overlaps": overlaps, "ambiguous_method_config_seeds": conflicts,
               "unmatched_proposed_records": unmatched})
    write_csv(output / "method_config_summary.csv", [flatten_summary(r) for r in summaries])
    write_csv(output / "paired_config_summary.csv", [flatten_summary(r) for r in paired])
    write_csv(output / "paired_seed_results.csv", pair_rows)
    write_csv(output / "legacy_mismatch.csv", curves)
    with (output / "normalized_runs.jsonl").open("w", encoding="utf-8") as stream:
        for run in runs:
            stream.write(canonical(run) + "\n")
    print(json.dumps({"output": str(output), "runs": len(runs), "paired_configurations": len(paired),
                      "mnist_fixed_configurations": len(plot_configs), "mismatch_curves": len(report["mismatch_curve_sources"]),
                      "unmatched_proposed": len(unmatched), "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
