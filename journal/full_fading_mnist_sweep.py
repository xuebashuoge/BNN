"""Train and plot the matched full-fading MNIST finite-channel experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean

from .build_bayesian_campaign import base_config, entry
from .campaign import evaluate_trial, save_json, train_trial
from .run_task import read_manifest, verify_submission_snapshot


BANK_SIZES = (2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 200, 400, 800, 1600, 3200)
EXTENDED_BANK_SIZES = (6400, 12800, 25600)
SEEDS = (74101, 74102, 74103)
# Match the two optimizer settings used for the original MNIST bank sweep.
METHODS = {
    "T1": dict(method="proposed", lr=.01, eta_channel=.001,
               eta_data=.05380226956668572, rho_lr_scale=.1),
    "T2": dict(method="joint", lr=.03, eta_channel=0.,
               eta_data=.0019996546286550664, rho_lr_scale=0.),
}
LAW = dict(kind="full_fading", rx_correlation=.25, tx_correlation=.25, noise_std=1.)


def build_manifest(path, *, pilot=False, extended=False, law=LAW):
    sizes = (BANK_SIZES[-1],) if pilot else EXTENDED_BANK_SIZES if extended else BANK_SIZES
    seeds = SEEDS[:1] if pilot else SEEDS
    methods = ("T1",) if pilot else METHODS
    channel = law["kind"]
    stage = f"{channel}_pilot" if pilot else f"{channel}_m_sweep"
    base = base_config("mnist", "analog")
    base.update(train_channel=law, deployment_channel=law, train_size=10000, val_size=5000,
                epochs=2 if pilot else 30, eval_draws=2 if pilot else 10,
                val_draws=2 if pilot else 5)
    rows = []
    for m in sizes:
        for method in methods:
            for seed in seeds:
                config = {**base, **METHODS[method], "m": m}
                item = entry(config, seed, "mnist", channel, method, stage, "matched",
                             "m_sweep", "m", m, f"mnist_{channel}_m{m}")
                item.update(evaluation_draws=base["eval_draws"], scientific_result=not pilot)
                rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(len(rows))


def run_task(manifest, output, index, device, threads):
    verify_submission_snapshot(manifest)
    entries = read_manifest(manifest)
    if not 0 <= index < len(entries):
        raise ValueError(f"Task index {index} outside manifest of length {len(entries)}")
    item = entries[index]
    row = train_trial(item, output / "runs", threads=threads, resume=True, device=device)
    if row["status"] == "complete":
        row = evaluate_trial(row, item["evaluation_draws"], full_risks=True, resume=True)
    save_json(output / "task_reports" / f'{item["task_id"]}.json', row)
    print(json.dumps(row, allow_nan=False), flush=True)
    if row["status"] != "complete" or row.get("test_status") != "complete":
        raise SystemExit(1)


def read_observations(manifests, outputs, *, law=LAW):
    expected, observations = set(), []
    for manifest, output in zip(manifests, outputs, strict=True):
        for item in read_manifest(manifest):
            if not item["scientific_result"]:
                raise ValueError("Resource pilot trials cannot enter the MNIST figure")
            if any(item["config"][name] != law for name in ("train_channel", "deployment_channel")):
                raise ValueError("Expected matched training and deployment under the requested channel law")
            key = (item["method_label"], item["sweep_value"], item["config"]["seed"])
            if key in expected:
                raise ValueError(f"Duplicate scientific trial {key}")
            expected.add(key)
            row = json.loads((output / "task_reports" / f'{item["task_id"]}.json').read_text())
            if row["task_id"] != item["task_id"] or row.get("test_status") != "complete":
                raise ValueError(f"Incomplete scientific trial {key}")
            config = row["config"]
            if any(config[name] != value for name, value in item["config"].items()
                   if name not in {"device", "threads"}):
                raise ValueError(f"Training report has different settings for {key}")
            report_path = output / "runs" / item["task_id"] / "test_full" / "evaluation.json"
            report = json.loads(report_path.read_text())
            evaluation = report["evaluation_config"]
            expected_eval = {**config, "eval_draws": item["evaluation_draws"],
                             "eval_seed": config["eval_seed"] + 1000003}
            if report["split"] != "test" or report["risk_scope"] != "full" or evaluation != expected_eval:
                raise ValueError(f"Evaluation report has different settings for {key}")
            accuracy = float(report["heldout_gibbs_mc"]["accuracy"])
            gap = float(report["bounded_gap_mc"])
            if not (0 <= accuracy <= 1 and math.isfinite(gap) and abs(gap) <= 1):
                raise ValueError(f"Invalid metrics for {key}")
            observations.append(dict(method=key[0], m=key[1], seed=key[2], test_accuracy=accuracy,
                                     generalization_error=gap, seconds=row["seconds"]))
    sizes = tuple(sorted({row["m"] for row in observations}))
    if sizes not in (BANK_SIZES, BANK_SIZES + EXTENDED_BANK_SIZES):
        raise ValueError("Expected the reference bank grid, optionally with the entire extension")
    if expected != {(method, m, seed) for method in METHODS for m in sizes for seed in SEEDS}:
        raise ValueError("Expected both methods and all preregistered seeds at each bank size")
    return sizes, observations


def plot(sizes, observations, output, *, law=LAW):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    scores = defaultdict(list)
    for row in observations:
        scores[row["method"], row["m"]].append(row)
    means = [
        dict(method=method, m=m, n_seeds=len(scores[method, m]),
             test_accuracy=fmean(row["test_accuracy"] for row in scores[method, m]),
             generalization_error=fmean(row["generalization_error"] for row in scores[method, m]))
        for method in METHODS for m in sizes
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("observations", observations), ("means", means)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.linewidth": .65, "mathtext.fontset": "dejavuserif"})
    fig, (accuracy, gap) = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True, layout="constrained")
    colors = ("#1F5B91", "#BC540F")
    for method, color, mark, line in zip(METHODS, colors, ("o", "D"), ("-", "--"), strict=True):
        series = [row for row in means if row["method"] == method]
        for axis, field, scale in ((accuracy, "test_accuracy", 100), (gap, "generalization_error", 1)):
            axis.plot(sizes, [scale * row[field] for row in series], label=method, color=color,
                      marker=mark, linestyle=line, linewidth=1.2, markersize=3.4,
                      markerfacecolor="white", markeredgewidth=.8)

    for axis in (accuracy, gap):
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCE0E4", linewidth=.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.8, width=.65, labelsize=7)
    accuracy.set_ylabel("Test accuracy (%)")
    accuracy.legend(ncols=2, frameon=False)
    gap.axhline(0, color="#A0A9B3", linewidth=.55, zorder=0)
    gap.set_ylabel(r"Error $\Delta(U,S)$")
    gap.set_xlabel(r"Training channel-set size $m$")
    gap.set_xscale("log")
    ticks = [2, 5, 20, 50, 200, 800, 3200] if sizes == BANK_SIZES else [2, 5, 20, 100, 800, 3200, 25600]
    gap.xaxis.set_major_locator(FixedLocator(ticks))
    gap.xaxis.set_major_formatter(ScalarFormatter())
    gap.xaxis.set_minor_locator(NullLocator())
    fig.align_ylabels()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"mnist_{law['kind']}_finite_channel_generalization.{suffix}", dpi=450)
    plt.close(fig)

    # Use the last three grid points to describe saturation in percentage points.
    tail = {}
    for method in METHODS:
        rows = [row for row in means if row["method"] == method][-3:]
        values = [100 * row["test_accuracy"] for row in rows]
        tail[method] = dict(bank_sizes=[row["m"] for row in rows], accuracy_percent=values,
                            range_percentage_points=max(values) - min(values))
    save_json(output / "tail_summary.json", tail)


def main(law=LAW):
    parser = argparse.ArgumentParser(description=f"Matched MNIST finite-channel sweep for {law['kind']}")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    options = build.add_mutually_exclusive_group()
    options.add_argument("--pilot", action="store_true")
    options.add_argument("--extended", action="store_true")
    task = commands.add_parser("task")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--output-root", type=Path, required=True)
    task.add_argument("--index", type=int)
    task.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    task.add_argument("--threads", type=int, default=2)
    report = commands.add_parser("report")
    report.add_argument("--manifest", type=Path, action="append", required=True)
    report.add_argument("--stage-root", type=Path, action="append", required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_manifest(args.output, pilot=args.pilot, extended=args.extended, law=law)
    elif args.command == "task":
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_task(args.manifest, args.output_root, index, args.device, args.threads)
    else:
        sizes, rows = read_observations(args.manifest, args.stage_root, law=law)
        plot(sizes, rows, args.output, law=law)


if __name__ == "__main__":
    main()
