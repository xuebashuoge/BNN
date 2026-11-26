"""Deployment-accuracy table at a smaller shared bank size, T1 with the fresh Q_{W|S}.

Goal: the better of T1 and T2 beats the better of ERM and L2 in all eight rows
(four datasets x fading/erasure), by as wide a margin as possible, with one
channel-bank size m shared by every row and method.

Why a smaller m helps T1. The `fresh` stage-1 prior mu_S is trained on S with a
new P_tr draw for every minibatch and never sees the bank U, so it does not
depend on m. Stage 2 is T1 on the bank U of size m with Q_{W|S} = N(mu_S, s^2 I);
its channel term scales like 1/sqrt(m-1), so a small bank pulls T1 towards mu_S.
ERM and L2 only see the m channels in U and overfit them as m shrinks.

Protocol
1. ``screen``: on two tuning seed offsets (disjoint from the confirmation seeds)
   and every m in M_GRID, fit every row with
     * ERM (archived configuration, only m changed);
     * L2  with l2 x {1, 10, 100};
     * T2  with eta_data x {1, 10, 100};
     * T1  with the fresh prior, s x {3, 10, 30, 100} posterior_std and
       eta_channel x {1, 10, 100}.
   All are scored on deployment accuracy on validation.
2. ``select``: for each m pick each method's best candidate per row, then the m
   whose smallest row gap max(T1, T2) - max(ERM, L2) is largest (ties: mean gap).
3. ``confirm``: refit the selected candidates at that m on the archived paired
   training seeds with the archived test-evaluation seeds.

The fresh stage-1 fits keep the archived m = 2000 in their configs (m is unused
by fresh training, its sample term and its deployment-channel validation), so
their task IDs match the data-dependent campaign and those checkpoints can be
linked in with ``--reuse-stage1``. The perfect-training column is independent of
m and is copied from the archived table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import tarfile

from . import data_dependent as dd

PUBLISH = dd.SELECTED / "higher_accuracy"
PERFORMANCE_BASE = dd.PERFORMANCE_BASE
M_GRID = (20, 100, 500)
ARCHIVED_M = 2000
METHODS = ("ERM-Matched", "L2-Matched", "T2-Matched", "T1-Matched")
BASELINES = ("ERM-Matched", "L2-Matched")
PROPOSED = ("T1-Matched", "T2-Matched")
L2_MULTIPLIERS = (1., 10., 100.)
T2_MULTIPLIERS = (1., 10., 100.)
T1_GRID = [("fresh", s, e) for s in dd.STD_MULTIPLIERS for e in dd.ETA_MULTIPLIERS]
PRIOR_COUNT = len(dd.STD_MULTIPLIERS)  # union bound over the fresh scales only
# Longest fits first so the pool does not end on a CIFAR-10 tail.
DATASET_ORDER = ("cifar10", "mnist", "fashionmnist", "moons")


# --------------------------------------------------------------------------- archived sources

def archived_sources():
    """{(dataset, channel, method): [dict(config, evaluation, baseline_task, seed)]} at m=2000."""
    rows = [row for row in dd.performance_baselines() if row["method"] in METHODS]
    manifests = {}
    for dataset in dd.DATASETS:
        for phase in ("primary", "refined"):
            for entry in dd.read_manifest(PERFORMANCE_BASE / "provenance" / dataset / phase / "confirmation.jsonl"):
                manifests[entry["task_id"]] = entry
    sources = {}
    with tarfile.open(dd.SELECTED / "data" / "performance" / "selected_trials.tar.gz") as archive:
        for row in rows:
            if row["channel"] == "Fading":
                entry = manifests[row["task_id"]]
                config, evaluation = dict(entry["config"]), dict(entry["evaluation"])
            else:
                key = row["archive_key"]
                config = json.load(archive.extractfile(key + "/task_config.json"))
                evaluation = json.load(archive.extractfile(key + "/test_deployment/overrides.json"))
            if (int(config["seed"]) != int(row["seed"]) or config["dataset"] != row["dataset"]
                    or config["m"] != ARCHIVED_M):
                raise ValueError(f"Archived source mismatch for {row['task_id']}")
            if row["channel"] == "Fading" and not config.get("transmit_unit_power"):
                raise ValueError("Fading rows must keep unit transmit power")
            sources.setdefault((row["dataset"], row["channel"], row["method"]), []).append(
                dict(config=config, evaluation=evaluation, baseline_task=row["task_id"], seed=int(row["seed"])))
    for group in sources.values():
        group.sort(key=lambda source: source["seed"])
    return sources


# --------------------------------------------------------------------------- task construction

def candidates(method: str):
    if method == "ERM-Matched":
        return [("erm",)]
    if method == "L2-Matched":
        return [("l2", k) for k in L2_MULTIPLIERS]
    if method == "T2-Matched":
        return [("t2", k) for k in T2_MULTIPLIERS]
    return T1_GRID


def label(method: str, candidate) -> str:
    if method == "T1-Matched":
        return dd.variant_label(candidate)
    return candidate[0] if len(candidate) == 1 else f"{candidate[0]}-x{candidate[1]:g}"


def method_tasks(method: str, candidate, archived: dict, m: int, root: Path, evals, meta: dict):
    """Tasks for one fit at bank size m; archived is the m=2000 config with the run's seeds."""
    meta = dict(meta, method=method, candidate=list(candidate), label=label(method, candidate), m=m)
    if method == "T1-Matched":
        stage1 = dd.new_run_config(dd.stage1_config(archived, "fresh"))
        s1_id = f"performance-stage1-fresh-{dd.digest(stage1)[:12]}"
        config = dd.new_run_config(dd.stage2_config({**archived, "m": m}, dd.run_dir(root, s1_id),
                                                    candidate, PRIOR_COUNT))
        fit_id = f"ha-m{m}-T1-{label(method, candidate)}-{dd.digest(config)[:12]}"
        return [dict(task_id=s1_id, config=stage1, deps=[], evals=[], meta=dict(role="stage1", construction="fresh")),
                dict(task_id=fit_id, config=config, deps=[s1_id], evals=evals, meta=dict(meta, role="stage2"))]
    config = {**archived, "m": m}
    if method == "L2-Matched":
        config["l2"] = archived["l2"] * candidate[1]
    elif method == "T2-Matched":
        config["eta_data"] = archived["eta_data"] * candidate[1]
    config = dd.new_run_config(config)
    fit_id = f"ha-m{m}-{method.split('-')[0]}-{label(method, candidate)}-{dd.digest(config)[:12]}"
    return [dict(task_id=fit_id, config=config, deps=[], evals=evals, meta=dict(meta, role="stage2"))]


def ordered(tasks):
    tasks = dd.dedupe(tasks)
    return sorted(tasks, key=lambda task: DATASET_ORDER.index(task["config"]["dataset"]))


def plan_screen(root: Path, m_grid=M_GRID):
    sources = archived_sources()
    tasks = []
    for (dataset, channel, method), seeds in sorted(sources.items()):
        for offset in dd.TUNING_OFFSETS:
            archived = dd.with_seeds(seeds[0]["config"], offset)
            evals = [dd.validation_eval(dd.new_run_config(archived))]
            for m in m_grid:
                for candidate in candidates(method):
                    tasks.extend(method_tasks(method, candidate, archived, m, root, evals,
                                              dict(dataset=dataset, channel=channel, seed=archived["seed"],
                                                   phase="screen")))
    return ordered(tasks)


def plan_confirm(root: Path):
    selection = dd.read_json(root / "selection.json")
    m = selection["m"]
    tasks = []
    for (dataset, channel, method), seeds in sorted(archived_sources().items()):
        candidate = selection["rows"][f"{dataset}/{channel}"]["methods"][method]["candidate"]
        for source in seeds:
            test = dict(name="test", split="test", scope="deployment",
                        overrides={**source["evaluation"], "device": "cpu", "threads": 1})
            tasks.extend(method_tasks(method, tuple(candidate), source["config"], m, root, [test],
                                      dict(dataset=dataset, channel=channel, seed=source["seed"], phase="confirm",
                                           baseline_task=source["baseline_task"])))
    return ordered(tasks)


# --------------------------------------------------------------------------- selection

def screen_scores(root: Path):
    """{(m, row, method): {label: (candidate, [validation accuracy per seed])}}."""
    tasks = dd.read_manifest(root / "manifests" / "screen.jsonl")
    scores = {}
    for record in dd.done_records(root, tasks):
        meta = record["meta"]
        group = (meta["m"], f"{meta['dataset']}/{meta['channel']}", meta["method"])
        entry = scores.setdefault(group, {}).setdefault(meta["label"], (meta["candidate"], []))
        entry[1].append(record["evaluations"]["validation"]["accuracy"])
    return scores


def select(root: Path):
    scores = screen_scores(root)
    per_m = {}
    for (m, row, method), by_label in scores.items():
        counts = {len(values) for _, values in by_label.values()}
        if counts != {len(dd.TUNING_OFFSETS)}:
            raise ValueError(f"Incomplete tuning seeds for {m} {row} {method}: {counts}")
        ranked = sorted(by_label.items(), key=lambda item: -statistics.fmean(item[1][1]))
        best_label, (candidate, values) = ranked[0]
        per_m.setdefault(m, {}).setdefault(row, {})[method] = dict(
            candidate=candidate, label=best_label, validation_accuracy=statistics.fmean(values),
            leaderboard=[dict(label=k, mean=statistics.fmean(v), seeds=v) for k, (_, v) in ranked[:6]])
    summary = []
    for m, rows in sorted(per_m.items()):
        gaps = {}
        for row, methods in rows.items():
            if set(methods) != set(METHODS):
                raise ValueError(f"Missing methods at m={m} for {row}")
            proposed = max(methods[k]["validation_accuracy"] for k in PROPOSED)
            baseline = max(methods[k]["validation_accuracy"] for k in BASELINES)
            gaps[row] = 100 * (proposed - baseline)
        summary.append(dict(m=m, min_gap_pp=min(gaps.values()), mean_gap_pp=statistics.fmean(gaps.values()),
                            gaps_pp=gaps))
    best = max(summary, key=lambda item: (item["min_gap_pp"], item["mean_gap_pp"]))
    selection = dict(m=best["m"], rule="largest smallest-row validation gap max(T1,T2) - max(ERM,L2); ties: mean",
                     screen=summary, rows=per_m[best["m"]], all_m=per_m)
    dd.write_json(root / "selection.json", selection)
    print(json.dumps(summary, indent=1))
    return selection


# --------------------------------------------------------------------------- summary

def summarize(root: Path):
    selection = dd.read_json(root / "selection.json")
    m = selection["m"]
    tasks = [task for task in dd.read_manifest(root / "manifests" / "confirm.jsonl")
             if task["meta"]["role"] == "stage2"]
    records = {record["task_id"]: record for record in dd.done_records(root, tasks)}
    rows = []
    for task in tasks:
        meta, record = task["meta"], records[task["task_id"]]
        report = dd.read_json(dd.run_dir(root, task["task_id"]) / "test" / "evaluation.json")
        heldout = report["heldout_gibbs_mc"]
        if report["evaluation_config"]["deployment_channel"]["kind"] == "diagonal_fading":
            if not all(abs(r["feature_power_per_coordinate"] - 1.) <= 2e-5 for r in heldout["replicate_statistics"]):
                raise ValueError(f"Unit transmit power violated in {task['task_id']}")
        complexities = report.get("final_posterior_complexities", {})
        rows.append(dict(dataset=meta["dataset"], channel=meta["channel"], method=meta["method"],
                         seed=str(meta["seed"]), m=str(m), test_accuracy_pct=100 * heldout["accuracy"],
                         draws=heldout["draws"], examples=heldout["examples"], task_id=task["task_id"],
                         training_revision=dd.read_json(dd.run_dir(root, task["task_id"]) / "metadata.json")["git_sha"],
                         archive_key=f"higher_accuracy/{task['task_id']}", weights_sha256=record["weights_sha256"],
                         variant=meta["label"], channel_kl=complexities.get("channel_kl", ""),
                         channel_complexity=complexities.get("channel_complexity", "")))
    perfect = [dict(row, variant="archived (m-independent)", channel_kl="", channel_complexity="")
               for row in dd.performance_baselines() if row["method"] == "Proposed-Perfect"]
    rows = rows + perfect
    rows.sort(key=lambda r: (list(dd.DATASETS).index(r["dataset"]), dd.CHANNELS.index(r["channel"]),
                             r["method"], int(r["seed"])))
    m2000 = dd.read_json(dd.PUBLISH / "only_fresh" / "performance" / "comparison.json")
    comparison = []
    for dataset in dd.DATASETS:
        for channel in dd.CHANNELS:
            means = {method: statistics.fmean(float(r["test_accuracy_pct"]) for r in rows
                                              if (r["dataset"], r["channel"], r["method"]) == (dataset, channel, method))
                     for method in METHODS}
            reference = next(item for item in m2000 if (item["dataset"], item["channel"]) == (dataset, channel))
            comparison.append(dict(
                dataset=dataset, channel=channel, m=m, **means,
                best_proposed=max(PROPOSED, key=means.get), best_baseline=max(BASELINES, key=means.get),
                gap_pp=max(means[k] for k in PROPOSED) - max(means[k] for k in BASELINES),
                m2000_fresh_gap_pp=max(reference["t1_data_dependent"], reference["T2-Matched"])
                - max(reference["ERM-Matched"], reference["L2-Matched"]),
                variants={method: selection["rows"][f"{dataset}/{channel}"]["methods"][method]["label"]
                          for method in METHODS}))
    output = PUBLISH / "performance"
    dd.write_csv(output / "deployment_by_seed.csv", rows)
    dd.write_json(output / "comparison.json", comparison)
    dd.write_json(output / "selection.json", selection)
    (output / "performance_table.tex").write_text(dd.render_performance_table(
        rows, rf"Deployment accuracy (\%) comparison at channel-set size $m={m}$."))
    print(json.dumps(comparison, indent=1))


# --------------------------------------------------------------------------- driver

def reuse_stage1(tasks, root: Path, source: Path):
    """Symlink finished stage-1 checkpoints with identical task IDs from an earlier campaign."""
    linked = 0
    for task in tasks:
        if task["meta"]["role"] != "stage1":
            continue
        target, origin = dd.run_dir(root, task["task_id"]), dd.run_dir(source, task["task_id"])
        # The task ID hashes the full stage-1 config, so equal IDs mean equal fits.
        if not target.exists() and (origin / "done.json").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(origin.resolve(), target_is_directory=True)
            linked += 1
    print(json.dumps(dict(event="reuse_stage1", linked=linked, source=str(source))), flush=True)


def pipeline(root: Path, workers: int, stage1_source: Path | None, stop_after: str | None):
    root.mkdir(parents=True, exist_ok=True)
    screen_path = root / "manifests" / "screen.jsonl"
    if not screen_path.exists():
        dd.write_manifest(screen_path, plan_screen(root))
    screen = dd.read_manifest(screen_path)
    if stage1_source:
        reuse_stage1(screen, root, stage1_source)
    dd.run_pool(screen, root, workers)
    if not (root / "selection.json").exists():
        select(root)
    if stop_after == "screen":
        return
    confirm_path = root / "manifests" / "confirm.jsonl"
    if not confirm_path.exists():
        dd.write_manifest(confirm_path, plan_confirm(root))
    confirm = dd.read_manifest(confirm_path)
    if stage1_source:
        reuse_stage1(confirm, root, stage1_source)
    dd.run_pool(confirm, root, workers)
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pipeline", "plan", "select", "summarize"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0))
                                                           if hasattr(os, "sched_getaffinity") else os.cpu_count()))
    parser.add_argument("--reuse-stage1", type=Path,
                        help="Earlier data_dependent performance root whose fresh stage-1 fits to link")
    parser.add_argument("--stop-after", choices=("screen",))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "pipeline":
        pipeline(root, args.workers, args.reuse_stage1, args.stop_after)
    elif args.command == "plan":
        tasks = plan_screen(root)
        fits = [task for task in tasks if task["meta"]["role"] == "stage2"]
        by_dataset = {}
        for task in fits:
            by_dataset[task["config"]["dataset"]] = by_dataset.get(task["config"]["dataset"], 0) + 1
        print(json.dumps(dict(tasks=len(tasks), fits=len(fits), stage1=len(tasks) - len(fits), by_dataset=by_dataset)))
    elif args.command == "select":
        select(root)
    else:
        summarize(root)


if __name__ == "__main__":
    main()
