"""T1 with a channel prior trained on part of the bank: U = (U_1, U_2).

Theorem 1's channel complexity needs a prior independent of the channels it is
evaluated on. Here the bank is split by index, U_1 = U[:m_1] and U_2 = U[m_1:].
Its m entries are i.i.d., so conditional on (S, U_1) the part U_2 is an i.i.d.
sample of m_2 = m - m_1 channels, and the theorem holds with

    Q_{W|S,U_1} = N(mu_{S,U_1}, s^2 I),   channel complexity over U_2 with m_2,

where mu_{S,U_1} is the mean of a stage-1 PAC-Bayes fit on S x U_1 (sample term
only, prior Q_W). No channel outside U is ever used, so T1 sees exactly the
channels ERM, L2 and T2 see. Stage 2 is T1 on all of U, started at mu_{S,U_1};
its channel term uses m_2 and the prior above. The bound's empirical risk is
then evaluated on S x U_2 (see split_bound.py for certificates).

The split fraction m_1/m and s form a finite family fixed before U is seen, so
the union bound uses K = len(SPLITS) * len(STD_MULTIPLIERS).

Studies
* ``performance``: deployment accuracy at each shared m in M_GRID, tuned on
  validation seeds and confirmed on the archived paired seeds. ERM, L2 and T2 are
  the already confirmed fits at that m (higher_accuracy for m = 20, 200, 1000;
  the archived table for m = 2000), so only T1 is trained.
* ``mismatch``: the CIFAR-10 erasure training-probability sweep at m = 2000 with
  the archived ERM, L2 and T2.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import statistics

from . import data_dependent as dd
from . import higher_accuracy as ha

PUBLISH = dd.SELECTED / "split_u"
SPLITS = {"split25": .25, "split50": .5}           # m_1 / m
STD_MULTIPLIERS = dd.STD_MULTIPLIERS               # s / posterior_std: 3, 10, 30, 100
ETA_MULTIPLIERS = dd.ETA_MULTIPLIERS               # eta_channel x 1, 10, 100
PRIOR_COUNT = len(SPLITS) * len(STD_MULTIPLIERS)   # K = 8
M_GRID = (20, 200, 1000, 2000)
PRIOR_KEYS = {"channel_prior", "channel_prior_std", "channel_prior_checkpoint", "channel_prior_count",
              "init_from_prior", "bank_split", "bank_part"}
BASELINE_TABLES = {
    20: dd.SELECTED / "higher_accuracy" / "performance" / "deployment_by_seed.csv",
    200: dd.SELECTED / "higher_accuracy" / "m200" / "performance" / "deployment_by_seed.csv",
    1000: dd.SELECTED / "higher_accuracy" / "m1000" / "performance" / "deployment_by_seed.csv",
    2000: dd.PUBLISH / "only_fresh" / "performance" / "deployment_by_seed.csv",
}
DATASET_ORDER = ha.DATASET_ORDER


def grid():
    return list(itertools.product(SPLITS, STD_MULTIPLIERS, ETA_MULTIPLIERS))


def bank_cut(m: int, construction: str) -> int:
    """m_1, keeping at least two channels on each side."""
    return min(m - 2, max(2, round(SPLITS[construction] * m)))


def stage1_config(t1: dict, construction: str) -> dict:
    """PAC-Bayes on S x U_1 with Q_W; never U_2."""
    config = {key: value for key, value in t1.items() if key not in PRIOR_KEYS}
    config.update(method="pac_bayes", bank_split=bank_cut(t1["m"], construction), bank_part="first")
    return config


def stage2_config(t1: dict, stage1_dir: Path, variant, prior_count: int) -> dict:
    construction, std_multiplier, eta_multiplier = variant[:3]
    config = {key: value for key, value in t1.items() if key not in PRIOR_KEYS}
    config.update(method="proposed", channel_prior="split", channel_prior_std=t1["posterior_std"] * std_multiplier,
                  channel_prior_checkpoint=str(stage1_dir), channel_prior_count=prior_count,
                  eta_channel=t1["eta_channel"] * eta_multiplier, bank_split=bank_cut(t1["m"], construction))
    if len(variant) > 3:
        config["eta_data"] = t1["eta_data"] * variant[3]
    return config


def t1_pair(study: str, t1: dict, variant, root: Path, evals, meta: dict, prior_count: int = PRIOR_COUNT):
    s1 = dd.new_run_config(stage1_config(t1, variant[0]))
    s1_id = f"{study}-stage1-{variant[0]}-{dd.digest(s1)[:12]}"
    s2 = dd.new_run_config(stage2_config(t1, dd.run_dir(root, s1_id), variant, prior_count))
    s2_id = f"{study}-T1-{dd.variant_label(variant)}-{dd.digest(s2)[:12]}"
    return [dict(task_id=s1_id, config=s1, deps=[], evals=[],
                 meta=dict(role="stage1", construction=variant[0], bank_split=s1["bank_split"])),
            dict(task_id=s2_id, config=s2, deps=[s1_id], evals=evals,
                 meta=dict(meta, role="stage2", variant=list(variant), label=dd.variant_label(variant),
                           bank_split=s2["bank_split"]))]


def ordered(tasks):
    tasks = dd.dedupe(tasks)
    return sorted(tasks, key=lambda task: DATASET_ORDER.index(task["config"]["dataset"]))


# --------------------------------------------------------------------------- performance

def t1_sources():
    return {(dataset, channel): seeds for (dataset, channel, method), seeds in ha.archived_sources().items()
            if method == "T1-Matched"}


def plan_performance(root: Path, phase: str, m_grid=M_GRID):
    tasks = []
    sources = t1_sources()
    if phase == "tune":
        for (dataset, channel), seeds in sorted(sources.items()):
            for offset in dd.TUNING_OFFSETS:
                base = dd.with_seeds(seeds[0]["config"], offset)
                evals = [dd.validation_eval(dd.new_run_config(base))]
                for m in m_grid:
                    for variant in grid():
                        tasks.extend(t1_pair(f"perf-m{m}", {**base, "m": m}, variant, root, evals,
                                             dict(dataset=dataset, channel=channel, m=m, seed=base["seed"])))
        return ordered(tasks)
    selection = dd.read_json(root / "selection.json")
    for (dataset, channel), seeds in sorted(sources.items()):
        for m in m_grid:
            variant = selection[f"m{m}/{dataset}/{channel}"]["variant"]
            for source in seeds:
                test = dict(name="test", split="test", scope="deployment",
                            overrides={**source["evaluation"], "device": "cpu", "threads": 1})
                tasks.extend(t1_pair(f"perf-m{m}", {**source["config"], "m": m}, variant, root, [test],
                                     dict(dataset=dataset, channel=channel, m=m, seed=source["seed"], phase="confirm",
                                          baseline_task=source["baseline_task"])))
    return ordered(tasks)


def performance_group(record):
    meta = record["meta"]
    return f"m{meta['m']}/{meta['dataset']}/{meta['channel']}"


def summarize_performance(root: Path, m_grid=M_GRID):
    tasks = [task for task in dd.read_manifest(root / "manifests" / "confirm.jsonl")
             if task["meta"]["role"] == "stage2"]
    records = {record["task_id"]: record for record in dd.done_records(root, tasks)}
    selection = dd.read_json(root / "selection.json")
    overview = []
    for m in m_grid:
        with BASELINE_TABLES[m].open(newline="") as stream:
            baseline = list(csv.DictReader(stream))
        fresh_t1 = [row for row in baseline if row["method"] == "T1-Matched"]
        rows = [dict(row, variant=row.get("variant", ""), channel_kl=row.get("channel_kl", ""),
                     channel_complexity=row.get("channel_complexity", ""))
                for row in baseline if row["method"] != "T1-Matched"]
        for task in tasks:
            meta = task["meta"]
            if meta["m"] != m:
                continue
            report = dd.read_json(dd.run_dir(root, task["task_id"]) / "test" / "evaluation.json")
            heldout = report["heldout_gibbs_mc"]
            if report["evaluation_config"]["deployment_channel"]["kind"] == "diagonal_fading":
                if not all(abs(r["feature_power_per_coordinate"] - 1.) <= 2e-5 for r in heldout["replicate_statistics"]):
                    raise ValueError(f"Unit transmit power violated in {task['task_id']}")
            complexities = report["final_posterior_complexities"]
            rows.append(dict(rows[0], dataset=meta["dataset"], channel=meta["channel"], method="T1-Matched",
                             seed=str(meta["seed"]), m=str(m), test_accuracy_pct=100 * heldout["accuracy"],
                             draws=heldout["draws"], examples=heldout["examples"], task_id=task["task_id"],
                             training_revision=dd.read_json(dd.run_dir(root, task["task_id"]) / "metadata.json")["git_sha"],
                             archive_key=f"split_u/{task['task_id']}", weights_sha256=records[task["task_id"]]["weights_sha256"],
                             variant=meta["label"], channel_kl=complexities["channel_kl"],
                             channel_complexity=complexities["channel_complexity"]))
        rows = [{key: row.get(key, "") for key in rows[0]} for row in rows]
        rows.sort(key=lambda r: (list(dd.DATASETS).index(r["dataset"]), dd.CHANNELS.index(r["channel"]),
                                 r["method"], int(r["seed"])))
        comparison = []
        for dataset in dd.DATASETS:
            for channel in dd.CHANNELS:
                def mean(method, source=rows):
                    values = [float(r["test_accuracy_pct"]) for r in source
                              if (r["dataset"], r["channel"], r["method"]) == (dataset, channel, method)]
                    return statistics.fmean(values) if values else None
                means = {method: mean(method) for method in ha.METHODS}
                proposed = max(means["T1-Matched"], means["T2-Matched"])
                baseline_best = max(means["ERM-Matched"], means["L2-Matched"])
                comparison.append(dict(dataset=dataset, channel=channel, m=m, **means,
                                       t1_fresh=mean("T1-Matched", fresh_t1),
                                       best_proposed=max(ha.PROPOSED, key=means.get),
                                       best_baseline=max(ha.BASELINES, key=means.get),
                                       gap_pp=proposed - baseline_best,
                                       variant=selection[f"m{m}/{dataset}/{channel}"]["label"]))
        output = PUBLISH / f"performance_m{m}"
        dd.write_csv(output / "deployment_by_seed.csv", rows)
        dd.write_json(output / "comparison.json", comparison)
        (output / "performance_table.tex").write_text(dd.render_performance_table(
            rows, rf"Deployment accuracy (\%) comparison at channel-set size $m={m}$, T1 with $Q_{{W|S,U_1}}$."))
        gaps = [row["gap_pp"] for row in comparison]
        overview.append(dict(m=m, rows_won=sum(gap > 0 for gap in gaps), mean_gap_pp=statistics.fmean(gaps),
                             min_gap_pp=min(gaps),
                             gaps_pp={f"{r['dataset']}/{r['channel']}": r["gap_pp"] for r in comparison}))
    dd.write_json(PUBLISH / "performance_overview.json", overview)
    dd.write_json(PUBLISH / "performance_selection.json", selection)
    print(json.dumps(overview, indent=1))


# --------------------------------------------------------------------------- CIFAR-10 mismatch

def plan_mismatch(root: Path, phase: str):
    sources = [source for source in dd.mismatch_sources() if source["p"] > 0]
    tasks = []
    if phase == "tune":
        first = {}
        for source in sources:
            first.setdefault(source["p"], source)
        for p in dd.MISMATCH_TUNING_PROBABILITIES:
            for offset in dd.TUNING_OFFSETS:
                base = dd.with_seeds(first[p]["config"], offset)
                for variant in grid():
                    tasks.extend(t1_pair("mismatch", base, variant, root, [dd.validation_eval(dd.new_run_config(base))],
                                         dict(p=p, seed=base["seed"])))
        return dd.dedupe(tasks)
    variant = dd.read_json(root / "selection.json")["all"]["variant"]
    for source in sources:
        test = dict(name="test_full", split="test", scope="full",
                    overrides={**source["evaluation"], "device": "cpu", "threads": 1})
        tasks.extend(t1_pair("mismatch", source["config"], variant, root, [test],
                             dict(p=source["p"], seed=source["config"]["seed"], phase="confirm",
                                  archived_task=source["archived_task"])))
    return dd.dedupe(tasks)


def summarize_mismatch(root: Path):
    tasks = dd.read_manifest(root / "manifests" / "confirm.jsonl")
    with (dd.MISMATCH_DATA / "observations.csv").open(newline="") as stream:
        archived = list(csv.DictReader(stream))
    # p_tr = 0 has a deterministic bank and no channel complexity; the archived T1 fits stay.
    rows = [row for row in archived if row["method"] != "T1" or float(row["sweep_value"]) == 0.]
    for task in tasks:
        if task["meta"].get("role") != "stage2":
            continue
        report = dd.read_json(dd.run_dir(root, task["task_id"]) / "test_full" / "evaluation.json")
        p = task["meta"]["p"]
        rows.append(dict(task_id=task["task_id"], source_task_id=task["meta"]["archived_task"],
                         group=f"cifar10_outage_ptr{round(p * 1000):04d}", experiment="p_train_sweep", method="T1",
                         seed=str(task["meta"]["seed"]), sweep_value=str(p), m="2000",
                         test_accuracy=report["heldout_gibbs_mc"]["accuracy"],
                         bounded_test_risk=report["heldout_gibbs_mc"]["bounded_loss"]))
    output = PUBLISH / "cifar10_mismatch"
    dd.write_csv(output / "observations.csv", rows)
    dd.write_json(output / "selection.json", dd.read_json(root / "selection.json"))
    from results.final_chosen_results import draw_cifar10_channel_mismatch as drawing
    drawing.SOURCE = output / "observations.csv"
    drawing.OUTPUT_DIR = PUBLISH
    drawing.main()


# --------------------------------------------------------------------------- driver

STUDIES = {
    "performance": (plan_performance, summarize_performance, performance_group),
    "mismatch": (plan_mismatch, summarize_mismatch, lambda record: "all"),
}


def pipeline(study: str, root: Path, workers: int, stop_after: str | None = None):
    root.mkdir(parents=True, exist_ok=True)
    planner, summarize, group = STUDIES[study]
    tune_path = root / "manifests" / "tune.jsonl"
    if not tune_path.exists():
        dd.write_manifest(tune_path, planner(root, "tune"))
    tune = dd.read_manifest(tune_path)
    dd.run_pool(tune, root, workers)
    if not (root / "selection.json").exists():
        dd.write_json(root / "selection.json", dd.select_variant(dd.done_records(root, tune), group))
    if stop_after == "tune":
        return
    confirm_path = root / "manifests" / "confirm.jsonl"
    if not confirm_path.exists():
        dd.write_manifest(confirm_path, planner(root, "confirm"))
    dd.run_pool(dd.read_manifest(confirm_path), root, workers)
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pipeline", "plan", "summarize"))
    parser.add_argument("--study", required=True, choices=(*STUDIES, "bounds"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0))
                                                           if hasattr(os, "sched_getaffinity") else os.cpu_count()))
    parser.add_argument("--phase", choices=("tune", "confirm"), default="tune")
    parser.add_argument("--stop-after", choices=("tune",))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.study == "bounds":
        from . import split_bound
        split_bound.pipeline(root, args.workers)
        return
    if args.command == "pipeline":
        pipeline(args.study, root, args.workers, args.stop_after)
    elif args.command == "plan":
        tasks = STUDIES[args.study][0](root, args.phase)
        print(json.dumps(dict(tasks=len(tasks), stage2=sum(t["meta"]["role"] == "stage2" for t in tasks))))
    else:
        STUDIES[args.study][1](root)


if __name__ == "__main__":
    main()
