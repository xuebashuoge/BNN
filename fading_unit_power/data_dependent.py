"""T1 with a data-dependent channel prior Q_{W|S} for the four journal studies.

The revised Theorem 1 bounds the channel complexity with D(P_{W|US} || Q_{W|S})
for any prior that may depend on the dataset S but not on the channel bank U.
The sample complexity keeps the fixed prior N(0, prior_std^2 I). Each T1 fit
here is two-stage:

1. Stage 1 trains a Gaussian posterior on the same S without the bank U, using
   one of three constructions: the noiseless channel ("perfect"), an
   independently seeded bank U' of the same size ("independent_bank"), or fresh
   P_tr draws for every minibatch ("fresh"). Its posterior mean is mu_S.
2. Stage 2 is T1 with Q_{W|S} = N(mu_S, s^2 I), starting from mu_S.

Construction, s (a multiple of the posterior initial scale) and the channel
penalty weight are chosen by validation accuracy on tuning seeds that are
disjoint from the confirmation seeds. The confirmation reuses the paired
seeds and evaluation seeds of the archived baselines, so ERM, L2 and T2 are
not retrained. Each study then publishes its figure or table.

Studies: ``performance`` (deployment table at m=2000), ``mnist`` (bank-size
sweep), ``cifar_mismatch`` (training erasure probability sweep) and
``bounds`` (two-moons certificate decomposition, see dd_bounds.py).
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import tarfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
SELECTED = REPO / "results" / "final_chosen_results"
PUBLISH = SELECTED / "data_dependent"
# A construction-restricted rerun (see --constructions) sets these to tagged names.
SELECTION_NAME = "selection.json"
CONFIRM_NAME = "confirm.jsonl"
CONSTRUCTIONS = ("perfect", "independent_bank", "fresh")
STD_MULTIPLIERS = (3., 10., 30., 100.)
ETA_MULTIPLIERS = (1., 10., 100.)
TUNING_OFFSETS = (10_000_000, 20_000_000)
INDEPENDENT_BANK_OFFSET = 500_000
DATASETS = {"moons": "Two moons", "mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
CHANNELS = ("Fading", "Erasure")


# --------------------------------------------------------------------------- utilities

def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_manifest(path: Path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    path.write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks))


def run_dir(root: Path, task_id: str) -> Path:
    return root / "runs" / task_id


# --------------------------------------------------------------------------- task construction

def new_run_config(config: dict) -> dict:
    """Common runtime settings for the campaign; one thread per pooled worker."""
    return {**config, "threads": 1, "device": "cpu"}


def with_seeds(config: dict, offset: int) -> dict:
    return {**config, "seed": config["seed"] + offset, "channel_seed": config["channel_seed"] + offset,
            "eval_seed": config["eval_seed"] + offset}


def stage1_config(t1: dict, construction: str) -> dict:
    """Stage 1 shares S, the architecture and the schedule; never the bank U."""
    config = {key: value for key, value in t1.items()
              if key not in {"channel_prior", "channel_prior_std", "channel_prior_checkpoint",
                             "channel_prior_count", "init_from_prior"}}
    config["method"] = "pac_bayes"
    if construction == "perfect":
        config["train_channel"] = {"kind": "perfect"}
    elif construction == "independent_bank":
        config["channel_seed"] = t1["channel_seed"] + INDEPENDENT_BANK_OFFSET
    elif construction == "fresh":
        config["train_channel_source"] = "fresh"
    else:
        raise ValueError(construction)
    return config


def variant_label(variant) -> str:
    construction, std_multiplier, eta_multiplier = variant[:3]
    label = f"{construction}-s{std_multiplier:g}-e{eta_multiplier:g}"
    return label + (f"-d{variant[3]:g}" if len(variant) > 3 else "")


def stage2_config(t1: dict, stage1_dir: Path, variant, prior_count: int) -> dict:
    construction, std_multiplier, eta_multiplier = variant[:3]
    config = {**t1, "method": "proposed", "channel_prior": construction,
              "channel_prior_std": t1["posterior_std"] * std_multiplier,
              "channel_prior_checkpoint": str(stage1_dir), "channel_prior_count": prior_count,
              "eta_channel": t1["eta_channel"] * eta_multiplier}
    if len(variant) > 3:
        config["eta_data"] = t1["eta_data"] * variant[3]
    return config


def t1_pair(study: str, t1: dict, variant, root: Path, evals, meta: dict, prior_count: int):
    """Return (stage-1 task, stage-2 task) for one T1 fit."""
    s1 = new_run_config(stage1_config(t1, variant[0]))
    s1_id = f"{study}-stage1-{variant[0]}-{digest(s1)[:12]}"
    s2 = new_run_config(stage2_config(t1, run_dir(root, s1_id), variant, prior_count))
    s2_id = f"{study}-T1-{variant_label(variant)}-{digest(s2)[:12]}"
    stage1 = dict(task_id=s1_id, config=s1, deps=[], evals=[], meta=dict(role="stage1", construction=variant[0]))
    stage2 = dict(task_id=s2_id, config=s2, deps=[s1_id], evals=evals,
                  meta=dict(meta, role="stage2", variant=list(variant), label=variant_label(variant)))
    return stage1, stage2


def validation_eval(config: dict) -> dict:
    overrides = dict(eval_draws=10, eval_seed=config["eval_seed"] + 7_777, device="cpu", threads=1)
    if config["dataset"] == "moons":
        # The 200-example validation split cannot separate close candidates.
        overrides.update(moons_eval_data_seed=config["data_seed"] + 7_919, moons_eval_size=20_000)
    return dict(name="validation", split="validation", scope="deployment", overrides=overrides)


def dedupe(tasks):
    unique = {}
    for task in tasks:
        previous = unique.setdefault(task["task_id"], task)
        if previous["config"] != task["config"]:
            raise ValueError(f"Task ID collision: {task['task_id']}")
    return list(unique.values())


def default_grid():
    return list(itertools.product(CONSTRUCTIONS, STD_MULTIPLIERS, ETA_MULTIPLIERS))


PRIOR_COUNT = len(CONSTRUCTIONS) * len(STD_MULTIPLIERS)


# --------------------------------------------------------------------------- execution

def execute_task(task: dict, root: str) -> dict:
    """Train (resumably) and evaluate one task in a fresh worker process."""
    import torch
    from .trainer import run_experiment as trainer

    root = Path(root)
    checkpoint = run_dir(root, task["task_id"])
    done = checkpoint / "done.json"
    if done.exists():
        return read_json(done)
    start = time.perf_counter()
    config = trainer.normalize_config(task["config"])
    metadata_path = checkpoint / "metadata.json"
    complete = (metadata_path.exists() and (checkpoint / "weights.pt").exists()
                and "total_seconds" in read_json(metadata_path))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if not complete:
            trainer.check_device(config)
            torch.set_num_threads(config["threads"])
            trainer.train(config, checkpoint)
        results = {}
        for evaluation in task["evals"]:
            output = checkpoint / evaluation["name"]
            report_path = output / "evaluation.json"
            if not report_path.exists():
                overrides = output / "overrides.json"
                write_json(overrides, evaluation["overrides"])
                trainer.evaluate_checkpoint(checkpoint, str(overrides), output, evaluation["split"],
                                            evaluation["scope"])
            report = read_json(report_path)
            results[evaluation["name"]] = dict(accuracy=report["heldout_gibbs_mc"]["accuracy"],
                                               bounded_loss=report["heldout_gibbs_mc"]["bounded_loss"])
    metadata = read_json(metadata_path)
    record = dict(task_id=task["task_id"], meta=task["meta"], seconds=time.perf_counter() - start,
                  selected_epoch=metadata.get("selected_epoch"),
                  selected_validation_accuracy=metadata.get("selected_validation_accuracy"),
                  weights_sha256=trainer.file_fingerprint(checkpoint / "weights.pt")["sha256"],
                  evaluations=results)
    write_json(done, record)
    return record


def run_pool(tasks, root: Path, workers: int):
    """Dependency-ordered process pool; one torch thread per worker."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    by_id = {task["task_id"]: task for task in tasks}
    finished = {task_id for task_id in by_id if (run_dir(root, task_id) / "done.json").exists()}
    failed, running = set(), {}
    pending = [task_id for task_id in by_id if task_id not in finished]
    print(json.dumps(dict(event="pool", tasks=len(by_id), already_done=len(finished), workers=workers)), flush=True)
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        while pending or running:
            for task_id in list(pending):
                deps = by_id[task_id]["deps"]
                if any(dep in failed for dep in deps):
                    pending.remove(task_id)
                    failed.add(task_id)
                    print(json.dumps(dict(event="skipped", task=task_id)), flush=True)
                elif all(dep in finished for dep in deps) and len(running) < 2 * workers:
                    pending.remove(task_id)
                    running[pool.submit(execute_task, by_id[task_id], str(root))] = task_id
            if not running:
                if pending:
                    raise RuntimeError("Unsatisfiable dependencies: " + ", ".join(pending[:5]))
                break
            completed, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
            for future in completed:
                task_id = running.pop(future)
                try:
                    record = future.result()
                    finished.add(task_id)
                    print(json.dumps(dict(event="done", task=task_id, seconds=round(record["seconds"], 1),
                                          evaluations=record["evaluations"])), flush=True)
                except Exception:
                    failed.add(task_id)
                    print(json.dumps(dict(event="failed", task=task_id, error=traceback.format_exc())), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} tasks failed or were skipped")


def done_records(root: Path, tasks):
    return [read_json(run_dir(root, task["task_id"]) / "done.json") for task in tasks
            if task["meta"].get("role") == "stage2"]


def select_variant(records, key):
    """Highest mean validation accuracy per group; ties favour larger penalty."""
    scores = {}
    for record in records:
        group = key(record)
        scores.setdefault(group, {}).setdefault(tuple(record["meta"]["variant"]), []).append(
            record["evaluations"]["validation"]["accuracy"])
    selection = {}
    for group, variants in scores.items():
        counts = {len(values) for values in variants.values()}
        if len(counts) != 1:
            raise ValueError(f"Unequal tuning seeds in {group}")
        ranked = sorted(variants.items(), key=lambda item: (-statistics.fmean(item[1]), -item[0][2], -item[0][1]))
        selection[group] = dict(variant=list(ranked[0][0]), label=variant_label(ranked[0][0]),
                                validation_accuracy=statistics.fmean(ranked[0][1]),
                                leaderboard=[dict(label=variant_label(v), mean=statistics.fmean(a), seeds=a)
                                             for v, a in ranked[:10]],
                                candidates=len(ranked))
    return selection


# --------------------------------------------------------------------------- performance table

PERFORMANCE_BASE = SELECTED / "fading_unit_power" / "performance"


def performance_baselines():
    with (PERFORMANCE_BASE / "deployment_by_seed.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def performance_t1_sources():
    """Archived T1 confirmation configs and test-evaluation overrides per row."""
    rows = [row for row in performance_baselines() if row["method"] == "T1-Matched"]
    fading_manifests = {}
    for dataset in DATASETS:
        for phase in ("primary", "refined"):
            path = PERFORMANCE_BASE / "provenance" / dataset / phase / "confirmation.jsonl"
            for entry in read_manifest(path):
                fading_manifests[entry["task_id"]] = entry
    erasure = {}
    with tarfile.open(SELECTED / "data" / "performance" / "selected_trials.tar.gz") as archive:
        for row in rows:
            if row["channel"] != "Erasure":
                continue
            key = row["archive_key"]
            config = json.load(archive.extractfile(key + "/task_config.json"))
            overrides = json.load(archive.extractfile(key + "/test_deployment/overrides.json"))
            erasure[row["task_id"]] = dict(config=config, evaluation=overrides)
    sources = {}
    for row in rows:
        entry = fading_manifests[row["task_id"]] if row["channel"] == "Fading" else erasure[row["task_id"]]
        config, evaluation = dict(entry["config"]), dict(entry["evaluation"])
        if int(config["seed"]) != int(row["seed"]) or config["dataset"] != row["dataset"] or config["m"] != 2000:
            raise ValueError(f"Archived T1 source mismatch for {row['task_id']}")
        if row["channel"] == "Fading" and not config.get("transmit_unit_power"):
            raise ValueError("Fading rows must keep unit transmit power")
        sources.setdefault((row["dataset"], row["channel"]), []).append(
            dict(config=config, evaluation=evaluation, baseline_task=row["task_id"]))
    return sources


def plan_performance(root: Path, phase: str):
    sources = performance_t1_sources()
    tasks = []
    if phase == "tune":
        for (dataset, channel), seeds in sorted(sources.items()):
            base = seeds[0]["config"]
            for offset in TUNING_OFFSETS:
                t1 = with_seeds(base, offset)
                for variant in default_grid():
                    tasks.extend(t1_pair("performance", t1, variant, root, [validation_eval(new_run_config(t1))],
                                         dict(dataset=dataset, channel=channel, seed=t1["seed"]), PRIOR_COUNT))
        return dedupe(tasks)
    selection = read_json(root / SELECTION_NAME)
    for (dataset, channel), seeds in sorted(sources.items()):
        variant = selection[f"{dataset}/{channel}"]["variant"]
        for source in seeds:
            t1 = source["config"]
            test = dict(name="test", split="test", scope="deployment",
                        overrides={**source["evaluation"], "device": "cpu", "threads": 1})
            tasks.extend(t1_pair("performance", t1, variant, root, [test, validation_eval(new_run_config(t1))],
                                 dict(dataset=dataset, channel=channel, seed=t1["seed"], phase="confirm",
                                      baseline_task=source["baseline_task"]), PRIOR_COUNT))
    return dedupe(tasks)


def render_performance_table(rows, caption):
    methods = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched", "Proposed-Perfect")
    lines = [r"\begin{table*}[htb]", r"\centering", r"\caption{" + caption + "}",
             r"\label{tab:performance}", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{}}",
             r"\toprule", r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Channel}",
             r"& \multicolumn{4}{c}{Matched training} & \multirow{2}{*}{Perfect training} \\",
             r"\cmidrule(lr){3-6}", r"& & ERM & L2 & T1 & T2 &  \\"]
    for dataset, label in DATASETS.items():
        lines.extend([r"\midrule", r"\multirow{2}{*}{" + label + "}"])
        for channel in CHANNELS:
            values = [[float(r["test_accuracy_pct"]) for r in rows
                       if (r["dataset"], r["channel"], r["method"]) == (dataset, channel, method)]
                      for method in methods]
            means = [statistics.fmean(v) for v in values]
            cells = []
            for by_seed, average in zip(values, means):
                printed = f"{average:.2f}"
                if max(means) - average <= 1e-10:
                    printed = r"\mathbf{" + printed + "}"
                cells.append("$" + printed + r"\pm" + f"{statistics.stdev(by_seed):.2f}$")
            lines.extend(["& " + channel, "& " + " & ".join(cells) + r" \\"])
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])


def summarize_performance(root: Path):
    tasks = read_manifest(root / "manifests" / CONFIRM_NAME)
    records = {record["task_id"]: record for record in done_records(root, tasks)}
    baselines = performance_baselines()
    new_rows, comparison = [], []
    for task in tasks:
        if task["meta"].get("role") != "stage2":
            continue
        meta, record = task["meta"], records[task["task_id"]]
        report = read_json(run_dir(root, task["task_id"]) / "test" / "evaluation.json")
        heldout = report["heldout_gibbs_mc"]
        if report["evaluation_config"]["deployment_channel"]["kind"] == "diagonal_fading":
            if not all(math.isclose(r["feature_power_per_coordinate"], 1., abs_tol=2e-5)
                       for r in heldout["replicate_statistics"]):
                raise ValueError(f"Unit transmit power violated in {task['task_id']}")
        new_rows.append(dict(dataset=meta["dataset"], channel=meta["channel"], method="T1-Matched",
                             seed=str(meta["seed"]), m="2000", test_accuracy_pct=100 * heldout["accuracy"],
                             draws=heldout["draws"], examples=heldout["examples"], task_id=task["task_id"],
                             training_revision=read_json(run_dir(root, task["task_id"]) / "metadata.json")["git_sha"],
                             archive_key=f"data_dependent/{task['task_id']}", weights_sha256=record["weights_sha256"],
                             variant=meta["label"],
                             channel_kl=report["final_posterior_complexities"]["channel_kl"],
                             channel_complexity=report["final_posterior_complexities"]["channel_complexity"]))
    rows = [dict(row, variant="", channel_kl="", channel_complexity="") for row in baselines
            if row["method"] != "T1-Matched"] + new_rows
    key = lambda r: (list(DATASETS).index(r["dataset"]), CHANNELS.index(r["channel"]), r["method"], int(r["seed"]))
    rows.sort(key=key)
    for dataset in DATASETS:
        for channel in CHANNELS:
            def mean_of(method, source):
                return statistics.fmean(float(r["test_accuracy_pct"]) for r in source
                                        if (r["dataset"], r["channel"], r["method"]) == (dataset, channel, method))
            others = {m: mean_of(m, rows) for m in ("ERM-Matched", "L2-Matched", "T2-Matched")}
            comparison.append(dict(dataset=dataset, channel=channel, t1_data_dependent=mean_of("T1-Matched", rows),
                                   t1_fixed_prior=mean_of("T1-Matched", baselines), **others,
                                   gap_to_best_other=mean_of("T1-Matched", rows) - max(others.values())))
    output = PUBLISH / "performance"
    write_csv(output / "deployment_by_seed.csv", rows)
    write_json(output / "comparison.json", comparison)
    (output / "performance_table.tex").write_text(render_performance_table(
        rows, r"Deployment accuracy (\%) comparison at channel-set size $m=2000$."))
    write_json(output / "selection.json", read_json(root / SELECTION_NAME))
    print(json.dumps(comparison, indent=1))


# --------------------------------------------------------------------------- MNIST bank sweep

MNIST_DATA = SELECTED / "data" / "mnist_bank"


def mnist_sources():
    entries = read_manifest(MNIST_DATA / "manifest.jsonl")
    sources = []
    with tarfile.open(MNIST_DATA / "run_records.tar.gz", "r:gz") as records:
        for entry in entries:
            report = json.load(records.extractfile(f"runs/{entry['task_id']}/test_full/evaluation.json"))
            evaluation = {key: report["evaluation_config"][key] for key in ("eval_seed", "eval_draws")}
            sources.append(dict(config={**entry["config"], "transmit_unit_power": True}, evaluation=evaluation,
                                m=entry["sweep_value"], archived_task=entry["task_id"]))
    return sources


def plan_mnist(root: Path, phase: str):
    sources = mnist_sources()
    tasks = []
    if phase == "tune":
        first_seed = {}
        for source in sources:
            first_seed.setdefault(source["m"], source)
        for m, source in sorted(first_seed.items()):
            for offset in TUNING_OFFSETS:
                t1 = with_seeds(source["config"], offset)
                for variant in default_grid():
                    tasks.extend(t1_pair("mnist", t1, variant, root, [validation_eval(new_run_config(t1))],
                                         dict(m=m, seed=t1["seed"]), PRIOR_COUNT))
        return dedupe(tasks)
    variant = read_json(root / SELECTION_NAME)["all"]["variant"]
    for source in sources:
        test = dict(name="test_full", split="test", scope="full",
                    overrides={**source["evaluation"], "device": "cpu", "threads": 1})
        tasks.extend(t1_pair("mnist", source["config"], variant, root, [test],
                             dict(m=source["m"], seed=source["config"]["seed"], phase="confirm",
                                  archived_task=source["archived_task"]), PRIOR_COUNT))
    return dedupe(tasks)


def summarize_mnist(root: Path):
    from .fixed_studies_plots import plot_mnist

    tasks = read_manifest(root / "manifests" / CONFIRM_NAME)
    observations = []
    for task in tasks:
        if task["meta"].get("role") != "stage2":
            continue
        checkpoint = run_dir(root, task["task_id"])
        report = read_json(checkpoint / "test_full" / "evaluation.json")
        heldout = report["heldout_gibbs_mc"]
        for row in (heldout, *heldout["replicate_statistics"]):
            if not math.isclose(row["feature_power_per_coordinate"], 1., abs_tol=2e-5):
                raise ValueError(f"Unit transmit power violated in {task['task_id']}")
        complexities = report["final_posterior_complexities"]
        observations.append(dict(method="T1", m=task["meta"]["m"], seed=task["meta"]["seed"],
                                 test_accuracy=heldout["accuracy"], generalization_error=report["bounded_gap_mc"],
                                 seconds=read_json(checkpoint / "metadata.json")["total_seconds"],
                                 feature_power_per_coordinate=heldout["feature_power_per_coordinate"],
                                 channel_kl=complexities["channel_kl"],
                                 channel_complexity=complexities["channel_complexity"],
                                 variant=task["meta"]["label"],
                                 weights_sha256=report["checkpoint_source_fingerprints"]["weights"]["sha256"]))
    observations.sort(key=lambda row: (row["m"], row["seed"]))
    output = PUBLISH / "mnist_bank"
    write_csv(output / "observations.csv", observations)
    write_json(output / "selection.json", read_json(root / SELECTION_NAME))
    plot_mnist(observations, PUBLISH / "mnist_analog_finite_channel_generalization")


# --------------------------------------------------------------------------- CIFAR-10 erasure mismatch

MISMATCH_DATA = SELECTED / "data" / "cifar10_mismatch"
MISMATCH_TUNING_PROBABILITIES = (.3, .5, .7)


def mismatch_sources():
    sources = []
    with tarfile.open(MISMATCH_DATA / "run_records.tar.gz", "r:gz") as records:
        for shard in range(13):
            for entry in read_manifest(MISMATCH_DATA / "manifests" / f"{shard}.jsonl"):
                if entry["method_label"] != "T1":
                    continue
                report = json.load(records.extractfile(f"runs/{entry['task_id']}/test_full/evaluation.json"))
                evaluation = {key: report["evaluation_config"][key] for key in ("eval_seed", "eval_draws")}
                sources.append(dict(config=entry["config"], evaluation=evaluation, p=entry["sweep_value"],
                                    archived_task=entry["task_id"]))
    return sources


def plan_mismatch(root: Path, phase: str):
    sources = [source for source in mismatch_sources() if source["p"] > 0]
    tasks = []
    if phase == "tune":
        first_seed = {}
        for source in sources:
            first_seed.setdefault(source["p"], source)
        for p in MISMATCH_TUNING_PROBABILITIES:
            for offset in TUNING_OFFSETS:
                t1 = with_seeds(first_seed[p]["config"], offset)
                for variant in default_grid():
                    tasks.extend(t1_pair("mismatch", t1, variant, root, [validation_eval(new_run_config(t1))],
                                         dict(p=p, seed=t1["seed"]), PRIOR_COUNT))
        return dedupe(tasks)
    variant = read_json(root / SELECTION_NAME)["all"]["variant"]
    for source in sources:
        test = dict(name="test_full", split="test", scope="full",
                    overrides={**source["evaluation"], "device": "cpu", "threads": 1})
        tasks.extend(t1_pair("mismatch", source["config"], variant, root, [test],
                             dict(p=source["p"], seed=source["config"]["seed"], phase="confirm",
                                  archived_task=source["archived_task"]), PRIOR_COUNT))
    return dedupe(tasks)


def summarize_mismatch(root: Path):
    tasks = read_manifest(root / "manifests" / CONFIRM_NAME)
    with (MISMATCH_DATA / "observations.csv").open(newline="") as stream:
        archived = list(csv.DictReader(stream))
    # p_tr = 0 has a deterministic bank and no channel complexity, so the
    # data-dependent prior cannot enter; the archived T1 fits are unchanged.
    rows = [row for row in archived if row["method"] != "T1" or float(row["sweep_value"]) == 0.]
    for task in tasks:
        if task["meta"].get("role") != "stage2":
            continue
        report = read_json(run_dir(root, task["task_id"]) / "test_full" / "evaluation.json")
        p = task["meta"]["p"]
        rows.append(dict(task_id=task["task_id"], source_task_id=task["meta"]["archived_task"],
                         group=f"cifar10_outage_ptr{round(p * 1000):04d}", experiment="p_train_sweep", method="T1",
                         seed=str(task["meta"]["seed"]), sweep_value=str(p), m="2000",
                         test_accuracy=report["heldout_gibbs_mc"]["accuracy"],
                         bounded_test_risk=report["heldout_gibbs_mc"]["bounded_loss"]))
    output = PUBLISH / "cifar10_mismatch"
    write_csv(output / "observations.csv", rows)
    write_json(output / "selection.json", read_json(root / SELECTION_NAME))
    from results.final_chosen_results import draw_cifar10_channel_mismatch as drawing
    drawing.SOURCE = output / "observations.csv"
    drawing.OUTPUT_DIR = PUBLISH
    drawing.main()


# --------------------------------------------------------------------------- driver

STUDIES = {
    "performance": (plan_performance, summarize_performance,
                    lambda record: f"{record['meta']['dataset']}/{record['meta']['channel']}"),
    "mnist": (plan_mnist, summarize_mnist, lambda record: "all"),
    "cifar_mismatch": (plan_mismatch, summarize_mismatch, lambda record: "all"),
}


def restrict(constructions):
    """Select only among the given stage-1 constructions; publish under a tag.

    The tuning fits are shared, so a restricted selection needs no new tuning.
    The union bound keeps the full prior family (PRIOR_COUNT), which is
    conservative and leaves task hashes unchanged.
    """
    global SELECTION_NAME, CONFIRM_NAME, PUBLISH
    unknown = set(constructions) - set(CONSTRUCTIONS)
    if unknown:
        raise ValueError(f"Unknown constructions {sorted(unknown)}")
    tag = "only_" + "_".join(sorted(constructions))
    SELECTION_NAME, CONFIRM_NAME = f"selection-{tag}.json", f"confirm-{tag}.jsonl"
    PUBLISH = SELECTED / "data_dependent" / tag
    return tag


def pipeline(study: str, root: Path, workers: int, stop_after: str | None = None, constructions=None):
    root.mkdir(parents=True, exist_ok=True)
    if study == "bounds":
        from . import dd_bounds
        dd_bounds.pipeline(root, workers, constructions)
        return
    planner, summarize, group = STUDIES[study]
    tune_path = root / "manifests" / "tune.jsonl"
    if not tune_path.exists():
        write_manifest(tune_path, planner(root, "tune"))
    tune = read_manifest(tune_path)
    run_pool(tune, root, workers)
    if not (root / SELECTION_NAME).exists():
        records = done_records(root, tune)
        if constructions:
            records = [record for record in records if record["meta"]["variant"][0] in constructions]
        write_json(root / SELECTION_NAME, select_variant(records, group))
    if stop_after == "tune":
        return
    confirm_path = root / "manifests" / CONFIRM_NAME
    if not confirm_path.exists():
        write_manifest(confirm_path, planner(root, "confirm"))
    run_pool(read_manifest(confirm_path), root, workers)
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pipeline", "plan", "summarize"))
    parser.add_argument("--study", required=True, choices=(*STUDIES, "bounds"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0))
                                                           if hasattr(os, "sched_getaffinity") else os.cpu_count()))
    parser.add_argument("--phase", choices=("tune", "confirm"), default="tune")
    parser.add_argument("--stop-after", choices=("tune",))
    parser.add_argument("--constructions", nargs="+", choices=CONSTRUCTIONS,
                        help="Select only among these stage-1 constructions (tagged outputs)")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.constructions:
        restrict(args.constructions)
    if args.command == "pipeline":
        pipeline(args.study, root, args.workers, args.stop_after, args.constructions)
    elif args.command == "plan":
        tasks = STUDIES[args.study][0](root, args.phase)
        print(json.dumps(dict(tasks=len(tasks), stage2=sum(t["meta"]["role"] == "stage2" for t in tasks))))
    else:
        STUDIES[args.study][1](root)


if __name__ == "__main__":
    main()
