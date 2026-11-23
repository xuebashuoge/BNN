"""Screen the unit-power deployment configs and report paired test results."""

import argparse
from collections import defaultdict
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

from results.final_chosen_results import rebuild_performance_table as historical_table


DATA = historical_table.DATA
DATASETS = tuple(historical_table.DATASETS)
METHODS = historical_table.METHODS
PROPOSED = ("T1-Matched", "T2-Matched")
BASELINES = tuple(method for method in METHODS if method not in PROPOSED)
FADING = {"kind": "diagonal_fading", "rho_c": .35}
CONFIRMATION_SEEDS = (2026112301, 2026112302, 2026112303)
VALIDATION_EVAL_OFFSET = 2_000_003


def read_json(path):
    return json.loads(path.read_text())


def write_text(path, text, *, frozen=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_text() != text:
            raise ValueError(f"Existing snapshot differs: {path}")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        if frozen:
            try:
                path.hardlink_to(temporary)
            except FileExistsError:
                if path.read_text() != text:
                    raise ValueError(f"Concurrent snapshot differs: {path}") from None
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, *, frozen=False):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n", frozen=frozen)


def manifest_path(output, dataset, phase):
    return output / "manifests" / f"{dataset}-{phase}.jsonl"


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate task IDs: {path}")
    return rows


def write_manifest(path, rows):
    write_text(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), frozen=True)


def historical_configs():
    index = read_json(DATA / "provenance.json")
    archive_path = historical_table.REPORTS
    digest = historical_table.file_sha256(archive_path)
    if digest != index["files"]["data/performance/selected_trials.tar.gz"]["sha256"]:
        raise ValueError("The chosen performance trial archive differs from its inventory")
    with tarfile.open(archive_path) as archive:
        for cohort in index["cohorts"]:
            if cohort["channel"] != "Fading":
                continue
            dataset = cohort["dataset"]
            seeds = cohort["confirmation_seeds"][:2]
            if len(seeds) < 2 or set(seeds) & set(CONFIRMATION_SEEDS):
                raise ValueError(f"Expected two search seeds disjoint from confirmation seeds for {dataset}")
            for method in METHODS:
                for seed in seeds:
                    key = f"{cohort['archive_directory']}/{method}/{seed}"
                    config = json.load(archive.extractfile(key + "/task_config.json"))
                    overrides = json.load(archive.extractfile(key + "/test_deployment/overrides.json"))
                    metadata = json.load(archive.extractfile(key + "/metadata.json"))
                    if (config["dataset"], config["m"], config["deployment_channel"], config["seed"]) != (
                        dataset, 2000, FADING, seed
                    ):
                        raise ValueError(f"Unexpected archived training conditions for {key}")
                    yield method, config, overrides, {"archive_key": key, "revision": metadata["git_sha"]}


def candidate_variants(method, config):
    yield "archive", config
    # Give each method the same number of candidates. Preserve the T1 penalty
    # ratio and keep the perfect-training channel coefficient at zero.
    if method in PROPOSED:
        for label, scale in (("kl-low", .1), ("kl-high", 4.)):
            yield label, {
                **config,
                "eta_channel": config["eta_channel"] * scale,
                "eta_data": config["eta_data"] * scale,
            }
    elif method == "L2-Matched":
        for label, scale in (("l2-low", .1), ("l2-high", 10.)):
            yield label, {**config, "l2": config["l2"] * scale}
    else:
        for label, scale in (("lr-low", .5), ("lr-high", 1.5)):
            yield label, {**config, "lr": config["lr"] * scale}


def task_entry(phase, method, candidate, config, overrides, source):
    return dict(
        task_id=f"{config['dataset']}-{method}-{candidate}-s{config['seed']}",
        phase=phase,
        method=method,
        candidate=candidate,
        config=config,
        evaluation=overrides,
        source=source,
    )


def plan(output):
    groups = defaultdict(list)
    for method, config, overrides, source in historical_configs():
        config = {**config, "transmit_unit_power": True}
        overrides = {**overrides, "eval_seed": config["eval_seed"] + VALIDATION_EVAL_OFFSET}
        for candidate, values in candidate_variants(method, config):
            groups[config["dataset"]].append(task_entry("search", method, candidate, values, overrides, source))
    for dataset in DATASETS:
        path = manifest_path(output, dataset, "search")
        rows = groups[dataset]
        write_manifest(path, rows)
        print(f"{path} ({len(rows)} tasks)")


def task_dir(output, entry):
    return output / entry["phase"] / entry["task_id"]


def split_for(entry):
    return {"search": "validation", "confirmation": "test"}[entry["phase"]]


def evaluate_result(output, entry):
    directory = task_dir(output, entry)
    split = split_for(entry)
    path = directory / split / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} evaluation for {entry['task_id']}: {path}")
    if read_json(directory / "manifest_entry.json") != entry:
        raise ValueError(f"Manifest entry differs for {entry['task_id']}")
    report = read_json(path)
    evaluated = report["evaluation_config"]
    mc = report["heldout_gibbs_mc"]
    expected = {**entry["config"], **entry["evaluation"]}
    for key in expected.keys() - {"device", "threads"}:
        if evaluated[key] != expected[key]:
            raise ValueError(f"Evaluation config differs for {entry['task_id']}: {key}")
    replicates = mc["replicate_statistics"]
    if (report["split"], report["risk_scope"], mc["channel_source"], mc["draws"], len(replicates)) != (
        split, "deployment", "deployment", expected["eval_draws"], expected["eval_draws"]
    ):
        raise ValueError(f"Unexpected evaluation scope for {entry['task_id']}")
    for metric in ("accuracy", "cross_entropy", "feature_power_per_coordinate"):
        if not math.isclose(mc[metric], statistics.fmean(r[metric] for r in replicates), abs_tol=1e-12):
            raise ValueError(f"Invalid replicate mean for {entry['task_id']}: {metric}")
    if not math.isclose(mc["feature_power_per_coordinate"], 1., abs_tol=1e-5):
        raise ValueError(f"Unexpected transmitted power for {entry['task_id']}: {mc['feature_power_per_coordinate']}")
    return report


def run_task(output, manifest, index, device, threads):
    rows = read_manifest(manifest)
    if not 0 <= index < len(rows):
        raise ValueError(f"Task index {index} outside manifest of length {len(rows)}")
    entry = rows[index]
    directory = task_dir(output, entry)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(directory / "manifest_entry.json", entry, frozen=True)
        runtime = dict(device=device, threads=threads)
        config_path = directory / "task_config.json"
        write_json(config_path, {**entry["config"], **runtime}, frozen=True)
        evaluation_dir = directory / split_for(entry)
        overrides_path = evaluation_dir / "overrides.json"
        write_json(overrides_path, {**entry["evaluation"], **runtime}, frozen=True)
        command = [sys.executable, "-m", "fading_unit_power.trainer.run_experiment"]
        metadata_path = directory / "metadata.json"
        training_completed = (directory / "weights.pt").is_file() and metadata_path.is_file() and (
            "total_seconds" in read_json(metadata_path)
        )
        if not training_completed:
            subprocess.run(command + ["--mode", "train", "--config", str(config_path),
                                      "--output", str(directory)], check=True)
        if not (evaluation_dir / "evaluation.json").is_file():
            subprocess.run(command + ["--mode", "evaluate", "--config", str(overrides_path),
                                      "--checkpoint", str(directory), "--output", str(evaluation_dir),
                                      "--split", split_for(entry), "--risk-scope", "deployment"], check=True)
        evaluate_result(output, entry)


def select(output, dataset):
    path = manifest_path(output, dataset, "search")
    rows = read_manifest(path)
    validation = defaultdict(list)
    for entry in rows:
        if entry["phase"] != "search" or entry["config"]["dataset"] != dataset:
            raise ValueError(f"Unexpected search entry: {entry['task_id']}")
        score = evaluate_result(output, entry)["heldout_gibbs_mc"]
        validation[entry["method"], entry["candidate"]].append((entry, score))

    chosen, scores = {}, {}
    search_seeds = {entry["config"]["seed"] for entry in rows}
    if len(search_seeds) != 2 or search_seeds & set(CONFIRMATION_SEEDS):
        raise ValueError(f"Unexpected search seeds for {dataset}")
    for method in METHODS:
        by_candidate = {name: items for (label, name), items in validation.items() if label == method}
        if len(by_candidate) != 3 or any(len(items) != 2 or {row["config"]["seed"] for row, _ in items} != search_seeds
                                         for items in by_candidate.values()):
            raise ValueError(f"Incomplete or unpaired search for {dataset}/{method}")
        scores[method] = {
            name: dict(mean_accuracy=statistics.fmean(score["accuracy"] for _, score in items),
                       mean_cross_entropy=statistics.fmean(score["cross_entropy"] for _, score in items),
                       accuracies_by_seed={row["config"]["seed"]: score["accuracy"] for row, score in items})
            for name, items in by_candidate.items()
        }
        winner = min(scores[method], key=lambda name: (-scores[method][name]["mean_accuracy"],
                                                      scores[method][name]["mean_cross_entropy"], name))
        chosen[method] = by_candidate[winner][0][0]

    confirmation_rows = []
    for method, winner in chosen.items():
        source_config = winner["config"]
        for seed in CONFIRMATION_SEEDS:
            shift = seed - source_config["seed"]
            config = {**source_config, "seed": seed, "channel_seed": source_config["channel_seed"] + shift,
                      "eval_seed": source_config["eval_seed"] + shift}
            overrides = {**winner["evaluation"], "eval_seed": config["eval_seed"] + 1_000_003}
            source = {**winner["source"], "validation_task": winner["task_id"]}
            confirmation_rows.append(task_entry("confirmation", method, winner["candidate"], config, overrides, source))

    best_proposed = max(scores[method][chosen[method]["candidate"]]["mean_accuracy"] for method in PROPOSED)
    best_baseline = max(scores[method][chosen[method]["candidate"]]["mean_accuracy"] for method in BASELINES)
    selection = dict(dataset=dataset, search_manifest_sha256=historical_table.file_sha256(path),
                     confirmation_seeds=CONFIRMATION_SEEDS, candidates=scores,
                     chosen={method: row["candidate"] for method, row in chosen.items()},
                     validation_gap_pp=(best_proposed - best_baseline) * 100)
    write_json(output / "manifests" / f"{dataset}-selection.json", selection, frozen=True)
    confirmation_path = manifest_path(output, dataset, "confirmation")
    write_manifest(confirmation_path, confirmation_rows)
    print(f"{confirmation_path} ({len(confirmation_rows)} tasks), validation gap {selection['validation_gap_pp']:+.3f} pp")


def comparison(rows):
    accuracies = {method: {row["seed"]: row["test_accuracy_pct"] for row in rows if row["method"] == method}
                  for method in METHODS}
    seed_sets = {tuple(sorted(values)) for values in accuracies.values()}
    if len(seed_sets) != 1 or not next(iter(seed_sets)):
        raise ValueError("Expected complete paired test results")
    stats = {method: dict(mean=statistics.fmean(values.values()), sd=statistics.stdev(values.values()))
             for method, values in accuracies.items()}
    proposed = max(PROPOSED, key=lambda method: stats[method]["mean"])
    baseline = max(BASELINES, key=lambda method: stats[method]["mean"])
    gaps = {seed: accuracy - accuracies[baseline][seed] for seed, accuracy in accuracies[proposed].items()}
    return dict(methods=stats, best_proposed=proposed, best_baseline=baseline,
                gap_pp=statistics.fmean(gaps.values()), gap_sd_pp=statistics.stdev(gaps.values()), gaps_by_seed_pp=gaps)


def summarize(output):
    previous = historical_table.seed_observations(read_json(DATA / "provenance.json"))
    rows = [row for row in previous if row["channel"] == "Erasure"]
    summary = {}
    for dataset in DATASETS:
        entries = read_manifest(manifest_path(output, dataset, "confirmation"))
        keys = {(entry["method"], entry["config"]["seed"]) for entry in entries}
        if len(entries) != len(keys) or keys != {(method, seed) for method in METHODS for seed in CONFIRMATION_SEEDS}:
            raise ValueError(f"Incomplete confirmation manifest for {dataset}")
        normalized = []
        for entry in entries:
            if entry["phase"] != "confirmation" or entry["config"]["dataset"] != dataset:
                raise ValueError(f"Unexpected confirmation entry: {entry['task_id']}")
            report = evaluate_result(output, entry)
            mc = report["heldout_gibbs_mc"]
            directory = task_dir(output, entry)
            normalized.append(dict(
                dataset=dataset, channel="Fading", method=entry["method"], seed=entry["config"]["seed"], m=2000,
                test_accuracy_pct=mc["accuracy"] * 100, draws=mc["draws"], examples=mc["examples"],
                task_id=entry["task_id"], training_revision=read_json(directory / "metadata.json")["git_sha"],
                archive_key=str(directory.relative_to(output)), weights_sha256=report["checkpoint_source_fingerprints"]["weights"]["sha256"],
            ))
        rows.extend(normalized)
        old_rows = [row for row in previous if row["dataset"] == dataset and row["channel"] == "Fading"]
        summary[dataset] = comparison(normalized)
        summary[dataset]["historical_fading_gap_pp"] = comparison(old_rows)["gap_pp"]
        print(f"{dataset}: {summary[dataset]['best_proposed']} vs {summary[dataset]['best_baseline']}, "
              f"{summary[dataset]['gap_pp']:+.3f} pp")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(output / "deployment_by_seed.csv", stream.getvalue())
    table = historical_table.render_table(rows).replace(
        "at channel-set size $m=2000$.",
        r"at channel-set size $m=2000$. Fading uses unit transmit power per channel coordinate.",
    )
    write_text(output / "performance_table.tex", table)
    write_json(output / "performance_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("confirmation-count")
    for name in ("plan", "run", "select", "summarize"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    run = commands.choices["run"]
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    commands.choices["select"].add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    if args.command == "confirmation-count":
        print(len(METHODS) * len(CONFIRMATION_SEEDS))
        return
    output = args.output.resolve()
    if args.command == "plan":
        plan(output)
    elif args.command == "run":
        index = args.index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_task(output, args.manifest, index, args.device, args.threads)
    elif args.command == "select":
        select(output, args.dataset)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
