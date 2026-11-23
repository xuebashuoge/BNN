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


def search_groups(rows, dataset):
    groups = defaultdict(lambda: defaultdict(list))
    for entry in rows:
        if entry["phase"] != "search" or entry["config"]["dataset"] != dataset or entry["method"] not in METHODS:
            raise ValueError(f"Unexpected search entry: {entry['task_id']}")
        groups[entry["method"]][entry["candidate"]].append(entry)

    seeds = {entry["config"]["seed"] for entry in rows}
    if len(seeds) != 2 or seeds & set(CONFIRMATION_SEEDS):
        raise ValueError(f"Unexpected search seeds for {dataset}")
    for method in METHODS:
        if not groups[method] or any(len(entries) != 2 or {entry["config"]["seed"] for entry in entries} != seeds
                                     for entries in groups[method].values()):
            raise ValueError(f"Incomplete or unpaired search for {dataset}/{method}")
    return groups


def refinement_variants(method):
    yield "refine-lr-low", {"lr": .25}
    yield "refine-lr-high", {"lr": 2.}
    if method in PROPOSED:
        fields, scales, label = ("eta_channel", "eta_data"), (.25, 4.), "kl"
    elif method == "L2-Matched":
        fields, scales, label = ("l2",), (.25, 4.), "l2"
    elif method == "ERM-Matched":
        fields, scales, label = ("posterior_std",), (1 / 3, 3.), "posterior"
    else:
        fields, scales, label = ("eta_data",), (.25, 4.), "kl"
    for side, scale in zip(("low", "high"), scales):
        yield f"refine-{label}-{side}", dict.fromkeys(fields, scale)


def plan_refinement(source, output, dataset):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Use a separate directory for refinement")
    source_manifest = manifest_path(source, dataset, "search")
    selection_path = source / "manifests" / f"{dataset}-selection.json"
    selection = read_json(selection_path)
    digest = historical_table.file_sha256(source_manifest)
    if (selection["dataset"], selection["search_manifest_sha256"]) != (dataset, digest):
        raise ValueError(f"Selection does not match the primary search manifest for {dataset}")
    groups = search_groups(read_manifest(source_manifest), dataset)
    rows = []
    candidate_names = {}
    for method in METHODS:
        considered = groups[method]
        anchor = considered[selection["chosen"][method]]
        primary_configs = [entry["config"] for entries in considered.values() for entry in entries]
        names = []
        for name, scales in [("primary-winner", {}), *refinement_variants(method)]:
            configs = [{**entry["config"], **{key: entry["config"][key] * scale for key, scale in scales.items()}}
                       for entry in anchor]
            if scales and any(config in primary_configs for config in configs):
                continue
            names.append(name)
            for entry, config in zip(anchor, configs):
                provenance = {**entry["source"], "primary_validation_task": entry["task_id"]}
                rows.append(task_entry("search", method, name, config, entry["evaluation"], provenance))
        if not 3 <= len(names) <= 5:
            raise ValueError(f"Expected two to four new candidates for {dataset}/{method}")
        candidate_names[method] = names

    write_json(output / "manifests" / f"{dataset}-refinement.json", dict(
        dataset=dataset, primary_directory=str(source), primary_search_manifest_sha256=digest,
        primary_selection_sha256=historical_table.file_sha256(selection_path),
        primary_validation_gap_pp=selection["validation_gap_pp"], candidates=candidate_names,
    ), frozen=True)
    path = manifest_path(output, dataset, "search")
    write_manifest(path, rows)
    print(f"{path} ({len(rows)} validation tasks)")


def task_dir(output, entry):
    return output / entry["phase"] / entry["task_id"]


def split_for(entry):
    return {"search": "validation", "refine": "validation", "confirmation": "test"}[entry["phase"]]


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
    groups = search_groups(read_manifest(path), dataset)
    refinement_path = output / "manifests" / f"{dataset}-refinement.json"
    refinement = read_json(refinement_path) if refinement_path.is_file() else None

    chosen, scores = {}, {}
    for method in METHODS:
        candidates = groups[method]
        if refinement is None:
            if ("archive" not in candidates or
                    set(candidates) != {name for name, _ in candidate_variants(method, candidates["archive"][0]["config"])}):
                raise ValueError(f"Unexpected primary candidates for {dataset}/{method}")
        elif list(candidates) != refinement["candidates"][method] or refinement["dataset"] != dataset:
            raise ValueError(f"Unexpected refinement candidates for {dataset}/{method}")
        scores[method] = {}
        for name, entries in candidates.items():
            validation = [evaluate_result(output, row)["heldout_gibbs_mc"] for row in entries]
            scores[method][name] = dict(
                mean_accuracy=statistics.fmean(score["accuracy"] for score in validation),
                mean_cross_entropy=statistics.fmean(score["cross_entropy"] for score in validation),
                accuracies_by_seed={row["config"]["seed"]: score["accuracy"] for row, score in zip(entries, validation)},
            )
        winner = min(scores[method], key=lambda name: (-scores[method][name]["mean_accuracy"],
                                                      scores[method][name]["mean_cross_entropy"], name))
        chosen[method] = candidates[winner][0]

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
    if refinement is not None:
        selection["refinement_plan_sha256"] = historical_table.file_sha256(refinement_path)
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


def confirmation_observations(output, source, dataset):
    entries = read_manifest(manifest_path(source, dataset, "confirmation"))
    keys = {(entry["method"], entry["config"]["seed"]) for entry in entries}
    if len(entries) != len(keys) or keys != {(method, seed) for method in METHODS for seed in CONFIRMATION_SEEDS}:
        raise ValueError(f"Incomplete confirmation manifest for {dataset}")
    for entry in entries:
        if entry["phase"] != "confirmation" or entry["config"]["dataset"] != dataset:
            raise ValueError(f"Unexpected confirmation entry: {entry['task_id']}")
        report = evaluate_result(source, entry)
        mc = report["heldout_gibbs_mc"]
        directory = task_dir(source, entry)
        yield dict(
            dataset=dataset, channel="Fading", method=entry["method"], seed=entry["config"]["seed"], m=2000,
            test_accuracy_pct=mc["accuracy"] * 100, draws=mc["draws"], examples=mc["examples"],
            task_id=entry["task_id"], training_revision=read_json(directory / "metadata.json")["git_sha"],
            archive_key=os.path.relpath(directory, output),
            weights_sha256=report["checkpoint_source_fingerprints"]["weights"]["sha256"],
        )


def summarize(output, only_dataset=None, sources=None):
    sources = sources or {}
    datasets = (only_dataset,) if only_dataset else DATASETS
    if sources.keys() - set(datasets):
        raise ValueError("Every supplied source must be included in the summary")
    previous = historical_table.seed_observations(read_json(DATA / "provenance.json"))
    rows = [row for row in previous if row["channel"] == "Erasure" and row["dataset"] in datasets]
    summary = {}
    for dataset in datasets:
        source = sources.get(dataset, output)
        normalized = list(confirmation_observations(output, source, dataset))
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
    suffix = f"-{only_dataset}" if only_dataset else ""
    write_text(output / f"deployment_by_seed{suffix}.csv", stream.getvalue())
    if only_dataset is None:
        table = historical_table.render_table(rows).replace(
            "at channel-set size $m=2000$.",
            r"at channel-set size $m=2000$. Fading uses unit transmit power per channel coordinate.",
        )
        write_text(output / "performance_table.tex", table)
    write_json(output / f"performance_summary{suffix}.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("confirmation-count")
    for name in ("plan", "plan-refinement", "run", "select", "summarize"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    refinement = commands.choices["plan-refinement"]
    refinement.add_argument("--source", type=Path, required=True, help="Primary performance directory")
    refinement.add_argument("--dataset", choices=DATASETS, required=True)
    run = commands.choices["run"]
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    commands.choices["select"].add_argument("--dataset", choices=DATASETS, required=True)
    summarize_command = commands.choices["summarize"]
    summarize_command.add_argument("--dataset", choices=DATASETS, help="Summarize one completed dataset")
    summarize_command.add_argument("--source", action="append", default=[], metavar="DATASET=RUN_DIR",
                                   help="Use a different confirmation directory for this dataset")
    args = parser.parse_args()
    if args.command == "confirmation-count":
        print(len(METHODS) * len(CONFIRMATION_SEEDS))
        return
    output = args.output.resolve()
    if args.command == "plan":
        plan(output)
    elif args.command == "plan-refinement":
        plan_refinement(args.source, output, args.dataset)
    elif args.command == "run":
        index = args.index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_task(output, args.manifest, index, args.device, args.threads)
    elif args.command == "select":
        select(output, args.dataset)
    else:
        sources = {}
        for setting in args.source:
            dataset, separator, source = setting.partition("=")
            if not separator or not source or dataset not in DATASETS or dataset in sources:
                parser.error(f"Expected one DATASET=RUN_DIR per supplied dataset, got {setting!r}")
            sources[dataset] = Path(source).resolve()
        summarize(output, only_dataset=args.dataset, sources=sources)


if __name__ == "__main__":
    main()
