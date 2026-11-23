"""Explore validation-tuned training seeds using the existing refinement weights."""

import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from . import performance
from . import performance_sweep as sweep
from .performance_sweep_report import file_record


TEST_DRAWS = 128
TEST_EVAL_OFFSET = 2_000_007
CHECKPOINT_FILES = {"weights": "weights.pt", "metadata": "metadata.json"}


def selection_path(output, dataset):
    return output / "manifests" / f"{dataset}-seed-selection.json"


def fingerprints(checkpoint):
    return {kind: file_record(checkpoint / name) for kind, name in CHECKPOINT_FILES.items()}


def rank(trial):
    return -trial["accuracy"], trial["cross_entropy"], trial["task_id"]


def plan(output, source, dataset):
    if source == output:
        raise ValueError("Use a separate output directory for seed tuning")
    manifest = performance.manifest_path(source, dataset, "refine")
    entries = performance.read_manifest(manifest)
    by_id = {entry["task_id"]: entry for entry in entries}
    candidates = defaultdict(dict)
    pairing = {}
    for entry in entries:
        config = entry["config"]
        m, seed, method = config["m"], config["seed"], entry["method"]
        if (entry["phase"], config["dataset"]) != ("refine", dataset):
            raise ValueError(f"Unexpected refinement entry: {entry['task_id']}")
        report = performance.evaluate_result(source, entry)
        score = report["heldout_gibbs_mc"]
        settings = (config["data_seed"], config["channel_seed"], entry["evaluation"]["eval_seed"],
                    score["draws"], score["examples"])
        if settings != pairing.setdefault(seed, settings):
            raise ValueError(f"Unpaired refinement validation for {dataset}/seed {seed}")
        group = candidates[m, seed, method]
        if entry["candidate"] in group:
            raise ValueError(f"Repeated candidate for {dataset}/{m}/{seed}/{method}")
        group[entry["candidate"]] = dict(task_id=entry["task_id"], accuracy=score["accuracy"],
                                         cross_entropy=score["cross_entropy"])

    expected = {(m, seed, method) for m in sweep.BANK_SIZES for seed in sweep.REFINE_SEEDS for method in sweep.METHODS}
    if candidates.keys() != expected or any(len(group) != 2 for group in candidates.values()):
        raise ValueError(f"Expected two refinement candidates per bank size, seed and method for {dataset}")

    leaderboard = []
    for m in sweep.BANK_SIZES:
        for seed in sweep.REFINE_SEEDS:
            trials = {method: candidates[m, seed, method] for method in sweep.METHODS}
            winners = {method: min(group.values(), key=rank) for method, group in trials.items()}
            proposed = min(sweep.PROPOSED, key=lambda method: rank(winners[method]))
            baseline = min(sweep.BASELINES, key=lambda method: rank(winners[method]))
            gap = winners[proposed]["accuracy"] - winners[baseline]["accuracy"]
            leaderboard.append(dict(m=m, seed=seed, proposed=proposed, baseline=baseline,
                                    validation_gap_pp=100 * gap, winners=winners, trials=trials))

    selected = min(leaderboard, key=lambda row: (
        -row["validation_gap_pp"], *rank(row["winners"][row["proposed"]])[:2], row["m"], row["seed"]))
    evaluation = dict(eval_draws=TEST_DRAWS, eval_seed=selected["seed"] + TEST_EVAL_OFFSET)
    tasks = []
    for method in sweep.METHODS:
        entry = by_id[selected["winners"][method]["task_id"]]
        checkpoint = performance.task_dir(source, entry).resolve()
        validation_report = performance.evaluate_result(source, entry)
        expected_fingerprints = fingerprints(checkpoint)
        if expected_fingerprints != validation_report["checkpoint_source_fingerprints"]:
            raise ValueError(f"Refinement weights or metadata changed after validation: {entry['task_id']}")
        validation_path = checkpoint / "validation" / "evaluation.json"
        tasks.append(dict(
            entry={**entry, "phase": "confirmation", "evaluation": evaluation},
            checkpoint=str(checkpoint), checkpoint_fingerprints=expected_fingerprints,
            validation_report_sha256=sweep.sha256(validation_path), selected_epoch=validation_report["selected_epoch"],
        ))

    path = selection_path(output, dataset)
    performance.write_json(path, dict(dataset=dataset, source=str(source), refine_manifest_sha256=sweep.sha256(manifest),
                                      selected=selected, leaderboard=leaderboard, tasks=tasks), frozen=True)
    print(f"{path}: m={selected['m']}, seed={selected['seed']}, validation gap {selected['validation_gap_pp']:+.3f} pp")


def test_report(output, task):
    entry = task["entry"]
    report = performance.evaluate_result(output, entry)
    if (report["checkpoint"], report["checkpoint_source_fingerprints"], report["selected_epoch"]) != (
            task["checkpoint"], task["checkpoint_fingerprints"], task["selected_epoch"]):
        raise ValueError(f"Test report differs from validation-selected checkpoint: {entry['task_id']}")
    return report


def run(output, dataset, index):
    selection = performance.read_json(selection_path(output, dataset))
    task = selection["tasks"][index]
    entry = task["entry"]
    checkpoint = Path(task["checkpoint"])
    if fingerprints(checkpoint) != task["checkpoint_fingerprints"]:
        raise ValueError(f"Refinement checkpoint changed after seed selection: {entry['task_id']}")
    directory = performance.task_dir(output, entry)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        performance.write_json(directory / "manifest_entry.json", entry, frozen=True)
        test_dir = directory / "test"
        overrides = test_dir / "overrides.json"
        runtime = dict(device="cpu", threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
        performance.write_json(overrides, {**entry["evaluation"], **runtime}, frozen=True)
        if not (test_dir / "evaluation.json").is_file():
            subprocess.run([sys.executable, "-m", "fading_unit_power.trainer.run_experiment", "--mode", "evaluate",
                            "--config", str(overrides), "--checkpoint", str(checkpoint), "--output", str(test_dir),
                            "--split", "test", "--risk-scope", "deployment"], check=True)
        test_report(output, task)


def summarize(output, dataset=None):
    results = {}
    for current in (dataset,) if dataset else sweep.DATASETS:
        path = selection_path(output, current)
        selection = performance.read_json(path)
        selected = selection["selected"]
        methods = {}
        for task in selection["tasks"]:
            entry = task["entry"]
            method = entry["method"]
            mc = test_report(output, task)["heldout_gibbs_mc"]
            methods[method] = dict(test_accuracy_pct=100 * mc["accuracy"], test_cross_entropy=mc["cross_entropy"],
                                   validation_accuracy_pct=100 * selected["winners"][method]["accuracy"],
                                   test_draws=mc["draws"], test_examples=mc["examples"],
                                   task_id=entry["task_id"], selected_epoch=task["selected_epoch"],
                                   checkpoint_fingerprints=task["checkpoint_fingerprints"])
        if tuple(methods) != sweep.METHODS:
            raise ValueError(f"Incomplete ordered seed-tuned comparison for {current}")
        proposed = selected["proposed"]
        accuracy = methods[proposed]["test_accuracy_pct"]
        gaps = {baseline: accuracy - methods[baseline]["test_accuracy_pct"] for baseline in sweep.BASELINES}
        results[current] = dict(bank_size=selected["m"], training_seed=selected["seed"], methods=methods,
                                validation_proposed=proposed, validation_baseline=selected["baseline"],
                                validation_gap_pp=selected["validation_gap_pp"], test_gaps_pp=gaps,
                                beats_both_baselines=min(gaps.values()) > 0, selection_sha256=sweep.sha256(path))
        formatted = ", ".join(f"{baseline} {gap:+.3f} pp" for baseline, gap in gaps.items())
        print(f"{current}: {proposed} vs {formatted}")
    suffix = f"-{dataset}" if dataset else ""
    performance.write_json(output / f"seed_sweep_summary{suffix}.json", dict(
        analysis="Exploratory single training seed selected on refinement validation",
        all_reported_datasets_have_validation_selected_proposed_wins=all(
            row["beats_both_baselines"] for row in results.values()),
        datasets=results,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dataset", choices=sweep.DATASETS, required=name != "summarize")
    commands.choices["plan"].add_argument("--source", type=Path, required=True)
    commands.choices["run"].add_argument("--index", type=int, choices=range(len(sweep.METHODS)), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "plan":
        plan(output, args.source.resolve(), args.dataset)
    elif args.command == "run":
        run(output, args.dataset, args.index)
    else:
        summarize(output, args.dataset)


if __name__ == "__main__":
    main()
