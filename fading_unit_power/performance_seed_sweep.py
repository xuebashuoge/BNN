"""Explore validation-tuned training seeds using existing trained checkpoints."""

import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys

from . import performance
from . import performance_regularization_sweep as regularization
from . import performance_sweep as sweep
from .performance_sweep_report import file_record


TEST_DRAWS = 128
TEST_EVAL_OFFSET = 2_000_007
AUDIT_EVAL_OFFSET = 3_000_019
AUDIT_POPULATION = dict(moons_eval_data_seed=261125901, moons_eval_size=30000)
CHECKPOINT_FILES = {"weights": "weights.pt", "metadata": "metadata.json"}


def selection_path(output, dataset):
    return output / "manifests" / f"{dataset}-seed-selection.json"


def fingerprints(checkpoint):
    return {kind: file_record(checkpoint / name) for kind, name in CHECKPOINT_FILES.items()}


def rank(trial):
    return -trial["accuracy"], trial["cross_entropy"], trial["task_id"]


def initial_scores_path(source, dataset):
    return source / "manifests" / f"{dataset}-search-scores.json"


def validation_design(source, dataset, manifest, source_phase, bank_size, entry_count):
    manifest_hash = sweep.sha256(manifest)
    if source_phase == "refine":
        return sweep.BANK_SIZES, sweep.REFINE_SEEDS, None, dict(refine_manifest_sha256=manifest_hash)
    if source_phase != "search":
        raise ValueError(f"Unsupported validation phase: {source_phase}")

    planned_path = regularization.plan_path(source, dataset)
    if not planned_path.is_file() and bank_size is not None:
        scores_path = initial_scores_path(source, dataset)
        scores = performance.read_json(scores_path)
        if (scores["dataset"], scores["manifest_sha256"]) != (dataset, manifest_hash):
            raise ValueError(f"Initial search scores differ from the manifest for {dataset}")
        profiles = {
            (m, method): {f"sweep-search-m{m}-{name}" for name in sweep.SEARCH_CANDIDATES}
            for m in sweep.BANK_SIZES for method in sweep.METHODS
        }
        provenance = dict(source_phase="search", search_origin="initial", search_manifest_sha256=manifest_hash,
                          search_scores_sha256=sweep.sha256(scores_path))
        return sweep.BANK_SIZES, sweep.SEARCH_SEEDS, profiles, provenance

    planned = performance.read_json(planned_path)
    m = planned["bank_size"]
    fixed_source = "fixed_bank_size" in planned
    validation_seeds, _, profile_count = regularization.search_design(planned)
    if (planned["dataset"] != dataset or
            (fixed_source and planned["fixed_bank_size"] != m) or
            (not fixed_source and m not in sweep.BANK_SIZES) or
            entry_count != len(sweep.METHODS) * profile_count * len(validation_seeds) or
            planned["search_manifest_sha256"] != manifest_hash or
            tuple(planned["validation_seeds"]) != validation_seeds):
        raise ValueError(f"Regularization search plan differs from the manifest for {dataset}")
    profiles = {method: set(names) for method, names in planned["profiles"].items()}
    if profiles.keys() != set(sweep.METHODS) or any(
            len(names) != profile_count or len(profiles[method]) != len(names)
            for method, names in planned["profiles"].items()):
        raise ValueError(f"Expected {profile_count} profiles per method for {dataset}")
    provenance = dict(source_phase="search", search_manifest_sha256=manifest_hash,
                      search_plan_sha256=sweep.sha256(planned_path))
    if fixed_source:
        provenance["fixed_bank_size"] = m
    if planned.get("schedule_panel"):
        provenance["search_origin"] = "schedule-panel"
    if planned.get("strength_panel"):
        provenance["search_origin"] = "strength-panel"
        provenance["validation_evaluation"] = regularization.strength_validation_evaluation()
    return (m,), validation_seeds, {(m, method): names for method, names in profiles.items()}, provenance


def verify_initial_scores(source, dataset, candidates, banks, seeds):
    recorded = performance.read_json(initial_scores_path(source, dataset))
    scores, winners = {}, {}
    for m in banks:
        scores[str(m)], winners[str(m)] = {}, {}
        for method in sweep.METHODS:
            method_scores = {}
            for name in candidates[m, seeds[0], method]:
                trials = {str(seed): candidates[m, seed, method][name] for seed in seeds}
                method_scores[name] = dict(
                    mean_accuracy=statistics.fmean(trial["accuracy"] for trial in trials.values()),
                    mean_cross_entropy=statistics.fmean(trial["cross_entropy"] for trial in trials.values()),
                    accuracy_by_seed={seed: trial["accuracy"] for seed, trial in trials.items()},
                )
            scores[str(m)][method] = method_scores
            winners[str(m)][method] = sweep.best(method_scores)
    if (recorded["bank_sizes"], recorded["winners"]) != (scores, winners):
        raise ValueError(f"Initial search scores differ from the validation results for {dataset}")


def plan(output, source, dataset, *, source_phase="refine", bank_size=None, audit_population=False):
    if source == output:
        raise ValueError("Use a separate output directory for seed tuning")
    manifest = performance.manifest_path(source, dataset, source_phase)
    entries = performance.read_manifest(manifest)
    banks, seeds, profiles, provenance = validation_design(source, dataset, manifest, source_phase, bank_size, len(entries))
    if bank_size is not None:
        if bank_size not in banks:
            raise ValueError(f"Bank size {bank_size} is unavailable for {dataset}; available sizes are {banks}")
        provenance["fixed_bank_size"] = bank_size
    initial_search = provenance.get("search_origin") == "initial"
    validation_evaluation = provenance.get("validation_evaluation", {})
    search_draws = validation_evaluation.get("eval_draws", (sweep if initial_search else regularization).EVAL_DRAWS["search"])
    if audit_population and provenance.get("search_origin") != "strength-panel":
        raise ValueError("The independent IID audit requires a Two moons strength-panel selection")
    by_id = {entry["task_id"]: entry for entry in entries}
    candidates = defaultdict(dict)
    profile_configs = {}
    pairing = {}
    for entry in entries:
        config = entry["config"]
        m, seed, method = config["m"], config["seed"], entry["method"]
        if (entry["phase"], config["dataset"]) != (source_phase, dataset):
            raise ValueError(f"Unexpected {source_phase} entry: {entry['task_id']}")
        if profiles is not None:
            if (seed not in seeds or entry["candidate"] not in profiles.get((m, method), ()) or
                    config["method"] != sweep.METHOD_OBJECTIVE[method] or
                    entry["evaluation"]["eval_draws"] != search_draws or
                    any(entry["evaluation"].get(key) != value for key, value in validation_evaluation.items())):
                raise ValueError(f"Search entry differs from its validation design: {entry['task_id']}")
            settings = {field: value for field, value in config.items() if field not in regularization.SEED_FIELDS}
            if settings != profile_configs.setdefault((m, method, entry["candidate"]), settings):
                raise ValueError(f"Unpaired search profile configuration: {entry['task_id']}")
        report = performance.evaluate_result(source, entry)
        score = report["heldout_gibbs_mc"]
        if score["examples"] != validation_evaluation.get("moons_eval_size", score["examples"]):
            raise ValueError(f"Validation report has the wrong IID sample size: {entry['task_id']}")
        settings = (config["data_seed"], config["channel_seed"], entry["evaluation"]["eval_seed"],
                    score["draws"], score["examples"])
        if settings != pairing.setdefault(seed, settings):
            raise ValueError(f"Unpaired {source_phase} validation for {dataset}/seed {seed}")
        group = candidates[m, seed, method]
        if entry["candidate"] in group:
            raise ValueError(f"Repeated candidate for {dataset}/{m}/{seed}/{method}")
        group[entry["candidate"]] = dict(task_id=entry["task_id"], accuracy=score["accuracy"],
                                         cross_entropy=score["cross_entropy"])

    expected = {(m, seed, method) for m in banks for seed in seeds for method in sweep.METHODS}
    if profiles is None:
        if candidates.keys() != expected or any(len(group) != 2 for group in candidates.values()):
            raise ValueError(f"Expected two refinement candidates per bank size, seed and method for {dataset}")
    elif candidates.keys() != expected or any(
            group.keys() != profiles[m, method] for (m, _, method), group in candidates.items()):
        raise ValueError(f"Incomplete search candidates per bank size, seed and method for {dataset}")
    if initial_search:
        verify_initial_scores(source, dataset, candidates, banks, seeds)

    leaderboard = []
    for m in banks if bank_size is None else (bank_size,):
        for seed in seeds:
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
    if audit_population:
        evaluation.update(AUDIT_POPULATION, eval_seed=selected["seed"] + AUDIT_EVAL_OFFSET)
        provenance["audit_population"] = AUDIT_POPULATION
    tasks = []
    for method in sweep.METHODS:
        entry = by_id[selected["winners"][method]["task_id"]]
        checkpoint = performance.task_dir(source, entry).resolve()
        validation_report = performance.evaluate_result(source, entry)
        expected_fingerprints = fingerprints(checkpoint)
        if expected_fingerprints != validation_report["checkpoint_source_fingerprints"]:
            raise ValueError(f"Weights or metadata changed after {source_phase} validation: {entry['task_id']}")
        validation_path = checkpoint / "validation" / "evaluation.json"
        tasks.append(dict(
            entry={**entry, "phase": "confirmation", "evaluation": evaluation},
            checkpoint=str(checkpoint), checkpoint_fingerprints=expected_fingerprints,
            validation_report_sha256=sweep.sha256(validation_path), selected_epoch=validation_report["selected_epoch"],
        ))

    path = selection_path(output, dataset)
    performance.write_json(path, dict(dataset=dataset, source=str(source), **provenance,
                                      selected=selected, leaderboard=leaderboard, tasks=tasks), frozen=True)
    print(f"{path}: m={selected['m']}, seed={selected['seed']}, validation gap {selected['validation_gap_pp']:+.3f} pp")


def test_report(output, task):
    entry = task["entry"]
    report = performance.evaluate_result(output, entry)
    if (report["checkpoint"], report["checkpoint_source_fingerprints"], report["selected_epoch"]) != (
            task["checkpoint"], task["checkpoint_fingerprints"], task["selected_epoch"]):
        raise ValueError(f"Test report differs from validation-selected checkpoint: {entry['task_id']}")
    return report


def validate_bank(selection, dataset):
    bank_size = selection["selected"]["m"]
    fixed_bank_size = selection.get("fixed_bank_size")
    if fixed_bank_size is not None and (bank_size != fixed_bank_size or any(
            row["m"] != fixed_bank_size for row in selection["leaderboard"])):
        raise ValueError(f"Selection differs from fixed bank size {fixed_bank_size} for {dataset}")
    if any(task["entry"]["config"]["m"] != bank_size or
           task["entry"]["evaluation"].get("m", bank_size) != bank_size for task in selection["tasks"]):
        raise ValueError(f"Selected tasks differ from bank size {bank_size} for {dataset}")
    if "audit_population" in selection:
        expected = dict(eval_draws=TEST_DRAWS, eval_seed=selection["selected"]["seed"] + AUDIT_EVAL_OFFSET,
                        **AUDIT_POPULATION)
        if (dataset != "moons" or selection.get("search_origin") != "strength-panel" or
                selection["audit_population"] != AUDIT_POPULATION or
                any(task["entry"]["evaluation"] != expected for task in selection["tasks"])):
            raise ValueError(f"Invalid IID audit population for {dataset}")


def run(output, dataset, index):
    selection = performance.read_json(selection_path(output, dataset))
    validate_bank(selection, dataset)
    task = selection["tasks"][index]
    entry = task["entry"]
    checkpoint = Path(task["checkpoint"])
    if fingerprints(checkpoint) != task["checkpoint_fingerprints"]:
        raise ValueError(f"Checkpoint changed after seed selection: {entry['task_id']}")
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
    validation_source = None
    fixed_bank_size = None
    audit_population = None
    for current in (dataset,) if dataset else sweep.DATASETS:
        path = selection_path(output, current)
        selection = performance.read_json(path)
        validate_bank(selection, current)
        bank = selection.get("fixed_bank_size")
        if results and bank != fixed_bank_size:
            raise ValueError(f"Inconsistent fixed bank size for {current}")
        fixed_bank_size = bank
        population = selection.get("audit_population")
        if results and population != audit_population:
            raise ValueError(f"Inconsistent evaluation population for {current}")
        audit_population = population
        phase = selection.get("source_phase", "refine")
        origin = selection.get("search_origin")
        source = {("refine", None): "refinement", ("search", None): "regularization",
                  ("search", "initial"): "initial search",
                  ("search", "schedule-panel"): "schedule panel",
                  ("search", "strength-panel"): "shrinkage panel"}.get((phase, origin))
        if source is None or (validation_source is not None and source != validation_source):
            raise ValueError(f"Inconsistent validation source phase for {current}")
        validation_source = source
        selected = selection["selected"]
        methods = {}
        for task in selection["tasks"]:
            entry = task["entry"]
            method = entry["method"]
            mc = test_report(output, task)["heldout_gibbs_mc"]
            if audit_population is not None and mc["examples"] != audit_population["moons_eval_size"]:
                raise ValueError(f"Unexpected IID audit sample size for {entry['task_id']}")
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
    restriction = {} if fixed_bank_size is None else dict(fixed_bank_size=fixed_bank_size)
    if audit_population is not None:
        restriction["audit_population"] = audit_population
    performance.write_json(output / f"seed_sweep_summary{suffix}.json", dict(
        analysis=f"Exploratory single training seed selected on {validation_source} validation",
        all_reported_datasets_have_validation_selected_proposed_wins=all(
            row["beats_both_baselines"] for row in results.values()),
        datasets=results, **restriction,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dataset", choices=sweep.DATASETS, required=name != "summarize")
    commands.choices["plan"].add_argument("--source", type=Path, required=True)
    commands.choices["plan"].add_argument("--source-phase", choices=("refine", "search"), default="refine",
                                         help="Use refinement or search validation")
    commands.choices["plan"].add_argument("--bank-size", type=int, help="Use one fixed channel bank size")
    commands.choices["plan"].add_argument("--audit-population", action="store_true",
                                         help="Evaluate the same Two moons selection on an independent IID test sample")
    commands.choices["run"].add_argument("--index", type=int, choices=range(len(sweep.METHODS)), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "plan":
        plan(output, args.source.resolve(), args.dataset, source_phase=args.source_phase, bank_size=args.bank_size,
             audit_population=args.audit_population)
    elif args.command == "run":
        run(output, args.dataset, args.index)
    else:
        summarize(output, args.dataset)


if __name__ == "__main__":
    main()
