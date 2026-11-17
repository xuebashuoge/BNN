"""Transparent test-selected paired-seed examples, separate from main means.

Select at most one of the five existing paired seeds per condition using the
saved twenty-draw scores. Freeze every input hash before any new Monte Carlo.
Verify every method at that same seed with exactly 500 fresh deployment-only
draws. No training, new seeds, source writes, or retries after a measured loss.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from .build_bayesian_campaign import CHANNELS, DATASETS, SEEDS
from .build_bayesian_mnist_scale import encoded, write_once
from .run_experiment import bounded_loss_description, normalize_config, save_json


# Fixed before any fresh verification MC; initial20 scores select exploratory examples.
DRAWS, INITIAL_DRAWS, SEED_OFFSET, THREADS = 500, 20, 9000019, 2
TOLERANCE = 1e-12
LABELS = tuple(f"{method}-{regime}" for method in ("ERM", "L2", "T1", "T2")
               for regime in ("Matched", "Perfect"))
PROPOSED = ("T1-Matched", "T2-Matched")
SOURCE_FILES = ("journal/run_experiment.py", "journal/theorem1.py", "journal/cifar_models.py",
                "journal/cifar_search_models.py", "journal/campaign.py",
                "journal/build_bayesian_campaign.py", "journal/build_campaign.py")
CHECKPOINT_FILES = ("metadata.json", "weights.pt", "history.json", "task_config.json", "manifest_entry.json", "train.log")
EXPLORATORY = "Test-selected exploratory paired-seed example; not fresh-seed confirmation or general improvement"


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fingerprint(path):
    checksum, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            size += len(block)
    return dict(sha256=checksum.hexdigest(), bytes=size)


def safe_task_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value, flags=re.ASCII) is None:
        raise ValueError("Task IDs must be safe ASCII path components")
    return value


def disjoint_output(output, protected):
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError("Owned exploratory output must not be a symbolic link")
    output = requested.resolve()
    for value in protected:
        source = Path(value).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Owned exploratory output overlaps a protected assembled or physical source tree")
    return output


def owned_path(root, path):
    """Reject writable aliases, broken links and escapes before any file open."""
    root, path = Path(root).resolve(), Path(path).absolute()
    if not path.is_relative_to(root) or any(part in {".", ".."} for part in path.relative_to(root).parts):
        raise ValueError("Writable exploratory path escapes its owned root")
    current = root
    if current.is_symlink():
        raise ValueError("Owned exploratory cache must not contain symbolic links")
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Owned exploratory cache must not contain symbolic links")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Writable exploratory path resolves outside its owned root")
    return path


def guard_cache(root):
    root = Path(root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("Owned exploratory output must be a real directory")
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Owned exploratory cache must not contain symbolic links")


def owned_write(root, path, content):
    path = owned_path(root, path)
    owned_path(root, path.with_suffix(path.suffix+".tmp"))
    write_once(path, content)


def owned_save(root, path, value):
    path = owned_path(root, path)
    owned_path(root, path.with_suffix(path.suffix+".tmp"))
    save_json(path, value)


def preflight_output(source_root, output):
    """Protect physical source roots outside the assembled alias directory."""
    source_root = Path(source_root).resolve()
    state = json.loads((source_root / "assembly_status.json").read_text())
    audit = json.loads((source_root / "assembly_audit.json").read_text())
    manifest = source_root / "manifests" / "confirm.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(entries) != 320 or len({safe_task_id(item["task_id"]) for item in entries}) != 320:
        raise ValueError("Output preflight requires all 320 safe confirmation identities")
    protected = [source_root, state["main_root"], state["mnist_root"]]
    for case in audit.get("source_map", []):
        checkpoint = Path(case["actual_checkpoint"]).resolve()
        protected.extend((checkpoint, checkpoint.parent, checkpoint.parent.parent))
    for stage in audit.get("source_stages", {}).values():
        protected.extend((stage["resolved"], Path(stage["resolved"]).parent))
    for item in entries:
        checkpoint = (source_root / "confirm" / item["task_id"]).resolve(strict=True)
        protected.extend((checkpoint, checkpoint.parent, checkpoint.parent.parent))
    output = disjoint_output(output, protected)
    guard_cache(output)
    return output


def statistics_of(values):
    values = list(values)
    if not values or not all(finite(value) for value in values):
        raise ValueError("Require all finite declared observations")
    sd = statistics.stdev(values) if len(values) > 1 else 0.
    return dict(n=len(values), mean=statistics.fmean(values), sd=sd,
                standard_error=sd/math.sqrt(len(values)))


def best(scores, labels):
    return min(labels, key=lambda label: (-scores[label], label))


def margin(scores, candidate):
    other = best(scores, [label for label in LABELS if label != candidate])
    value = scores[candidate] - scores[other]
    return dict(candidate=candidate, strongest_other=other, margin=value,
                strictly_best=value > TOLERANCE, ties_best=abs(value) <= TOLERANCE)


def select_examples(rows, all_settings=False):
    """Keep all five seeds and select one common seed across all eight methods."""
    grouped = defaultdict(lambda: defaultdict(dict))
    ids = set()
    for row in rows:
        safe_task_id(row["task_id"])
        group, seed, label = row["group"], row["seed"], row["method_label"]
        if (row["task_id"] in ids or label not in LABELS or seed not in SEEDS["confirm"]
                or row.get("test_status") != "complete" or not finite(row.get("test_accuracy"))
                or not 0 <= row["test_accuracy"] <= 1 or label in grouped[group][seed]):
            raise ValueError("Require distinct complete paired confirmation scores")
        ids.add(row["task_id"])
        grouped[group][seed][label] = row
    if len(rows) != 320 or len(grouped) != 8:
        raise ValueError("Seed examples require the complete 320-case, eight-setting main cohort")
    output = []
    for group, by_seed in sorted(grouped.items()):
        if set(by_seed) != set(SEEDS["confirm"]) or any(set(methods) != set(LABELS) for methods in by_seed.values()):
            raise ValueError("Every setting must retain all five paired seeds and all eight methods")
        search = []
        for seed, methods in sorted(by_seed.items()):
            scores = {label: row["test_accuracy"] for label, row in methods.items()}
            candidate = best(scores, PROPOSED)
            search.append(dict(seed=seed, scores=scores, **margin(scores, candidate)))
        winner = min(search, key=lambda item: (-item["margin"], item["seed"]))
        selected = winner["seed"] if winner["strictly_best"] or all_settings else None
        means = {label: statistics_of(by_seed[seed][label]["test_accuracy"] for seed in SEEDS["confirm"])
                 for label in LABELS}
        output.append(dict(group=group, interpretation=EXPLORATORY, selected_seed=selected,
            selection_status=("selected_existing_paired_seed" if winner["strictly_best"] else
                              "bestavailable_no_initial_win" if all_settings else "no_strictly_winning_paired_seed"),
            initial_best_margin=winner["margin"], initial_strict_win=winner["strictly_best"],
            initial_candidate=winner["candidate"] if selected is not None else None,
            initial_strongest_other=winner["strongest_other"] if selected is not None else None,
            initial_margin=winner["margin"] if selected is not None else None,
            all_five_seed_search=search, original_five_seed_means=means,
            selected_task_ids=([by_seed[selected][label]["task_id"] for label in LABELS] if selected is not None else [])))
    return output


def verify_code(main_revision, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
    source = git("rev-parse", "--verify", "--end-of-options", main_revision + "^{commit}")
    execution = git("rev-parse", "HEAD^{commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use a clean committed checkout for exploratory MC verification")
    blobs = {}
    for filename in SOURCE_FILES:
        expected, actual = git("rev-parse", source+":"+filename), git("rev-parse", execution+":"+filename)
        if expected != actual or git("hash-object", "--", filename) != expected:
            raise ValueError(f"Exploratory evaluator source differs from prescribed main source: {filename}")
        blobs[filename] = expected
    driver = "journal/verify_bayesian_seed_examples.py"
    driver_blob = git("rev-parse", execution+":"+driver)
    if git("hash-object", "--", driver) != driver_blob:
        raise ValueError("Exploratory driver differs from its committed source")
    return dict(main_revision=source, execution_revision=execution, scientific_blobs=blobs, driver_blob=driver_blob)


def recheck(protocol):
    for path, expected in protocol["source_fingerprints"].items():
        if fingerprint(path) != expected:
            raise ValueError(f"Frozen exploratory selection source changed: {path}")
    for task in protocol["all_original_tasks"]:
        safe_task_id(task["task_id"])
        if str(Path(task["logical_checkpoint"]).resolve(strict=True)) != task["checkpoint"]:
            raise ValueError("An assembled checkpoint alias changed after exploratory selection")


def build_protocol(source_root, code, frozen_at, all_settings=False):
    """Rebuild every selection/task field from attested immutable source data."""
    from .assemble_bayesian_results import POLICY
    from .plot_bayesian_campaign import Evidence
    source_root = Path(source_root).resolve()
    files = {}
    def read(path):
        files[str(path.resolve())] = fingerprint(path)
        return json.loads(path.read_text())
    state = read(source_root / "assembly_status.json")
    if (state.get("status") != "complete" or state.get("final_runs") != 2048 or state.get("policy") != POLICY
            or {stage: state.get("stages", {}).get(stage, {}).get("tasks") for stage in
                ("confirm", "m_sweep", "p_train_sweep")} != dict(confirm=320, m_sweep=1056, p_train_sweep=672)):
        raise ValueError("Exploratory selection waits for the complete final assembled cohort")
    manifest = source_root / "manifests" / "confirm.jsonl"
    files[str(manifest.resolve())] = fingerprint(manifest)
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if state["stages"]["confirm"].get("manifest_fingerprint") != fingerprint(manifest):
        raise ValueError("Assembled confirmation manifest differs from its recorded fingerprint")
    rows = read(source_root / "confirm" / "test_results.json")
    original_means = read(source_root / "main_means.json")
    original_outcome = read(source_root / "accuracy_outcome.json")
    from .run_bayesian_campaign import ranking_outcome
    if original_outcome != {**ranking_outcome(source_root / "confirm", manifest=manifest),
                            "policy": POLICY, "accuracy_tolerance": TOLERANCE}:
        raise ValueError("Original mean-ranking outcome differs from complete five-seed replay")
    audit = read(source_root / "assembly_audit.json")
    assembly_code = audit.get("code_sources", {})
    if (audit.get("policy") != POLICY or state.get("code_sources") != assembly_code
            or assembly_code.get("main_revision") != code["main_revision"]
            or assembly_code.get("scientific_blobs") != code["scientific_blobs"]
            or audit.get("source_bytes_modified") is not False):
        raise ValueError("Assembly policy or prescribed scientific-source attestation differs")
    source_map = audit.get("source_map", [])
    indexed = {(item["stage"], safe_task_id(item["task_id"])): item for item in source_map}
    counts = {stage: sum(item["stage"] == stage for item in source_map) for stage in ("confirm", "m_sweep", "p_train_sweep")}
    if len(source_map) != 2048 or len(indexed) != 2048 or counts != dict(confirm=320, m_sweep=1056, p_train_sweep=672):
        raise ValueError("Assembly audit must identify all 2048 distinct source checkpoints")
    evidence = Evidence(source_root / "confirm", manifest)
    if evidence.excluded_task_ids or len(evidence.rows) != 320 or len(entries) != 320:
        raise ValueError("Require exact complete final confirmation evidence")
    conditions, tasks = set(), []
    expected = {entry["task_id"]: entry for entry in entries}
    if len(expected) != 320:
        raise ValueError("Duplicate assembled confirmation tasks")
    by_id = {row["task_id"]: row for row in evidence.rows}
    for row in rows:
        safe_task_id(row["task_id"])
        entry = expected[row["task_id"]]
        cfg = normalize_config(entry["config"])
        channel = entry["channel_name"]
        if (cfg["dataset"] not in DATASETS or channel not in CHANNELS
                or cfg["deployment_channel"] != CHANNELS[channel] or entry.get("stage") != "bayes_confirm"
                or cfg["predictor"] != "gaussian" or cfg["posterior_std"] <= 0
                or cfg["channel_seed"] != cfg["seed"]+10000 or cfg["eval_seed"] != cfg["seed"]+20000):
            raise ValueError("Exploratory examples require the unchanged Gaussian main confirmation design")
        conditions.add((cfg["dataset"], channel))
        logical = source_root / "confirm" / row["task_id"]
        checkpoint = logical.resolve(strict=True)
        case = indexed[("confirm", row["task_id"])]
        cohort = "mnist32" if cfg["dataset"] == "mnist" else "main16"
        source_stage = Path(state["mnist_root"] if cohort == "mnist32" else state["main_root"]) / "confirm"
        if (not logical.is_symlink() or case.get("cohort") != cohort
                or Path(case["alias"]) != logical or Path(case["actual_checkpoint"]).resolve() != checkpoint
                or checkpoint != (source_stage / row["task_id"]).resolve()
                or entry.get("assembly_source_cohort") != cohort or row.get("assembly_source_cohort") != cohort):
            raise ValueError("Assembled case cohort or logical/physical checkpoint identity differs from its audit")
        algorithms = dict(ERM="erm", L2="l2", T1="proposed", T2="joint")
        if cfg["method"] != algorithms[row["method_label"].split("-", 1)[0]]:
            raise ValueError("Method label differs from its actual Gaussian training objective")
        for filename in CHECKPOINT_FILES:
            path = checkpoint / filename
            files[str(path)] = fingerprint(path)
            recorded = audit.get("source_files", {}).get(str(path), {})
            if recorded != dict(resolved=str(path.resolve()), **files[str(path)]) or path.is_symlink():
                raise ValueError("Checkpoint scientific bytes differ from the assembly source-file audit")
        metadata = read(checkpoint / "metadata.json")
        expected_revision = assembly_code["mnist_revision"] if cohort == "mnist32" else assembly_code["main_revision"]
        if (normalize_config(metadata["config"]) != cfg or metadata.get("git_dirty") is not False
                or metadata.get("git_sha") != case.get("checkpoint_revision") or metadata.get("git_sha") != expected_revision
                or normalize_config(read(checkpoint / "task_config.json")) != cfg):
            raise ValueError("Actual checkpoint configuration or committed source revision differs from the assembly audit")
        saved_entry = read(checkpoint / "manifest_entry.json")
        if normalize_config(saved_entry["config"]) != cfg or any(saved_entry.get(key) != entry.get(key) for key in
                ("task_id", "group", "method_label", "family", "variant_id", "stage")):
            raise ValueError("Actual saved manifest identity differs from the assembled case")
        exported = source_root / "evaluation_reports" / "confirm" / f"{row['task_id']}.json"
        source_report = checkpoint / "test_full" / "evaluation.json"
        if (Path(row["test_report"]) != exported or Path(case["exported_evaluation"]) != exported
                or Path(case["source_evaluation"]).resolve() != source_report.resolve()
                or exported.is_symlink() or fingerprint(exported) != case["exported_evaluation_fingerprint"]
                or fingerprint(source_report) != case["exported_evaluation_fingerprint"]):
            raise ValueError("Initial numerical report differs from its byte-exact exported/source audit")
        files[str(source_report.resolve())] = fingerprint(source_report)
        full = read(exported)
        heldout = full["heldout_gibbs_mc"]
        if (heldout["draws"] != INITIAL_DRAWS or full["evaluation_config"]["eval_draws"] != INITIAL_DRAWS
                or full["evaluation_config"]["eval_seed"] != cfg["eval_seed"]+1000003
                or full["checkpoint_source_fingerprints"] != {
                    "metadata": files[str(checkpoint / "metadata.json")], "weights": files[str(checkpoint / "weights.pt")]}
                or heldout["accuracy"] != row["test_accuracy"]
                or heldout["cross_entropy"] != row["test_cross_entropy"]):
            raise ValueError("Initial twenty-draw source does not identify the declared frozen checkpoint")
        if case.get("source_fingerprints") != {"metadata": files[str(checkpoint / "metadata.json")],
                "weights": files[str(checkpoint / "weights.pt")], "evaluation": files[str(source_report.resolve())]}:
            raise ValueError("Case-level source fingerprints differ from the assembly audit")
        if by_id[row["task_id"]]["_report"]["heldout_gibbs_mc"] != heldout:
            raise ValueError("Exported and source full test reports disagree")
        for path in evidence.files:
            if path not in files:
                files[path] = fingerprint(path)
        validation = {cfg["eval_seed"]} if cfg["validation_seed_policy"] == "fixed" else {
            cfg["eval_seed"]+epoch*1009 for epoch in range(cfg["epochs"])}
        fresh_seed = cfg["eval_seed"]+SEED_OFFSET
        used = {seed+offset for seed in validation | {full["evaluation_config"]["eval_seed"]} for offset in (0, 1)}
        if {fresh_seed, fresh_seed+1} & used:
            raise ValueError("Fresh exploratory MC overlaps validation or original test streams")
        tasks.append(dict(task_id=row["task_id"], group=row["group"], method_label=row["method_label"],
            seed=row["seed"], config=cfg, checkpoint=str(checkpoint), logical_checkpoint=str(logical),
            initial_report=str(exported.resolve()), initial_report_fingerprint=files[str(exported.resolve())],
            initial_accuracy=row["test_accuracy"], fresh_evaluation_seed=fresh_seed,
            checkpoint_source_fingerprints={"metadata": files[str(checkpoint / "metadata.json")],
                                            "weights": files[str(checkpoint / "weights.pt")]}))
    if conditions != {(dataset, channel) for dataset in DATASETS for channel in CHANNELS}:
        raise ValueError("Retain all four datasets and both deployment laws")
    chosen = select_examples(rows, all_settings)
    mean_rows = {(row["group"], row["method_label"]): row for row in original_means}
    if len(original_means) != 64 or len(mean_rows) != 64:
        raise ValueError("Retain every original five-seed main mean")
    for case in chosen:
        for label in LABELS:
            saved = mean_rows[(case["group"], label)]
            stats = case["original_five_seed_means"][label]
            if (saved.get("n") != 5 or abs(saved["accuracy_mean"]-stats["mean"]) > TOLERANCE
                    or abs(saved["accuracy_sd"]-stats["sd"]) > TOLERANCE
                    or json.loads(saved["seeds_json"]) != list(SEEDS["confirm"])):
                raise ValueError("Original five-seed mean table disagrees with complete main confirmation scores")
    protocol = dict(schema_version=1, interpretation=EXPLORATORY, frozen_at=frozen_at,
        source_root=str(source_root), code_verification=code, initial_draws=INITIAL_DRAWS,
        fresh_draws=DRAWS, fresh_seed_offset=SEED_OFFSET, all_original_seeds=list(SEEDS["confirm"]),
        all_settings=all_settings, selection_mode="all_settings_best_available" if all_settings else "strict_positive_only",
        coverage_policy_timing="Coverage mode is frozen before any fresh500-draw verification; initial20-draw results may inform this exploratory analysis",
        selection=("Every setting: choose the best available existing paired seed, regardless of the sign of its initial20-draw margin; seed-number tiebreak"
                   if all_settings else "Per setting maximize initial20-draw margin of best matched T1/T2 versus strongest of all other seven methods; require strict positive margin; seed-number tiebreak"),
        no_winning_seed=("Still verify the best available paired seed once; label bestavailable_no_initial_win and preserve negative margins/ties"
                         if all_settings else "Report none; no new seed training or retry until a win"),
        all_original_tasks=tasks, original_main_means=original_means, original_mean_rankings=original_outcome,
        conditions=chosen, source_fingerprints=files, source_writes=False, new_training=False)
    recheck(protocol)
    return protocol


def freeze_selection(source_root, output, code, all_settings=False):
    source_root, output = Path(source_root).resolve(), preflight_output(source_root, output)
    selection = owned_path(output, output / "selection_protocol.json")
    previous = json.loads(selection.read_text()) if selection.exists() else None
    protocol = build_protocol(source_root, code, previous["frozen_at"] if previous else now(), all_settings)
    if previous is not None and previous != protocol:
        raise ValueError("Cached exploratory selection protocol differs from complete source replay")
    output.mkdir(parents=True, exist_ok=True)
    owned_write(output, selection, encoded(protocol))
    return protocol


def validate_fresh(report, task):
    safe_task_id(task["task_id"])
    cfg = normalize_config(task["config"])
    if task["fresh_evaluation_seed"] != cfg["eval_seed"]+SEED_OFFSET:
        raise ValueError("Fresh exploratory seed must use the prespecified offset")
    expected = normalize_config({**cfg, "device": "cpu", "threads": THREADS, "download": False,
                                 "eval_draws": DRAWS, "eval_seed": task["fresh_evaluation_seed"]})
    if (report.get("split") != "test" or report.get("risk_scope") != "deployment"
            or str(Path(report["checkpoint"]).resolve(strict=True)) != task["checkpoint"]
            or normalize_config(report["evaluation_config"]) != expected
            or report.get("checkpoint_source_fingerprints") != task["checkpoint_source_fingerprints"]):
        raise ValueError("Fresh result differs from its exact frozen checkpoint or 500-draw request")
    risk = report["heldout_gibbs_mc"]
    if (risk.get("draws") != DRAWS or risk.get("examples") != 10000 or risk.get("channel_source") != "deployment"
            or risk.get("posterior_rule") != "one weight draw fixed across entire loader per replicate"
            or risk.get("predictor", {}).get("stochastic") is not True
            or risk.get("predictor", {}).get("family") != "diagonal_gaussian"
            or risk.get("bounded_loss_definition") != bounded_loss_description(expected)):
        raise ValueError("Fresh deployment verification must use the same Gaussian full-loader Gibbs predictor")
    reps = risk.get("replicate_statistics", [])
    if len(reps) != DRAWS:
        raise ValueError("Every fresh method must retain exactly 500 complete replicate observations")
    summaries = {}
    for metric in ("accuracy", "bounded_loss", "cross_entropy"):
        values = [rep.get(metric) for rep in reps]
        if any(not finite(value) or value < 0 or (metric != "cross_entropy" and value > 1) for value in values):
            raise ValueError("Fresh MC replicate metrics are invalid")
        stats = statistics_of(values)
        if not finite(risk.get(metric)) or abs(stats["mean"]-risk[metric]) > 1e-10:
            raise ValueError("Fresh MC mean differs from saved replicate observations")
        summaries[metric] = stats
    return summaries


def evaluate_one(task, output, protocol_hash, environment):
    safe_task_id(task["task_id"])
    checkpoint = Path(task["checkpoint"])
    root = disjoint_output(output, (checkpoint, checkpoint.parent, checkpoint.parent.parent))
    guard_cache(root)
    output = owned_path(root, root / "evaluations" / task["task_id"])
    if task["fresh_evaluation_seed"] != normalize_config(task["config"])["eval_seed"]+SEED_OFFSET:
        raise ValueError("Fresh exploratory seed must use the prespecified offset")
    for filename in ("request.json", "overrides.json", "evaluation.json", "evaluate.log"):
        owned_path(root, output / filename)
        owned_path(root, (output / filename).with_suffix(Path(filename).suffix+".tmp"))
    before = {name: fingerprint(checkpoint / filename) for name, filename in
              (("metadata", "metadata.json"), ("weights", "weights.pt"))}
    if before != task["checkpoint_source_fingerprints"]:
        raise ValueError("Selected checkpoint changed after test-based seed selection")
    overrides = dict(device="cpu", threads=THREADS, download=False, eval_draws=DRAWS,
                     eval_seed=task["fresh_evaluation_seed"])
    if fingerprint(task["initial_report"]) != task["initial_report_fingerprint"]:
        raise ValueError("Frozen initial test report changed before fresh verification")
    request = dict(task_id=task["task_id"], protocol_sha256=protocol_hash,
        checkpoint=task["checkpoint"], checkpoint_source_fingerprints=before,
        initial_report_fingerprint=task["initial_report_fingerprint"], overrides=overrides,
        interpretation=EXPLORATORY, no_retry_after_measured_loss=True)
    report_path = output / "evaluation.json"
    if report_path.exists() and not (output / "request.json").is_file():
        raise ValueError("Refuse a preexisting MC result without its prior frozen request")
    owned_write(root, output / "request.json", encoded(request))
    owned_write(root, output / "overrides.json", encoded(overrides))
    if not report_path.exists():
        with owned_path(root, output / "evaluate.log").open("a") as stream:
            subprocess.run([sys.executable, "-u", "-m", "journal.run_experiment", "--mode", "evaluate",
                "--checkpoint", str(checkpoint), "--config", str(output / "overrides.json"), "--output", str(output),
                "--split", "test", "--risk-scope", "deployment"], stdout=stream,
                stderr=subprocess.STDOUT, env=environment, check=True)
    report = json.loads(report_path.read_text())
    summaries = validate_fresh(report, task)
    if before != {name: fingerprint(checkpoint / filename) for name, filename in
                  (("metadata", "metadata.json"), ("weights", "weights.pt"))}:
        raise ValueError("Selected source checkpoint changed during fresh MC")
    return dict(task_id=task["task_id"], group=task["group"], method_label=task["method_label"],
        seed=task["seed"], fresh_evaluation_seed=task["fresh_evaluation_seed"], statistics=summaries,
        accuracy_draws=[rep["accuracy"] for rep in report["heldout_gibbs_mc"]["replicate_statistics"]],
        report=str(report_path.resolve()), report_fingerprint=fingerprint(report_path), interpretation=EXPLORATORY)


def summarize(protocol, evaluations):
    by_id = {row["task_id"]: row for row in evaluations}
    expected = {task_id for case in protocol["conditions"] for task_id in case["selected_task_ids"]}
    if len(by_id) != len(evaluations) or set(by_id) != expected:
        raise ValueError("Fresh verification requires all eight methods for every selected paired seed")
    conditions = []
    for case in protocol["conditions"]:
        if case["selected_seed"] is None:
            conditions.append({**case, "fresh_verification": "not_run_no_initial_winning_seed", "methods": []})
            continue
        methods = {by_id[task]["method_label"]: by_id[task] for task in case["selected_task_ids"]}
        scores = {label: row["statistics"]["accuracy"]["mean"] for label, row in methods.items()}
        candidate = best(scores, PROPOSED)
        fixed = case["initial_candidate"]
        top_score = max(scores.values())
        top_methods = [label for label in LABELS if abs(scores[label]-top_score) <= TOLERANCE]
        initial_scores = next(item["scores"] for item in case["all_five_seed_search"]
                              if item["seed"] == case["selected_seed"])
        initial_top_score = max(initial_scores.values())
        initial_top_methods = [label for label in LABELS if abs(initial_scores[label]-initial_top_score) <= TOLERANCE]
        pairs = {}
        for selected in {fixed, candidate}:
            pairs[selected] = {}
            for other in LABELS:
                if other != selected:
                    pairs[selected][other] = statistics_of(a-b for a, b in zip(methods[selected]["accuracy_draws"], methods[other]["accuracy_draws"]))
        conditions.append({**case, "fresh_verification": "complete_once_500_draws_all_eight_methods",
            "fresh_scores": scores, "fresh_fixed_initial_candidate": margin(scores, fixed),
            "fresh_best_matched_proposed": margin(scores, candidate), "fresh_highest_method": best(scores, LABELS),
            "initial_top_methods": initial_top_methods, "fresh_top_methods": top_methods,
            "winner_changed": set(initial_top_methods) != set(top_methods),
            "fixed_candidate_not_fresh_top": fixed not in top_methods,
            "initial_strict_win_lost": case.get("initial_strict_win", False) and not margin(scores, fixed)["strictly_best"],
            "paired_mc_differences": pairs,
            "paired_mc_scope": "Descriptive SD/SE across common-seed full-loader MC replicates; fixed checkpoint/test set; excludes test-selection and training-seed uncertainty",
            "methods": [methods[label] for label in LABELS]})
    return dict(schema_version=1, interpretation=EXPLORATORY, conditions=conditions,
        original_five_seed_means_retained=True, original_main_files_modified=False,
        all_settings=protocol.get("all_settings", False),
        selection_mode=protocol.get("selection_mode", "strict_positive_only"),
        fresh_draws=DRAWS, retry_after_loss=False, new_training=False,
        original_main_means=protocol["original_main_means"], original_mean_rankings=protocol["original_mean_rankings"])


def run(source_root, output, main_revision="f602987", workers=16, select_only=False, all_settings=False):
    from .batch_campaign import validate_cpu_budget
    if workers != 16:
        raise ValueError("Use the prescribed sixteen two-thread CPU workers")
    validate_cpu_budget(workers, THREADS)
    code = verify_code(main_revision)
    if os.environ.get("JOURNAL_GIT_SHA", code["execution_revision"]) != code["execution_revision"]:
        raise ValueError("Submitted exploratory evaluator revision differs from the committed checkout")
    source_root = Path(source_root).resolve()
    output = preflight_output(source_root, output)
    output.mkdir(parents=True, exist_ok=True)
    with owned_path(output, output / ".verification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze_selection(source_root, output, code, all_settings)
        protocol_hash = fingerprint(output / "selection_protocol.json")["sha256"]
        if select_only:
            return dict(status="selection_frozen_before_fresh_mc", selection=str(output / "selection_protocol.json"),
                        all_settings=all_settings, selection_mode=protocol["selection_mode"], interpretation=EXPLORATORY)
        chosen = {task for case in protocol["conditions"] for task in case["selected_task_ids"]}
        tasks = [task for task in protocol["all_original_tasks"] if task["task_id"] in chosen]
        status = dict(status="running", started_at=now(), interpretation=EXPLORATORY,
                      fixed_draws=DRAWS, all_settings=all_settings, selected_checkpoints=len(tasks), completed=[], failed=[])
        owned_save(output, output / "verification_status.json", status)
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        evaluations = []
        try:
            recheck(protocol)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(evaluate_one, task, output, protocol_hash, environment): task for task in tasks}
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        value = future.result()
                        evaluations.append(value)
                        status["completed"].append(task["task_id"])
                    except Exception as error:
                        status["failed"].append(dict(task_id=task["task_id"], error=f"{type(error).__name__}: {error}"))
                    owned_save(output, output / "verification_status.json", status)
            if status["failed"]:
                raise RuntimeError("Exploratory verification has incomplete methods; all failures retained")
            recheck(protocol)
            if verify_code(main_revision) != code:
                raise ValueError("Committed evaluator sources changed during exploratory MC")
            report = summarize(protocol, evaluations)
            report.update(selection_protocol=str((output / "selection_protocol.json").resolve()), protocol_sha256=protocol_hash)
            owned_save(output, output / "exploratory_seed_examples.json", report)
            status.update(status="complete", completed_at=now(), report=str(output / "exploratory_seed_examples.json"))
            owned_save(output, output / "verification_status.json", status)
            return report
        except BaseException as error:
            status.update(status="failed", failed_at=now(), error=f"{type(error).__name__}: {error}")
            owned_save(output, output / "verification_status.json", status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-revision", default="f602987")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--all-settings", action="store_true",
                        help="Verify the best available existing paired seed in every setting, including initial losses/ties")
    args = parser.parse_args()
    result = run(args.source_root, args.output, args.main_revision, args.workers, args.select_only, args.all_settings)
    print(json.dumps(dict(output=str(args.output.resolve()), interpretation=EXPLORATORY,
                         selected_conditions=sum(case["selected_seed"] is not None for case in result.get("conditions", [])))))


if __name__ == "__main__":
    main()
