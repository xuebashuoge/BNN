"""Assemble the prespecified main16/non-MNIST + validation-selected MNIST32 cohort.

This is read-only postprocessing of completed source pipelines. It never trains,
loads datasets, chooses favorable test seeds, or switches the MNIST cohort based
on its test scores. Owned leaf aliases point at verified existing checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

from .build_bayesian_campaign import DATASETS, SEEDS, read_entries, screen_entries, selected_entries, select_validation
from .build_bayesian_mnist_scale import (canonical_validation_rows, combine_and_select, manifest_text,
                                        original_mnist_entries, read_training_records)
from .campaign import save_json, validate_entries
from .run_bayesian_campaign import ACCURACY_TOLERANCE, ranking_outcome
from .run_experiment import normalize_config


STAGES = ("confirm", "m_sweep", "p_train_sweep")
COUNTS = dict(confirm=320, m_sweep=1056, p_train_sweep=672)
MNIST_COUNTS = dict(confirm=80, m_sweep=264, p_train_sweep=168)
MAIN_REVISION = "f602987"
MNIST_REVISION = "7087207"
SOURCE_FILES = ("journal/run_experiment.py", "journal/theorem1.py", "journal/cifar_models.py",
                "journal/cifar_search_models.py", "journal/campaign.py",
                "journal/build_bayesian_campaign.py", "journal/build_campaign.py")
CONTROL_FILES = ("journal/build_bayesian_mnist_scale.py", "journal/run_bayesian_mnist_scale.py",
                 "journal/mnist_scale_control.slurm")
POLICY = dict(mnist="Always the completed 32-candidate validation-selected MNIST control, regardless of test ranking",
              other_datasets="All selected main16 moons/FashionMNIST/CIFAR-10 cases",
              original_mnist="Retained unchanged at the original source, explicitly excluded from this composite",
              seed_selection="Every fixed confirmation/sweep seed; no favorable per-seed method switching",
              main_failure_budget=.025, separate_small_model_failure_budget=.025,
              combined_union_failure_budget=.05, final_runs=2048)


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


class SourceReader:
    """Record bytes and logical-to-physical paths, then recheck before delivery."""
    def __init__(self):
        self.files, self.directories = {}, {}

    def track(self, path):
        path = Path(path).absolute()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Require a regular source file: {path}")
        value = dict(resolved=str(path.resolve()), **fingerprint(path))
        key = str(path)
        if key in self.files and self.files[key] != value:
            raise ValueError(f"Source changed while assembling: {path}")
        self.files[key] = value
        return path

    def json(self, path):
        path = self.track(path)
        return json.loads(path.read_text())

    def entries(self, path):
        path = self.track(path)
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def directory(self, path):
        path = Path(path).absolute()
        stat = path.stat()
        value = dict(resolved=str(path.resolve()), device=stat.st_dev, inode=stat.st_ino,
                     symlink=os.readlink(path) if path.is_symlink() else None)
        self.directories[str(path)] = value
        return value

    def recheck(self):
        for path, expected in self.files.items():
            actual = dict(resolved=str(Path(path).resolve()), **fingerprint(path))
            if Path(path).is_symlink() or actual != expected:
                raise ValueError(f"A frozen source changed: {path}")
        for path, expected in self.directories.items():
            stat = Path(path).stat()
            actual = dict(resolved=str(Path(path).resolve()), device=stat.st_dev, inode=stat.st_ino,
                          symlink=os.readlink(path) if Path(path).is_symlink() else None)
            if actual != expected:
                raise ValueError(f"A source stage alias/directory changed: {path}")


def verify_code_sources(repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
    main = git("rev-parse", "--verify", "--end-of-options", MAIN_REVISION + "^{commit}")
    control = git("rev-parse", "--verify", "--end-of-options", MNIST_REVISION + "^{commit}")
    execution = git("rev-parse", "HEAD^{commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Assemble using a clean committed postprocessing checkout")
    files = {}
    for filename in SOURCE_FILES:
        reference = git("rev-parse", main + ":" + filename)
        if (git("rev-parse", control + ":" + filename) != reference
                or git("rev-parse", execution + ":" + filename) != reference
                or git("hash-object", "--", filename) != reference):
            raise ValueError(f"Scientific/builder source differs from the prescribed cohorts: {filename}")
        files[filename] = reference
    controls = {filename: git("rev-parse", control + ":" + filename) for filename in CONTROL_FILES}
    builder = CONTROL_FILES[0]
    if git("hash-object", "--", builder) != controls[builder]:
        raise ValueError("MNIST32 selection replay differs from its committed builder")
    postprocessing = {}
    for filename in ("journal/assemble_bayesian_results.py", "journal/run_bayesian_campaign.py",
                     "journal/bayesian_bounds.py", "journal/plot_bayesian_campaign.py",
                     "journal/stage_bayesian_checkpoints.py"):
        blob = git("rev-parse", execution + ":" + filename)
        if git("hash-object", "--", filename) != blob:
            raise ValueError("Assembly/postprocessing code differs from its committed source")
        postprocessing[filename] = blob
    return dict(main_revision=main, mnist_revision=control, assembly_revision=execution,
                scientific_blobs=files, mnist_control_blobs=controls, postprocessing_blobs=postprocessing)


def same_entries(actual, expected, label):
    left = {item["task_id"]: item for item in actual}
    right = {item["task_id"]: item for item in expected}
    if len(left) != len(actual) or len(right) != len(expected) or set(left) != set(right):
        raise ValueError(f"Missing, duplicate or foreign task in {label}")
    for task_id, item in left.items():
        if item != right[task_id]:
            raise ValueError(f"Validation-selected configuration/identity changed in {label}: {task_id}")


def require_fingerprints(recorded, paths):
    expected = {str(Path(path).resolve()): fingerprint(path) for path in paths}
    if recorded != expected:
        raise ValueError("Frozen validation source fingerprints differ from their recorded protocol")
    return expected


def replay_selections(main_root, mnist_root, reader, code):
    """Complete both validation replays before any source test table is opened."""
    main_root, mnist_root = Path(main_root), Path(mnist_root)
    main_state = reader.json(main_root / "pipeline_status.json")
    control_state = reader.json(mnist_root / "control_status.json")
    if (main_state.get("status") != "complete" or main_state.get("current_stage") is not None
            or main_state.get("git_sha") != code["main_revision"]
            or main_state.get("confirm_draws") != 20 or main_state.get("sweep_draws") != 10):
        raise ValueError("Require the completed prescribed main16 pipeline with its fixed evaluation draws")
    if (control_state.get("status") != "complete" or control_state.get("current_stage") is not None
            or control_state.get("git_sha") != code["mnist_revision"]
            or control_state.get("main_revision") != code["main_revision"]
            or Path(control_state.get("main_root", "")).resolve() != main_root.resolve()):
        raise ValueError("Require the completed prescribed MNIST32 control; no cohort substitution is allowed")
    verification = control_state.get("source_verification", {})
    if (verification.get("main_revision") != code["main_revision"]
            or verification.get("execution_revision") != code["mnist_revision"]
            or verification.get("control_files") != code["mnist_control_blobs"]):
        raise ValueError("MNIST control committed-source attestation differs from the prescribed source")
    expected_files = {path: dict(main_blob=blob, control_blob=blob, working_blob=blob)
                      for path, blob in code["scientific_blobs"].items()}
    if verification.get("files") != expected_files:
        raise ValueError("MNIST control training/builder blobs differ from main16")
    screen = main_root / "manifests" / "screen.jsonl"
    frozen = main_root / "validation_selection_results.json"
    main_entries, main_rows = reader.entries(screen), reader.json(frozen)
    same_entries(main_entries, screen_entries(), "prescribed main16 screening grid")
    if main_state.get("stages", {}).get("screen", {}).get("manifest_sha256") != fingerprint(screen)["sha256"]:
        raise ValueError("Main screening manifest differs from its immutable recorded hash")
    current_rows = reader.json(main_root / "screen" / "training_results.json")
    canonical = lambda rows: sorted(({key: value for key, value in row.items() if key != "resumed"}
                                     for row in rows), key=lambda row: row["task_id"])
    if canonical(main_rows) != canonical(current_rows):
        raise ValueError("Main frozen validation selection differs from complete training records")
    main_selected = select_validation(main_entries, main_rows)
    original = mnist_root / "inputs" / "original"
    original_manifest, original_results = original / "original_manifest.jsonl", original / "original_training_results.json"
    capture_paths = (mnist_root / "inputs" / "source_capture" / "screen.jsonl",
                     mnist_root / "inputs" / "source_capture" / "training_results.json")
    capture = reader.json(mnist_root / "inputs" / "capture_protocol.json")
    if (capture.get("main_root") != str(main_root.resolve()) or capture.get("main_revision") != code["main_revision"]
            or capture.get("main_manifest") != str(screen.resolve())
            or capture.get("main_training_results") != str((main_root / "screen" / "training_results.json").resolve())
            or capture.get("original_mnist_tasks") != 512
            or capture.get("official_test_files_opened") is not False):
        raise ValueError("MNIST original capture identifies another cohort or selection source")
    for path in capture_paths:
        reader.track(path)
    require_fingerprints(capture.get("capture_fingerprints"), capture_paths)
    snapshot = reader.json(original / "original_snapshot.json")
    require_fingerprints(snapshot.get("source_fingerprints"), capture_paths)
    if control_state.get("original_capture") != capture:
        raise ValueError("MNIST state differs from its frozen original capture")
    original_entries = reader.entries(original_manifest)
    original_rows = reader.json(original_results)
    same_entries(original_entries, original_mnist_entries(main_entries), "original MNIST capture")
    original_ids = {entry["task_id"] for entry in original_entries}
    from_main = canonical_validation_rows(original_entries, [row for row in main_rows if row["task_id"] in original_ids], 16)
    if canonical_validation_rows(original_entries, original_rows, 16) != from_main:
        raise ValueError("Captured original MNIST validation differs from the main frozen selection")
    captured_entries, captured_rows = reader.entries(capture_paths[0]), reader.json(capture_paths[1])
    same_entries(captured_entries, main_entries, "raw original main manifest capture")
    if any(row.get("config", {}).get("dataset") == "mnist" and row["task_id"] not in original_ids for row in captured_rows):
        raise ValueError("Foreign MNIST record in the raw pre-test source capture")
    captured_mnist = canonical_validation_rows(original_entries,
        [row for row in captured_rows if row["task_id"] in original_ids], 16)
    if captured_mnist != from_main:
        raise ValueError("Raw pre-test MNIST capture differs from the frozen original validation records")
    extra_manifest, extra_results = mnist_root / "manifests" / "screen.jsonl", mnist_root / "screen" / "training_results.json"
    paths = (original_manifest, original_results, extra_manifest, extra_results)
    extra, extra_rows = reader.entries(extra_manifest), reader.json(extra_results)
    combined, selected_rows, mnist_selected = combine_and_select(original_entries, original_rows, extra, extra_rows)
    selected_root = mnist_root / "selected"
    protocol = reader.json(selected_root / "selection_protocol.json")
    expected_fps = require_fingerprints(protocol.get("source_fingerprints"), paths)
    if (protocol.get("candidates_per_method_regime") != 32 or protocol.get("combined_trials") != 1024
            or protocol.get("selection_uses_test") is not False or protocol.get("official_test_files_opened") is not False
            or control_state.get("validation_selection") != protocol):
        raise ValueError("MNIST32 selection protocol is incomplete or changed")
    same_entries(reader.entries(selected_root / "combined_manifest.jsonl"), combined, "MNIST32 combined validation manifest")
    if reader.json(selected_root / "validation_selection_results.json") != selected_rows:
        raise ValueError("Frozen MNIST32 validation records changed")
    saved_selected = reader.json(selected_root / "selected_variants.json")
    selection_identity = lambda items: sorted((item["group"], item["method_label"], item["variant_id"],
                                               tuple(item["seeds"]), json.dumps(item["config"], sort_keys=True)) for item in items)
    if selection_identity(saved_selected) != selection_identity(mnist_selected):
        raise ValueError("MNIST32 finalist differs from validation-only reselection")
    manifests, exclusions = {}, {}
    for stage in STAGES:
        main_path, control_path = main_root / "manifests" / f"{stage}.jsonl", selected_root / f"{stage}.jsonl"
        primary, control = reader.entries(main_path), reader.entries(control_path)
        same_entries(primary, selected_entries(main_selected, SEEDS[stage], stage), "main16 " + stage)
        same_entries(control, selected_entries(mnist_selected, SEEDS[stage], stage), "MNIST32 " + stage)
        for state, path, count in ((main_state, main_path, COUNTS[stage]), (control_state, control_path, MNIST_COUNTS[stage])):
            recorded = state.get("stages", {}).get(stage, {})
            if (recorded.get("status") != "complete" or recorded.get("tasks") != count
                    or recorded.get("manifest_sha256") != fingerprint(path)["sha256"]):
                raise ValueError("A final source stage is incomplete or its manifest changed")
        primary_plan = reader.json(main_path.with_suffix(".plan.json"))
        control_plan = reader.json(control_path.with_suffix(".plan.json"))
        if primary_plan.get("source_sha256") != {str(path): fingerprint(path)["sha256"] for path in (screen, frozen)}:
            raise ValueError("Main final manifest is not bound to the frozen validation sources")
        if control_plan.get("source_sha256") != {path: value["sha256"] for path, value in expected_fps.items()}:
            raise ValueError("MNIST final manifest is not bound to the complete 32-candidate selection")
        manifests[stage] = dict(main=primary, mnist=control, main_path=main_path, mnist_path=control_path)
        exclusions[stage] = [entry["task_id"] for entry in primary if entry["config"]["dataset"] == "mnist"]
    reader.recheck()
    return manifests, exclusions, dict(main=main_state, mnist=control_state,
                                     mnist_selected_variants=[dict(group=item["group"], method=item["method_label"],
                                                                  variant_id=item["variant_id"]) for item in mnist_selected])


def choose_cohort(main_entries, mnist_entries, stage):
    """Fixed dataset rule, independent of all test scores and outcome rankings."""
    if (len(main_entries) != COUNTS[stage] or len(mnist_entries) != MNIST_COUNTS[stage]
            or any(item["config"]["dataset"] != "mnist" for item in mnist_entries)):
        raise ValueError("Source manifests lack the prescribed complete cohort counts")
    selected = [(entry, "main16") for entry in main_entries if entry["config"]["dataset"] != "mnist"]
    selected += [(entry, "mnist32") for entry in mnist_entries]
    if (len(selected) != COUNTS[stage] or len({entry["task_id"] for entry, _ in selected}) != COUNTS[stage]
            or Counter(entry["config"]["dataset"] for entry, _ in selected) != {dataset: MNIST_COUNTS[stage] for dataset in DATASETS}):
        raise ValueError("Composite contains missing, duplicate or foreign dataset cases")
    validate_entries([entry for entry, _ in selected])
    for entry, _ in selected:
        if entry["config"]["seed"] not in SEEDS[stage] or entry.get("stage") != "bayes_" + stage:
            raise ValueError("Composite contains a foreign stage or confirmation/sweep seed")
    return sorted(selected, key=lambda item: (item[0]["group"], item[0]["method_label"], item[0]["config"]["seed"], item[0]["task_id"]))


def verify_checkpoint_revision(revision, code, cache):
    if revision in cache:
        return
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Saved checkpoint has no valid full committed-source revision")
    for filename, expected in code["scientific_blobs"].items():
        actual = subprocess.check_output(["git", "rev-parse", revision + ":" + filename],
            cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.PIPE).strip()
        if actual != expected:
            raise ValueError("Saved checkpoint revision differs from the prescribed scientific sources")
    cache.add(revision)


def published_stage_records(root, entries, stage, cohort, reader, code):
    """Only main's approved whole p-stage alias may precede source checkpoints."""
    root = Path(root)
    if not root.is_symlink():
        return None
    if (cohort, stage) != ("main16", "p_train_sweep"):
        raise ValueError("Only the published main p_train stage may be a source stage symlink")
    source = root.resolve(strict=True)
    audit = reader.json(source.parent / "publication_audit.json")
    stat = source.stat()
    verification = audit.get("training_source_verification", {})
    if (audit.get("status") != "published" or audit.get("publication_kind") != "atomic_directory_symlink"
            or audit.get("source") != str(source) or audit.get("target") != str(root.absolute())
            or os.readlink(root) != str(source) or audit.get("source_must_remain_available") is not True
            or audit.get("stage_identity") != dict(device=stat.st_dev, inode=stat.st_ino)
            or verification.get("main_revision") != code["main_revision"]):
        raise ValueError("Published main stage link lacks its exact permanent-source audit")
    expected_files = {path: dict(main_blob=blob, external_blob=blob, working_blob=blob)
                      for path, blob in code["scientific_blobs"].items()}
    if verification.get("files") != expected_files:
        raise ValueError("Published stage training/builder blobs differ from main16")
    records = audit.get("complete_validation", {}).get("source_records", [])
    by_id = {record["task_id"]: record for record in records}
    if len(records) != 672 or len(by_id) != 672 or set(by_id) != {entry["task_id"] for entry in entries}:
        raise ValueError("Published main stage audit lacks all672 original cases")
    return dict(records=by_id, revision=verification["external_revision"])


def load_source_stage(root, manifest, entries, stage, cohort, reader, code, revision_cache):
    from .bayesian_bounds import observation
    from .plot_bayesian_campaign import Evidence
    from .stage_bayesian_checkpoints import CHECKPOINT_FILES, validate_checkpoint
    root = Path(root)
    reader.directory(root)
    published = published_stage_records(root, entries, stage, cohort, reader, code)
    summary = reader.json(root / "summary.json")
    expected = {entry["task_id"]: entry for entry in entries}
    if (summary.get("training_trials") != len(entries) or summary.get("completed_trials") != len(entries)
            or summary.get("failed_trials") or summary.get("failed_evaluations")):
        raise ValueError("Incomplete training/evaluation source stage")
    training, test = reader.json(root / "training_results.json"), reader.json(root / "test_results.json")
    for rows in (training, test):
        if len(rows) != len(expected) or {row["task_id"] for row in rows} != set(expected):
            raise ValueError("Missing, duplicate or foreign source result rows")
    training = {row["task_id"]: row for row in training}
    test = {row["task_id"]: row for row in test}
    evidence = Evidence(root, manifest)
    if evidence.excluded_task_ids or len(evidence.rows) != len(entries):
        raise ValueError("Incomplete source full-risk evidence")
    for path, checksum in evidence.files.items():
        reader.track(path)
        if fingerprint(path)["sha256"] != checksum:
            raise ValueError("Source risk report changed while validating")
    cases = {}
    for task_id, entry in expected.items():
        if cohort == "main16" and entry["config"]["dataset"] == "mnist":
            continue
        logical = root / task_id
        if logical.is_symlink():
            raise ValueError("Source leaf checkpoints cannot be substituted by directory aliases")
        checkpoint = logical.resolve(strict=True)
        if any(path.is_symlink() for path in checkpoint.rglob("*")):
            raise ValueError("Scientific checkpoint files cannot be symbolic links")
        metadata = reader.json(checkpoint / "metadata.json")
        revision = metadata["git_sha"]
        if cohort == "mnist32" and revision != code["mnist_revision"]:
            raise ValueError("MNIST composite must use the control cohort's actual checkpoints")
        verify_checkpoint_revision(revision, code, revision_cache)
        saved, reason = validate_checkpoint(checkpoint, entry, revision)
        if saved is None:
            raise ValueError(f"Invalid saved checkpoint {task_id}: {reason}")
        for rows in (training, test):
            row = rows[task_id]
            if (row.get("status") != "complete" or normalize_config(row["config"]) != normalize_config(entry["config"])
                    or any(row.get(key) != entry.get(key) for key in ("group", "method_label", "family", "variant_id", "stage"))
                    or row.get("seed") != entry["config"]["seed"]
                    or Path(row["checkpoint"]).resolve() != checkpoint):
                raise ValueError("Source result identity/configuration/checkpoint path mismatch")
        for rows in (training, test):
            for metric, value in saved.items():
                if rows[task_id].get(metric) != value or (isinstance(value, (int, float)) and isinstance(rows[task_id].get(metric), bool)):
                    raise ValueError("Saved checkpoint validation metadata differs from its result row")
        report_path = checkpoint / "test_full" / "evaluation.json"
        if Path(test[task_id].get("test_report", "")).resolve() != report_path:
            raise ValueError("Source test row points to a different full evaluation")
        report = reader.json(report_path)
        heldout = report["heldout_gibbs_mc"]
        if any(isinstance(test[task_id].get(metric), bool) or test[task_id].get(metric) != heldout[key]
               for metric, key in (("test_accuracy", "accuracy"), ("test_cross_entropy", "cross_entropy"))):
            raise ValueError("Source test summary differs from its exact full-risk report metrics")
        evaluated = report["evaluation_config"]
        if (evaluated["eval_draws"] != (20 if stage == "confirm" else 10)
                or evaluated["eval_seed"] != entry["config"]["eval_seed"] + 1000003
                or evaluated["device"] != "cpu" or evaluated["threads"] != 2):
            raise ValueError("Source uses a foreign final risk-evaluation stream")
        checked = observation(entry, test[task_id], root, 2048, .025)
        if published and (revision != published["revision"]
                or checked["source_fingerprints"] != published["records"][task_id]["source_fingerprints"]):
            raise ValueError("Published checkpoint or scientific evaluation bytes changed after publication")
        if not math.isfinite(checked["posterior_kl_float64"]):
            raise ValueError("Source Gaussian posterior has no finite KL")
        for filename in CHECKPOINT_FILES:
            reader.track(checkpoint / filename)
        cases[task_id] = dict(entry=entry, training=training[task_id], test=test[task_id], cohort=cohort,
            checkpoint=checkpoint, evaluation=report_path, report=report,
            source_fingerprints=checked["source_fingerprints"], checkpoint_revision=revision)
    return cases


def link_checkpoint(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if not source.is_dir() or destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace an assembly leaf alias or existing output")
    os.symlink(str(source), destination, target_is_directory=True)


def write_tables(output):
    root = Path(output)
    outcome = ranking_outcome(root / "confirm", manifest=root / "manifests" / "confirm.jsonl")
    if len(outcome["conditions"]) != 8:
        raise ValueError("Composite ranking requires all eight dataset/channel conditions")
    entries = read_entries(root / "manifests" / "confirm.jsonl")
    design = {entry["group"]: entry for entry in entries}
    rows = json.loads((root / "confirm" / "test_results.json").read_text())
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["method_label"])].append(row)
    means = []
    for (group, label), runs in sorted(grouped.items()):
        if sorted(row["seed"] for row in runs) != list(SEEDS["confirm"]):
            raise ValueError("Main means require all five fixed confirmation seeds")
        entry = design[group]
        reports = [json.loads(Path(row["test_report"]).read_text()) for row in runs]
        means.append(dict(dataset=entry["config"]["dataset"], channel=entry["channel_name"], group=group,
            method_label=label, source_cohort="mnist32" if entry["config"]["dataset"] == "mnist" else "main16",
            tuning_candidates_per_method_regime=32 if entry["config"]["dataset"] == "mnist" else 16,
            n=5, seeds_json=json.dumps(list(SEEDS["confirm"])), accuracy_mean=statistics.fmean(row["test_accuracy"] for row in runs),
            accuracy_sd=statistics.stdev(row["test_accuracy"] for row in runs),
            test_cross_entropy_mean=statistics.fmean(row["test_cross_entropy"] for row in runs),
            bounded_test_risk_mean=statistics.fmean(report["heldout_gibbs_mc"]["bounded_loss"] for report in reports),
            bounded_empirical_risk_mean=statistics.fmean(report["empirical_gibbs_mc"]["bounded_loss"] for report in reports),
            observed_bounded_gap_mean=statistics.fmean(report["signed_gap_decomposition_mc"]["bounded_loss"]["observed_gap"] for report in reports)))
    if len(means) != 64:
        raise ValueError("Expected all64 main method/condition means")
    goals = []
    for condition in outcome["conditions"]:
        entry = design[condition["group"]]
        goals.append(dict(dataset=entry["config"]["dataset"], channel=entry["channel_name"], group=condition["group"],
            fixed_matched_proposed=condition["best_matched_proposed_by_observed_mean"],
            strongest_other_method=condition["strongest_other_method"], mean_margin=condition["mean_accuracy_margin"],
            mean_margin_percentage_points=100 * condition["mean_accuracy_margin"],
            target_met=condition["target_met"], target_met_for_every_seed=condition["target_met_for_every_seed"],
            accuracy_tolerance=ACCURACY_TOLERANCE))
    save_json(root / "accuracy_outcome.json", {**outcome, "policy": POLICY, "accuracy_tolerance": ACCURACY_TOLERANCE})
    for name, values in (("main_means", means), ("goal_summary", goals)):
        save_json(root / (name + ".json"), values)
        with (root / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return outcome


def assemble(main_root, mnist_root, output, run_bounds=False, run_plots=False):
    main_root, mnist_root = Path(main_root).resolve(strict=True), Path(mnist_root).resolve(strict=True)
    requested = Path(output)
    if requested.exists() or requested.is_symlink():
        raise ValueError("Assembly requires a fresh absent owned output root")
    output = requested.resolve()
    for source in (main_root, mnist_root):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Assembly output must be separate from both source roots")
    # Small repeated tensor-integrity/KL operations should not create a full
    # node's worth of workers. This changes no saved model or training setting.
    import torch
    torch.set_num_threads(2)
    code = verify_code_sources()
    reader = SourceReader()
    manifests, excluded, replay = replay_selections(main_root, mnist_root, reader, code)
    all_cases, chosen = {}, {}
    revision_cache = set()
    for stage in STAGES:
        data = manifests[stage]
        chosen[stage] = choose_cohort(data["main"], data["mnist"], stage)
        for source_root, key, cohort in ((main_root, "main", "main16"), (mnist_root, "mnist", "mnist32")):
            actual_root = (source_root / stage).resolve(strict=True)
            if output == actual_root or actual_root in output.parents or output in actual_root.parents:
                raise ValueError("Assembly output overlaps an actual source stage")
            all_cases[(stage, cohort)] = load_source_stage(source_root / stage, data[key + "_path"], data[key],
                                                         stage, cohort, reader, code, revision_cache)
    physical = [str(case["checkpoint"]) for cases in all_cases.values() for case in cases.values()]
    if len(physical) != 2048 or len(set(physical)) != 2048:
        raise ValueError("Final composite requires2048 distinct actual checkpoints")
    reader.recheck()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="assembling", started_at=now(), policy=POLICY, main_root=str(main_root), mnist_root=str(mnist_root),
                 output=str(output), code_sources=code, stages={}, train_or_inference_jobs_started=0,
                 formal_bounds="pending", figures="pending")
    save_json(output / "assembly_status.json", state)
    source_map = []
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    try:
        (output / "manifests").mkdir()
        for stage in STAGES:
            stage_root = output / stage
            stage_root.mkdir()
            reports_root = output / "evaluation_reports" / stage
            reports_root.mkdir(parents=True)
            entries, training, tests = [], [], []
            for entry, cohort in chosen[stage]:
                case = all_cases[(stage, cohort)][entry["task_id"]]
                declared = copy.deepcopy(entry)
                declared["assembly_source_cohort"] = cohort
                entries.append(declared)
                alias = stage_root / entry["task_id"]
                link_checkpoint(case["checkpoint"], alias)
                exported_report = reports_root / (entry["task_id"] + ".json")
                report_fingerprint = fingerprint(case["evaluation"])
                shutil.copyfile(case["evaluation"], exported_report)
                if fingerprint(exported_report) != report_fingerprint or fingerprint(case["evaluation"]) != report_fingerprint:
                    raise ValueError("A full evaluation changed while exporting its numerical report")
                train = copy.deepcopy(case["training"])
                test = copy.deepcopy(case["test"])
                for row in (train, test):
                    row.update(checkpoint=str(alias), assembly_source_cohort=cohort)
                test["test_report"] = str(exported_report)
                training.append(train)
                tests.append(test)
                source_map.append(dict(stage=stage, task_id=entry["task_id"], cohort=cohort, alias=str(alias),
                    actual_checkpoint=str(case["checkpoint"]), source_evaluation=str(case["evaluation"]),
                    exported_evaluation=str(exported_report), exported_evaluation_fingerprint=report_fingerprint,
                    checkpoint_revision=case["checkpoint_revision"], source_fingerprints=case["source_fingerprints"]))
            manifest = output / "manifests" / (stage + ".jsonl")
            manifest.write_text(manifest_text(entries))
            save_json(stage_root / "training_results.json", training)
            save_json(stage_root / "test_results.json", tests)
            save_json(stage_root / "summary.json", dict(training_trials=len(entries), completed_trials=len(entries),
                failed_trials=[], failed_evaluations=[], assembly_only=True, new_optimization=0, new_monte_carlo=0))
            state["stages"][stage] = dict(tasks=len(entries), manifest=str(manifest), manifest_fingerprint=fingerprint(manifest))
        reader.recheck()
        outcome = write_tables(output)
        from .plot_bayesian_campaign import Evidence
        for stage in STAGES:
            evidence = Evidence(output / stage, output / "manifests" / (stage + ".jsonl"))
            if len(evidence.rows) != COUNTS[stage] or evidence.excluded_task_ids:
                raise ValueError("Assembled aliases/reports lack complete final coverage")
        audit = dict(policy=POLICY, code_sources=code, validation_replay=replay,
                     source_files=reader.files, source_stages=reader.directories, source_map=source_map,
                     excluded_original_mnist_task_ids=excluded, excluded_original_mnist_total=sum(map(len, excluded.values())),
                     excluded_original_mnist_sources=[dict(cohort="main16", stage=stage, task_id=task_id,
                         logical_checkpoint=str(main_root / stage / task_id),
                         actual_checkpoint=str((main_root / stage / task_id).resolve()),
                         reason="The MNIST32 source cohort was fixed before assembly test reads; no test-based fallback")
                         for stage in STAGES for task_id in excluded[stage]],
                     excluded_source_files_retained=True, source_bytes_modified=False,
                     operation="Read-only assembly with owned per-checkpoint directory aliases; no training or inference")
        save_json(output / "assembly_audit.json", audit)
        if run_bounds:
            command = [sys.executable, "-m", "journal.bayesian_bounds", "--output", str(output / "bounds"), "--delta", ".025"]
            for stage in STAGES:
                command.extend(["--campaign", str(output / "manifests" / (stage + ".jsonl")), str(output / stage)])
            subprocess.run(command, check=True, env=environment)
            bounds = json.loads((output / "bounds" / "risk_bounds.json").read_text())
            declared_ids = {entry["task_id"] for stage in STAGES for entry, _ in chosen[stage]}
            if (bounds.get("total_runs") != 2048 or bounds.get("delta") != .025 or len(bounds.get("rows", [])) != 2048
                    or {row["task_id"] for row in bounds["rows"]} != declared_ids):
                raise ValueError("Formal bounds do not cover the complete prespecified composite family")
            state["formal_bounds"] = "complete2048; delta=.025"
        if run_plots:
            command = [sys.executable, "-m", "journal.plot_bayesian_campaign", "--output", str(output / "figures")]
            for stage in STAGES:
                command.extend(["--source", str(output / stage), str(output / "manifests" / (stage + ".jsonl"))])
            subprocess.run(command, check=True, env=environment)
            index = json.loads((output / "figures" / "figure_index.json").read_text())
            if (len(index.get("figures", [])) != 28
                    or any(len(item["files"]) != 3 or {Path(path).suffix for path in item["files"]} != {".png", ".pdf", ".eps"}
                           or any(not Path(path).is_file() or not Path(path).resolve().is_relative_to((output / "figures").resolve())
                                  for path in item["files"]) for item in index["figures"])):
                raise ValueError("Require28 figure triples in PNG/PDF/EPS")
            state["figures"] = "complete28 triples; no confidence shading"
        reader.recheck()
        state.update(status="complete", completed_at=now(), final_runs=2048,
                     target_met_in_every_condition=outcome["target_met_in_every_condition"],
                     main_mean_rows=64, goal_rows=8, source_bytes_modified=False)
        save_json(output / "assembly_status.json", state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at=now())
        save_json(output / "assembly_status.json", state)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-bounds", action="store_true")
    parser.add_argument("--run-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(assemble(args.main_root, args.mnist_root, args.output, args.run_bounds, args.run_plots), indent=2))


if __name__ == "__main__":
    main()
