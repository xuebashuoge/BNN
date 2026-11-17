"""Append a fair MNIST fixed-positive-Gaussian-scale validation control.

The existing sixteen optimized-scale candidates and their IDs stay intact.
Sixteen counterparts per method/regime change only rho_lr_scale from .1 to 0.
Selection requires all 1024 MNIST trials and uses training/validation records
only. This module neither opens test-result files nor launches experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

from .build_bayesian_campaign import (CHANNELS, METHODS, REGIMES, SEEDS, make_plan,
                                      read_entries, screen_entries, selected_entries)
from .build_campaign import make_entry
from .campaign import select_variants, validate_entries
from .run_experiment import normalize_config


ORIGINAL_TASKS = EXTRA_TASKS = 512
TOTAL_TASKS = 1024
LABELS = {f"{name}-{regime.title()}" for _, name in METHODS for regime in REGIMES}
GROUPS = {f"mnist_{name}" for name in CHANNELS}
VALIDATION_FIELDS = (
    "task_id", "group", "method_label", "family", "variant_id", "stage", "role", "seed",
    "config", "status", "validation_accuracy", "validation_cross_entropy", "selected_epoch",
    "seconds", "peak_rss_mb", "git_sha", "checkpoint")


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def manifest_text(entries):
    return "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in entries)


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def write_once(path, content):
    """Retain immutable manifests/selection inputs across retries."""
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable MNIST scale-control artifact changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def require_training_path(path):
    path = Path(path)
    # Reject an explicitly supplied official-test file before opening it.
    if "test_result" in path.name.lower() or path.name.lower().startswith("test_"):
        raise ValueError("Selection accepts training/validation result files only")
    return path


def read_training_records(path):
    path = require_training_path(path)
    result = json.loads(path.read_text())
    if not isinstance(result, list):
        raise ValueError("Training results must contain a list")
    return result


def original_mnist_entries(entries):
    """Recognize the exact original 16-candidate grid without changing entries."""
    chosen = [copy.deepcopy(item) for item in entries if item["config"]["dataset"] == "mnist"]
    expected = {item["task_id"]: item for item in screen_entries(datasets=("mnist",))}
    actual = {item["task_id"]: item for item in chosen}
    if len(chosen) != ORIGINAL_TASKS or len(actual) != len(chosen) or set(actual) != set(expected):
        raise ValueError("Require all 512 original MNIST screening tasks, preserving their exact IDs")
    for task_id, item in actual.items():
        reference = expected[task_id]
        if normalize_config(item["config"]) != normalize_config(reference["config"]):
            raise ValueError(f"Original MNIST screening configuration changed: {task_id}")
        for key in ("group", "method_label", "variant_id", "family", "stage", "mode",
                    "channel_name", "training_regime", "experiment", "role"):
            if item.get(key) != reference.get(key):
                raise ValueError(f"Original MNIST screening {key} changed: {task_id}")
    validate_entries(chosen)
    return chosen


def extra_entries(original_entries):
    original = original_mnist_entries(original_entries)
    output = []
    for source in original:
        item = copy.deepcopy(source)
        candidate = {**item["config"], "rho_lr_scale": 0.}
        identity = make_entry(candidate, candidate["seed"], source["group"], source["method_label"],
                              "bayes_screen", source["role"])
        item.update(config=identity["config"], task_id=identity["task_id"], variant_id=identity["variant_id"],
                    appended_from_task=source["task_id"], appended_from_variant=source["variant_id"],
                    scale_control="Fixed positive Gaussian scale; optimize posterior means",
                    selection_protocol="Original 16 optimized-scale plus 16 fixed-scale candidates per method/regime; validation only")
        output.append(item)
    if (len(output) != EXTRA_TASKS or len({item["task_id"] for item in output}) != EXTRA_TASKS
            or {item["task_id"] for item in output} & {item["task_id"] for item in original}):
        raise ValueError("New scale-control tasks must be distinct from all original tasks")
    validate_entries(output)
    return output


def validate_extra_entries(original, extra):
    expected = {item["task_id"]: item for item in extra_entries(original)}
    actual = {item["task_id"]: item for item in extra}
    if len(extra) != EXTRA_TASKS or len(actual) != EXTRA_TASKS or set(actual) != set(expected):
        raise ValueError("Require all 512 explicitly appended fixed-scale counterparts")
    for task_id, item in actual.items():
        reference = expected[task_id]
        if normalize_config(item["config"]) != normalize_config(reference["config"]):
            raise ValueError(f"Appended fixed-scale configuration changed: {task_id}")
        for key in ("group", "method_label", "variant_id", "family", "stage", "mode",
                    "channel_name", "training_regime", "experiment", "role",
                    "appended_from_task", "appended_from_variant"):
            if item.get(key) != reference.get(key):
                raise ValueError(f"Appended fixed-scale {key} changed: {task_id}")
    validate_entries(extra)


def canonical_validation_rows(entries, records, variants_per_label):
    """Validate every declared result and retain only training/validation fields."""
    expected = {item["task_id"]: item for item in entries}
    actual = {row["task_id"]: row for row in records}
    if len(actual) != len(records) or set(actual) != set(expected):
        raise ValueError("Validation selection requires exactly every declared training result")
    counts, seed_sets, output = defaultdict(Counter), defaultdict(lambda: defaultdict(set)), []
    for task_id in sorted(expected):
        source, row = expected[task_id], actual[task_id]
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete MNIST scale-control training result: {task_id}")
        source_config, result_config = normalize_config(source["config"]), normalize_config(row["config"])
        material = lambda cfg: {key: value for key, value in cfg.items() if key not in {"device", "threads"}}
        if material(source_config) != material(result_config):
            raise ValueError(f"Validation result configuration mismatch: {task_id}")
        for key in ("group", "method_label", "variant_id", "family", "seed"):
            value = source_config["seed"] if key == "seed" else source[key]
            if row.get(key) != value:
                raise ValueError(f"Validation result {key} mismatch: {task_id}")
        for key in ("validation_accuracy", "validation_cross_entropy"):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (key == "validation_accuracy" and value > 1)):
                raise ValueError(f"Invalid validation metric: {task_id}")
        group, label, variant = source["group"], source["method_label"], source["variant_id"]
        counts[group][label] += 1
        seed_sets[group][(label, variant)].add(source_config["seed"])
        output.append({key: copy.deepcopy(row[key]) for key in VALIDATION_FIELDS if key in row})
    if set(counts) != GROUPS:
        raise ValueError("Retain both MNIST deployment laws")
    for group, labels in counts.items():
        if set(labels) != LABELS or set(labels.values()) != {variants_per_label * len(SEEDS["screen"])}:
            raise ValueError(f"Unequal or incomplete method/regime tuning budget: {group}")
        variants = Counter(label for label, _ in seed_sets[group])
        if set(variants.values()) != {variants_per_label}:
            raise ValueError(f"Require {variants_per_label} distinct candidates per method/regime: {group}")
        if any(seeds != set(SEEDS["screen"]) for seeds in seed_sets[group].values()):
            raise ValueError("Every candidate requires the fixed complete screening seeds 72101 and 72102")
    return output


def freeze_original(manifest, training_results, output):
    """Extract a canonical complete original MNIST subset into a separate folder."""
    manifest, training_results, output = Path(manifest), Path(training_results), Path(output)
    require_training_path(training_results)
    if any(output.resolve().is_relative_to(path.parent.resolve()) for path in (manifest, training_results)):
        raise ValueError("Freeze original inputs into a separate owned folder outside both source directories")
    before = {str(path.resolve()): fingerprint(path) for path in (manifest, training_results)}
    entries = original_mnist_entries(read_entries(manifest))
    ids = {item["task_id"] for item in entries}
    source_rows = read_training_records(training_results)
    rows = [row for row in source_rows if row["task_id"] in ids]
    if any(row.get("config", {}).get("dataset") == "mnist" and row["task_id"] not in ids for row in source_rows):
        raise ValueError("Foreign MNIST training result in original source")
    canonical = canonical_validation_rows(entries, rows, 16)
    if before != {str(path.resolve()): fingerprint(path) for path in (manifest, training_results)}:
        raise ValueError("An original screening source changed during freezing; retry a fresh snapshot")
    write_once(output / "original_manifest.jsonl", manifest_text(entries))
    write_once(output / "original_training_results.json", encoded(canonical))
    snapshot = dict(original_tasks=ORIGINAL_TASKS, datasets=["mnist"],
                    candidates_per_method_regime=16, screening_seeds=list(SEEDS["screen"]),
                    source_fingerprints=before, selection_uses_test=False,
                    original_ids_preserved=True, official_test_files_opened=False)
    write_once(output / "original_snapshot.json", encoded(snapshot))
    return snapshot


def screen_plan(original, extra, source_fingerprints):
    return dict(stage="bayes_screen", scientific_setup="mnist_fixed_positive_variance_control",
                tasks=len(extra), original_tasks=len(original), combined_training_tasks=TOTAL_TASKS,
                datasets=["mnist"], channels=list(CHANNELS), labels=sorted(LABELS),
                original_candidates_per_method_regime=16, added_candidates_per_method_regime=16,
                combined_candidates_per_method_regime=32, screening_seeds=list(SEEDS["screen"]),
                change="Only rho_lr_scale changes from .1 to 0; Gaussian posterior_std remains .001 or .003",
                predictor="All methods retain one positive-variance diagonal Gaussian; fixed scale is still Bayesian with finite Gaussian KL",
                fixed_conditions=dict(train_size=55000, val_size=5000, epochs=30, m=20, data_seed=20261117),
                selection="All 1024 paired training/validation records; mean validation accuracy, CE tiebreak, deterministic variant-ID tiebreak",
                source_fingerprints=source_fingerprints, original_ids_preserved=True,
                test_evaluation_during_screen=False, selection_uses_test=False, favorable_seed_selection=False,
                resource_note="Parent coordinator launches separately with measured CPU resources; this builder never submits jobs")


def build_screen(original_manifest, output):
    original_manifest, output = Path(original_manifest), Path(output)
    if output.resolve() == original_manifest.resolve():
        raise ValueError("New manifest must not overwrite the original")
    original = original_mnist_entries(read_entries(original_manifest))
    extra = extra_entries(original)
    plan = screen_plan(original, extra, {str(original_manifest.resolve()): fingerprint(original_manifest)})
    text = manifest_text(extra)
    plan["manifest_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    write_once(output, text)
    write_once(output.with_suffix(".plan.json"), encoded(plan))
    return plan


def combine_and_select(original_entries, original_records, extra, extra_records):
    original = original_mnist_entries(original_entries)
    validate_extra_entries(original, extra)
    original_rows = canonical_validation_rows(original, original_records, 16)
    extra_rows = canonical_validation_rows(extra, extra_records, 16)
    combined = original + copy.deepcopy(extra)
    if len({item["task_id"] for item in combined}) != TOTAL_TASKS:
        raise ValueError("Original and appended training tasks overlap")
    validate_entries(combined)
    rows = canonical_validation_rows(combined, original_rows + extra_rows, 32)
    _, selected = select_variants(rows, combined, 1)
    if len(selected) != 16 or any(item["seeds"] != list(SEEDS["screen"]) for item in selected):
        raise ValueError("Select exactly one complete paired variant per method/regime/law")
    return combined, rows, selected


def build_selected(original_manifest, original_results, extra_manifest, extra_results, output):
    paths = tuple(Path(path) for path in (original_manifest, original_results, extra_manifest, extra_results))
    require_training_path(paths[1])
    require_training_path(paths[3])
    output = Path(output)
    destinations = {output / "combined_manifest.jsonl", output / "validation_selection_results.json",
                    output / "selected_variants.json", output / "selection_protocol.json"}
    if {path.resolve() for path in paths} & {path.resolve() for path in destinations}:
        raise ValueError("Selection outputs must not replace any frozen input")
    before = {str(path.resolve()): fingerprint(path) for path in paths}
    combined, rows, selected = combine_and_select(
        read_entries(paths[0]), read_training_records(paths[1]),
        read_entries(paths[2]), read_training_records(paths[3]))
    manifests = {stage: selected_entries(selected, SEEDS[stage], stage)
                 for stage in ("confirm", "m_sweep", "p_train_sweep")}
    expected_counts = dict(confirm=80, m_sweep=264, p_train_sweep=168)
    for stage, items in manifests.items():
        if len(items) != expected_counts[stage]:
            raise ValueError(f"Unexpected prespecified final-stage coverage: {stage}")
        validate_entries(items)
    if before != {str(path.resolve()): fingerprint(path) for path in paths}:
        raise ValueError("A screening selection input changed while selecting")
    write_once(output / "combined_manifest.jsonl", manifest_text(combined))
    write_once(output / "validation_selection_results.json", encoded(rows))
    write_once(output / "selected_variants.json", encoded(selected))
    protocol = dict(scientific_setup="mnist_fixed_positive_variance_control", original_trials=512,
                    extra_trials=512, combined_trials=TOTAL_TASKS, candidates_per_method_regime=32,
                    selected_variants=len(selected), screening_seeds=list(SEEDS["screen"]),
                    selection_uses_test=False, official_test_files_opened=False,
                    selection="Mean paired validation accuracy; CE then variant-ID tiebreak; original candidate remains eligible",
                    source_fingerprints=before, final_stage_tasks=expected_counts,
                    final_seed_sets={stage: list(SEEDS[stage]) for stage in manifests})
    write_once(output / "selection_protocol.json", encoded(protocol))
    for stage, items in manifests.items():
        text = manifest_text(items)
        plan = make_plan(items, stage, SEEDS[stage], {path: value["sha256"] for path, value in before.items()})
        plan.update(scientific_setup="mnist_fixed_positive_variance_control", datasets=["mnist"],
                    tuning_variants_per_method_regime=32,
                    selection="Original16 plus fixed-positive-scale16; all1024 training/validation records; no test selection",
                    manifest_sha256=hashlib.sha256(text.encode()).hexdigest())
        write_once(output / f"{stage}.jsonl", text)
        write_once(output / f"{stage}.plan.json", encoded(plan))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "screen", "select"), default="screen")
    parser.add_argument("--original-manifest", type=Path, default=Path("journal/configs/bayesian_screen.jsonl"))
    parser.add_argument("--original-results", type=Path)
    parser.add_argument("--extra-manifest", type=Path)
    parser.add_argument("--extra-results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("journal/configs/bayesian_mnist_scale_screen.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "screen":
        result = build_screen(args.original_manifest, args.output)
    elif args.stage == "freeze":
        if args.original_results is None or args.output_dir is None:
            parser.error("freeze requires --original-results and a separate --output-dir")
        result = freeze_original(args.original_manifest, args.original_results, args.output_dir)
    else:
        if any(value is None for value in (args.original_results, args.extra_manifest, args.extra_results, args.output_dir)):
            parser.error("select requires --original-results, --extra-manifest, --extra-results, and --output-dir")
        result = build_selected(args.original_manifest, args.original_results, args.extra_manifest, args.extra_results, args.output_dir)
    print(encoded(result), end="")


if __name__ == "__main__":
    main()
