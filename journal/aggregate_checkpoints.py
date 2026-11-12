"""Assemble completed checkpoint metadata, preserving actual execution settings.

There is no training or inference fallback. In particular, each checkpoint's
training thread count must match its own explicitly recorded manifest entry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from .campaign import completed_trial, save_json, select_variants, validate_entries
from .run_experiment import normalize_config
from .run_task import read_manifest, verify_submission_snapshot


def aggregate(manifest, output_root):
    manifest, output_root = Path(manifest), Path(output_root)
    verify_submission_snapshot(manifest)
    entries = read_manifest(manifest)
    validate_entries(entries)
    rows = []
    for entry in entries:
        config = normalize_config(entry["config"])
        checkpoint = output_root / entry["task_id"]
        value = completed_trial(checkpoint, config)
        if value is None:
            raise ValueError(f"Incomplete or mismatched checkpoint: {entry['task_id']}")
        rows.append(dict(task_id=entry["task_id"], group=entry["group"],
                         method_label=entry["method_label"], family=entry["family"],
                         variant_id=entry["variant_id"], stage=entry["stage"],
                         role=entry.get("role", "comparison"), seed=config["seed"],
                         config=config, checkpoint=str(checkpoint.resolve()),
                         **value, status="complete", resumed=True))
    verify_submission_snapshot(manifest)
    variants, _ = select_variants(rows, entries, 0)
    # Publish only after every declared case has passed the strict cache gate.
    save_json(output_root / "training_results.json", rows)
    save_json(output_root / "aggregation_snapshot.json", dict(
        manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        completed_trials=len(rows), new_optimization=0, checkpoint_inference=False,
        training_threads_by_task={row["task_id"]: row["config"]["threads"] for row in rows},
        variant_validation_summaries=variants,
        selection="Withheld until separate fresh validation-only Monte Carlo rescoring"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    rows = aggregate(args.manifest, args.output_root)
    print(json.dumps(dict(complete=len(rows), new_optimization=0, inference=False)))


if __name__ == "__main__":
    main()
