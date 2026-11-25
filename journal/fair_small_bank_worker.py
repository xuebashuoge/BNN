"""Execute one fixed fair small-bank trial with resumable, locked reporting.

Screening and resource pilots use validation only. Test inference is available
only to explicitly declared confirmation tasks, after training has completed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .campaign import evaluate_trial, save_json, train_trial, validate_entries
from .full_fading_comparison import validate_result, validate_test
from .run_task import read_manifest, verify_submission_snapshot


STAGES = {"fair_small_bank_screen", "fair_small_bank_confirm", "fair_small_bank_pilot"}


def run_task(manifest: Path, output_root: Path, index: int | None = None,
             threads: int = 2) -> dict:
    manifest, output_root = Path(manifest), Path(output_root)
    verify_submission_snapshot(manifest)
    entries = read_manifest(manifest)
    validate_entries(entries)
    stages = {item.get("stage") for item in entries}
    if len(stages) != 1 or not stages <= STAGES:
        raise ValueError("Require one declared fair small-bank stage per manifest")
    if threads < 1:
        raise ValueError("Thread count must be positive")
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(entries):
        raise ValueError(f"Task index {index} outside a manifest of {len(entries)} entries")
    item = entries[index]
    result_path = output_root / "reports" / (item["task_id"] + ".json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with result_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = dict(task_id=item["task_id"], group=item["group"],
                   method_label=item["method_label"], family=item["family"],
                   variant_id=item["variant_id"], stage=item["stage"],
                   seed=item["config"]["seed"], config=item["config"])
        phase = "train"
        try:
            row = train_trial(item, output_root, threads, resume=True, device="cpu")
            row["source_manifest_sha256"] = manifest_sha256
            if row["status"] != "complete":
                raise RuntimeError(row.get("error", "Training did not complete"))
            validate_result(item, row)
            save_json(result_path, row)
            if item["stage"] == "fair_small_bank_confirm":
                phase = "test"
                row = evaluate_trial(row, row["config"]["eval_draws"],
                                     full_risks=False, resume=True)
                if row["test_status"] != "complete":
                    raise RuntimeError(row.get("test_error", "Test inference did not complete"))
                row = validate_test(item, row, output_root)
            save_json(result_path, row)
        except Exception as error:
            row["source_manifest_sha256"] = manifest_sha256
            detail = f"{type(error).__name__}: {error}"
            if phase == "test":
                row.update(test_status="failed", test_error=detail)
            else:
                row.update(status="failed", error=detail)
            save_json(result_path, row)
            raise
    return {key: row[key] for key in (
        "task_id", "status", "validation_accuracy", "test_status",
        "test_accuracy", "seconds", "resumed") if key in row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_task(args.manifest, args.output_root, args.index,
                              args.threads), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
