"""Train one fixed contiguous CPU slice of an immutable campaign manifest.

Each Slurm array element owns four manifest positions. Cached trials retain
their positions; the master manifest is never filtered or rewritten. Reports
belong to ``pool_reports/batch_i`` and shared trial directories have unique IDs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from .campaign import save_json, train_trial, validate_entries
from .run_task import read_manifest, verify_submission_snapshot


def batch_count(total: int, batch_size: int = 4) -> int:
    if (not isinstance(total, int) or isinstance(total, bool) or total < 1 or
            not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("Require positive manifest length and batch size")
    return (total + batch_size - 1) // batch_size


def select_batch(entries, index: int, batch_size: int = 4):
    count = batch_count(len(entries), batch_size)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
        raise ValueError(f"Batch index {index} outside {count} manifest batches")
    start = index * batch_size
    return entries[start:start + batch_size]


def allowed_cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def validate_cpu_budget(workers: int, threads: int) -> int:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (workers, threads)):
        raise ValueError("Workers and threads must be positive integers")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if allocated < 1 or workers * threads > allocated:
        raise ValueError("Pool worker threads exceed allocated CPUs")
    permitted = allowed_cpus()
    if permitted is not None and workers * threads > len(permitted):
        raise ValueError("Pool worker threads exceed the process CPU affinity mask")
    return allocated


def load_entries(manifest, verify=True):
    if verify:
        verify_submission_snapshot(manifest)
    entries = read_manifest(manifest)
    validate_entries(entries)
    if verify:
        # Catch a changed checkout/manifest before any trial can start.
        verify_submission_snapshot(manifest)
    return entries


def prepare_reports(directory, snapshot):
    """Refuse another batch identity; retain prior attempts when resuming."""
    directory = Path(directory)
    path = directory / "batch_snapshot.json"
    if path.exists():
        previous = json.loads(path.read_text())
        identity = ("manifest_sha256", "batch_index", "batch_size", "task_ids")
        if any(previous.get(key) != snapshot[key] for key in identity):
            raise ValueError("Pool reports belong to a different manifest/batch; use a fresh output root")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = directory / "previous_attempts" / f"{stamp}_{os.getpid()}"
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("batch_snapshot.json", "training_results.json"):
            old = directory / name
            if old.exists():
                shutil.copy2(old, archive / name)
        snapshot["previous_attempt_reports"] = str(archive)
    save_json(path, snapshot)
    save_json(directory / "training_results.json", [])


def run_batch(manifest, output_root, index, batch_size=4, workers=4, threads=2):
    before_import = allowed_cpus()
    allocated = validate_cpu_budget(workers, threads)
    entries = load_entries(manifest)
    validate_cpu_budget(workers, threads)
    selected = select_batch(entries, index, batch_size)
    report_dir = Path(output_root) / "pool_reports" / f"batch_{index}"
    positions = {entry["task_id"]: offset for offset, entry in enumerate(selected)}
    snapshot = dict(manifest=str(Path(manifest).resolve()),
                    manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    submitted_git_sha=os.environ.get("JOURNAL_GIT_SHA"),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    batch_index=index, batch_size=batch_size,
                    master_entries=len(entries), manifest_start=index * batch_size,
                    manifest_stop=index * batch_size + len(selected),
                    task_ids=[entry["task_id"] for entry in selected],
                    workers=workers, threads=threads, allocated_cpus=allocated,
                    device="cpu", resume=True, cache_filtering=False,
                    cpu_affinity_before_numerical_import=before_import,
                    cpu_affinity_before_training_children=allowed_cpus(),
                    environment={key: os.environ.get(key) for key in (
                        "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
                        "KMP_AFFINITY", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST")})
    prepare_reports(report_dir, snapshot)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
        futures = {pool.submit(train_trial, entry, output_root, threads,
                               resume=True, device="cpu"): entry for entry in selected}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                row = future.result()
            except Exception as error:
                # Retain an unexpected launcher failure without dropping other
                # entries or moving them into a different batch.
                row = dict(task_id=entry["task_id"], group=entry["group"],
                           method_label=entry["method_label"], family=entry["family"],
                           variant_id=entry["variant_id"], stage=entry["stage"],
                           role=entry.get("role", "comparison"),
                           seed=entry["config"]["seed"],
                           config={**entry["config"], "device": "cpu", "threads": threads},
                           checkpoint=str((Path(output_root) / entry["task_id"]).resolve()),
                           status="failed", resumed=False,
                           error=f"{type(error).__name__}: {error}")
            results.append(row)
            results.sort(key=lambda item: positions[item["task_id"]])
            save_json(report_dir / "training_results.json", results)
            print(json.dumps(dict(batch_index=index, finished=len(results), total=len(selected),
                                  task_id=row["task_id"], status=row["status"],
                                  resumed=row.get("resumed", False))), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="results/journal")
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        allocated = validate_cpu_budget(args.workers, args.threads)
        entries = load_entries(args.manifest, verify=False)
        print(json.dumps(dict(tasks=len(entries), batches=batch_count(len(entries), args.batch_size),
                              batch_size=args.batch_size, workers=args.workers,
                              threads=args.threads, allocated_cpus=allocated)))
        return 0
    index = args.batch_index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    results = run_batch(args.manifest, args.output_root, index,
                        args.batch_size, args.workers, args.threads)
    return int(any(row["status"] != "complete" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
