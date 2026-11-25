#!/bin/bash
# Checkpoint inference only; four two-thread workers share one CPU allocation.
# Set EMPIRICAL_QUEUE to the immutable, complete 51-run table queue.
#SBATCH --job-name=journal_empirical
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/journal/%x_%j.out
#SBATCH --error=logs/journal/%x_%j.err

set -euo pipefail
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${EMPIRICAL_QUEUE:?Set the complete checkpoint inference queue}"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

python -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

queue = Path(os.environ["EMPIRICAL_QUEUE"])
raw = queue.read_bytes()
entries = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
if int(os.environ["SLURM_CPUS_PER_TASK"]) < 8:
    raise ValueError("Four two-thread workers require eight allocated CPUs")
report_dir = Path("results/journal_empirical") / os.environ["SLURM_JOB_ID"]
report_dir.mkdir(parents=True, exist_ok=False)
fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (queue, Path("journal/evaluate_empirical_bank.py"),
                             Path("journal/empirical_queue.slurm"))}
(report_dir / "sources.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
(report_dir / "queue.jsonl").write_bytes(raw)
# Every process reads the same retained queue snapshot, including completed runs.
snapshot = report_dir / "queue.jsonl"

def run(index):
    entry = entries[index]
    start = time.monotonic()
    with (report_dir / f"run_{index}.log").open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "journal.evaluate_empirical_bank",
             "--queue", str(snapshot), "--index", str(index), "--draws", "10",
             "--threads", "2"], stdout=stream, stderr=subprocess.STDOUT)
    return dict(index=index, task_id=entry["task_id"], exit_code=result.returncode,
                seconds=time.monotonic() - start)

results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(run, index) for index in range(len(entries))]):
        result = future.result()
        results.append(result)
        results.sort(key=lambda item: item["index"])
        (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(dict(completed=len(results), total=len(entries), **result)), flush=True)
if any(result["exit_code"] for result in results):
    raise SystemExit("Some empirical evaluations failed; inspect the retained per-run logs")
PY
