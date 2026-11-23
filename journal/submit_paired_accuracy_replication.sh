#!/usr/bin/env bash
# From the Spartan checkout: bash journal/submit_paired_accuracy_replication.sh /absolute/output
# Run the same command after a failed allocation to resume its fixed seed/method shards.
set -euo pipefail

module=journal.paired_accuracy_replication
if [[ ${1:-} == --shard ]]; then
    root=${2:?Missing output root}
    channel=${3:?Missing channel}
    shard=${SLURM_ARRAY_TASK_ID:?A Slurm array is required}
    [[ $shard =~ ^[0-9]+$ ]] && (( shard < 50 )) || exit 2
    case $channel in
        fading) offset=0; threads=2 ;;
        erasure) offset=200; threads=1 ;;
        *) exit 2 ;;
    esac
    [[ ${SLURM_CPUS_PER_TASK:?} == "$threads" ]]
    # The manifest holds 50 seed/method pairs for each dataset, channel by channel.
    # Reuse the allocation sequentially, with one CIFAR-10 fit per shard.
    for dataset_offset in 0 50 100 150; do
        python -u -m "$module" task --output-root "$root" --index "$((offset + dataset_offset + shard))"
    done
    exit
fi

root=${1:?Supply an absolute output directory on shared Spartan storage}
[[ $# == 1 && $root == /* ]] || { echo 'Supply one absolute output directory' >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
: "${DATASET:?Set DATASET to the prepared shared dataset directory}"
[[ -d $DATASET ]] || { echo "Dataset directory missing: $DATASET" >&2; exit 2; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Submit from a clean committed checkout' >&2; exit 2; }

export JOURNAL_GIT_SHA JOURNAL_MANIFEST_SHA256
JOURNAL_GIT_SHA=$(git rev-parse HEAD)
export FULL_FADING_FTZ_LIBRARY=${FULL_FADING_FTZ_LIBRARY:-/data/gpfs/projects/punim2792/yangshuo/matrix_cpu_runtime_20261118/libbnn_ftz.so}
export FULL_FADING_FTZ_SHA256=${FULL_FADING_FTZ_SHA256:-5c2a8753f9eb46cdf1d039266ee3463c4bcff4534ba1a4d652c3a61cbb79cc3f}
actual=$(sha256sum "$FULL_FADING_FTZ_LIBRARY")
[[ ${actual%% *} == "$FULL_FADING_FTZ_SHA256" ]] || { echo 'CPU bootstrap checksum mismatch' >&2; exit 1; }
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
python_bin=${DIAGONAL_PYTHON:-/data/gpfs/projects/punim2792/anaconda3/envs/torch28/bin/python}
state="$root/slurm"
mkdir -p "$state/logs"
exec 9>"$state/submission.lock"
flock -x 9

# The lock and job ledger prevent concurrent launchers from doubling the CPU budget.
ledger="$state/jobs.tsv"
if [[ -s $ledger ]]; then
    active=$(squeue --noheader --user="$(id -un)" --format='%i' |
        awk 'NR==FNR {submitted[$2]=1; next} {split($1, a, "_"); if (a[1] in submitted) print a[1]}' "$ledger" - | sort -u)
    if [[ -n $active ]]; then
        printf 'Active replication job IDs\n%s\n' "$active"
        exit 0
    fi
fi

"$python_bin" -m "$module" build --output-root "$root"
actual=$(sha256sum "$root/manifest.jsonl")
JOURNAL_MANIFEST_SHA256=${actual%% *}

submit() {
    local label=$1 id
    shift
    id=$(sbatch --parsable --chdir="$PWD" --job-name="paired_$label" --kill-on-invalid-dep=yes \
        --output="$state/logs/%x_%A_%a.out" --error="$state/logs/%x_%A_%a.err" "$@") || return 1
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $id" >&2; return 1; }
    printf '%s\t%s\n' "$label" "$id" >> "$ledger"
    printf '%s\n' "$id"
}

# Recheck each existing result before skipping its shard on a resubmission.
shards() {
    "$python_bin" - "$root" "$1" <<'PY'
import json
from pathlib import Path
import sys
from journal.paired_accuracy_replication import validate_evaluation

root = Path(sys.argv[1])
offset = {"fading": 0, "erasure": 200}[sys.argv[2]]
entries = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
if len(entries) != 400:
    raise ValueError("Expected 400 frozen training tasks")
pending = []
for shard in range(50):
    tasks = [entries[offset + dataset_offset + shard] for dataset_offset in (0, 50, 100, 150)]
    for task in tasks:
        try:
            validate_evaluation(task, root / "runs" / task["task_id"])
        except (OSError, ValueError, KeyError, TypeError):
            pending.append(str(shard))
            break
print(",".join(pending))
PY
}

worker=journal/full_fading_spartan.slurm
dependencies=()
for channel in fading erasure; do
    pending=$(shards "$channel")
    [[ -n $pending ]] || continue
    # 16 two-core allocations plus 32 single-core allocations use at most 64 CPUs.
    if [[ $channel == fading ]]; then threads=2; cap=16; else threads=1; cap=32; fi
    job=$(submit "$channel" --cpus-per-task="$threads" --mem=4G --time=00:30:00 \
        --array="$pending%$cap" "$worker" bash journal/submit_paired_accuracy_replication.sh --shard "$root" "$channel")
    dependencies+=("$job")
    printf '%s array %s\n' "$channel" "$job"
done
after=()
if (( ${#dependencies[@]} )); then
    dependency=$(IFS=:; echo "${dependencies[*]}")
    after=(--dependency="afterok:$dependency")
fi
job=$(submit report --cpus-per-task=1 --mem=3G --time=00:05:00 "${after[@]}" \
    "$worker" python -u -m journal.report_paired_accuracy_replication "$root")
printf 'Report job %s\nJob ledger %s\n' "$job" "$ledger"
