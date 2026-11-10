#!/bin/bash
# Run from the repository root on Spartan after pulling TIT_gpt.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash journal/submit.sh pilot|cpu|gpu MANIFEST.jsonl [--test-only]" >&2
    exit 2
fi
profile="$1"
manifest="$2"
dry_run="${3:-}"
[[ "$dry_run" == "" || "$dry_run" == "--test-only" ]] || exit 2
[[ "$(git branch --show-current)" == TIT_gpt ]] || { echo "Use TIT_gpt before submitting" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit tracked changes before submitting" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "Manifest not found: $manifest" >&2; exit 2; }
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python -m journal.run_task --manifest "$manifest" --validate-only
task_count=$(python -c 'import sys; from journal.run_task import read_manifest; print(len(read_manifest(sys.argv[1])))' "$manifest")
mkdir -p logs/journal
export JOURNAL_MANIFEST="$manifest"
export JOURNAL_DEVICE=cpu
export JOURNAL_PILOT=0
options=()
[[ -z "$dry_run" ]] || options+=(--test-only)
case "$profile" in
    pilot)
        [[ "$task_count" == 4 ]] || { echo "Pilot expects exactly four trials" >&2; exit 2; }
        export JOURNAL_PILOT=1
        options+=(--job-name=tit_pilot --partition=interactive --time=00:20:00)
        ;;
    cpu)
        options+=(--array="0-$((task_count - 1))%${JOURNAL_CONCURRENCY:-8}")
        ;;
    gpu)
        # feit is allowed on feit-gpu-a100; public gpu-a100 does not allow feit.
        # Submit only after the same configuration has passed a CPU pilot.
        : "${JOURNAL_CPU_PILOT:?Set to the completed CPU pilot output directory}"
        [[ -f "$JOURNAL_CPU_PILOT/metadata.json" ]] || { echo "CPU pilot metadata not found" >&2; exit 2; }
        export JOURNAL_DEVICE=cuda
        options+=(--job-name=tit_gpu --partition=feit-gpu-a100 --qos=feit --gres=gpu:1 \
            --array="0-$((task_count - 1))%${JOURNAL_CONCURRENCY:-2}")
        ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
sbatch "${options[@]}" journal/task.slurm
