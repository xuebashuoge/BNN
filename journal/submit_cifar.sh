#!/bin/bash
# Run from the committed TIT_gpt checkout on Spartan.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash journal/submit_cifar.sh pilot|cpu|array|gpu MANIFEST OUTPUT [--test-only]" >&2
    exit 2
fi
profile="$1"
manifest="$2"
output="$3"
[[ "${4:-}" == "" || "${4:-}" == --test-only ]] || exit 2
[[ "$(git branch --show-current)" == TIT_gpt ]] || { echo "Use TIT_gpt" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit tracked changes first" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "Manifest not found: $manifest" >&2; exit 2; }
: "${DATASET:?Set the shared dataset root}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python -m journal.campaign --manifest "$manifest" --output-root "$output" --validate-only
mkdir -p logs/journal
export CIFAR_MANIFEST="$manifest" CIFAR_OUTPUT="$output"
export JOURNAL_GIT_SHA="$(git rev-parse HEAD)"
export JOURNAL_MANIFEST_SHA256="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$manifest")"
export CIFAR_DEVICE=cpu
export CIFAR_WORKERS="${CIFAR_WORKERS:-4}"
export CIFAR_THREADS="${CIFAR_THREADS:-2}"
options=(--job-name="${CIFAR_JOB_NAME:-tit_cifar}")
task_script=journal/cifar_task.slurm
[[ "${4:-}" != --test-only ]] || options+=(--test-only)
case "$profile" in
    pilot)
        options+=(--partition=interactive --time="${CIFAR_TIME:-00:15:00}" \
            --cpus-per-task="${CIFAR_CPUS:-8}" --mem="${CIFAR_MEM:-12G}")
        export CIFAR_EVALUATE_TOP=0
        ;;
    cpu)
        options+=(--partition="${CIFAR_CPU_PARTITION:-interactive}" \
            --time="${CIFAR_TIME:-00:30:00}" --cpus-per-task="${CIFAR_CPUS:-8}" \
            --mem="${CIFAR_MEM:-12G}")
        export CIFAR_EVALUATE_TOP="${CIFAR_EVALUATE_TOP:-1}"
        ;;
    array)
        task_count=$(python -c 'import sys; from journal.run_task import read_manifest; print(len(read_manifest(sys.argv[1])))' "$manifest")
        array_cap="${CIFAR_ARRAY_CONCURRENCY:-12}"
        [[ "$array_cap" =~ ^[0-9]+$ ]] && (( array_cap > 0 )) || { echo "CIFAR_ARRAY_CONCURRENCY must be positive" >&2; exit 2; }
        options+=(--partition="${CIFAR_CPU_PARTITION:-sapphire}" \
            --time="${CIFAR_TIME:-00:25:00}" --cpus-per-task="${CIFAR_CPUS:-2}" \
            --mem="${CIFAR_MEM:-3G}" --array="0-$((task_count-1))%${array_cap}")
        task_script=journal/cifar_array.slurm
        ;;
    gpu)
        : "${JOURNAL_CPU_PILOT:?Set the completed matching CPU pilot root}"
        : "${CIFAR_TIME:?Set GPU walltime from the CPU pilot estimate}"
        : "${CIFAR_MEM:?Set host memory from the CPU pilot peak RSS}"
        python -m journal.run_task --manifest "$manifest" --validate-only --cpu-pilot "$JOURNAL_CPU_PILOT"
        export JOURNAL_CPU_PILOT CIFAR_DEVICE=cuda CIFAR_WORKERS=1
        export CIFAR_THREADS="${CIFAR_GPU_THREADS:-4}"
        export CIFAR_EVALUATE_TOP="${CIFAR_EVALUATE_TOP:-1}"
        options+=(--partition=feit-gpu-a100 --qos=feit --gres=gpu:1 \
            --cpus-per-task="${CIFAR_GPU_CPUS:-4}" \
            --time="$CIFAR_TIME" --mem="$CIFAR_MEM")
        ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
sbatch "${options[@]}" "$task_script"
