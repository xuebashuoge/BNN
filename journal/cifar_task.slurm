#!/bin/bash
# One measured allocation processes a CIFAR stage; GPU trials are sequential.
#SBATCH --job-name=tit_cifar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/journal/%x_%j.out
#SBATCH --error=logs/journal/%x_%j.err

set -euo pipefail
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${CIFAR_MANIFEST:?Use journal/submit_cifar.sh}"
: "${CIFAR_OUTPUT:?Use journal/submit_cifar.sh}"
: "${JOURNAL_GIT_SHA:?Submitted Git revision required}"
: "${JOURNAL_MANIFEST_SHA256:?Submitted manifest hash required}"
cd "${SLURM_SUBMIT_DIR}"
# Leave placement across the pooled allocation available to each child.
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${CIFAR_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tit-cifar-${SLURM_JOB_ID}"
python -c 'import os; from journal.run_task import verify_submission_snapshot; verify_submission_snapshot(os.environ["CIFAR_MANIFEST"])'
if (( ${CIFAR_WORKERS:-4} * ${CIFAR_THREADS:-2} > SLURM_CPUS_PER_TASK )); then
    echo "CIFAR worker threads exceed allocated CPUs" >&2
    exit 2
fi
extra_options=()
if [[ "${CIFAR_DEVICE:-cpu}" == cuda ]]; then
    [[ "${CIFAR_WORKERS:-1}" == 1 ]] || { echo "Use one GPU trial at a time" >&2; exit 2; }
    : "${JOURNAL_CPU_PILOT:?A completed matching CPU pilot is required}"
    extra_options+=(--cpu-pilot "$JOURNAL_CPU_PILOT")
fi
/usr/bin/time -v python -u -m journal.campaign \
    --manifest "$CIFAR_MANIFEST" --output-root "$CIFAR_OUTPUT" \
    --workers "${CIFAR_WORKERS:-4}" --threads "${CIFAR_THREADS:-2}" \
    --device "${CIFAR_DEVICE:-cpu}" --evaluate-top "${CIFAR_EVALUATE_TOP:-1}" \
    --eval-draws "${CIFAR_EVAL_DRAWS:-10}" --resume "${extra_options[@]}"
