#!/bin/bash
# Each array allocation trains four fixed manifest entries in parallel.
#SBATCH --job-name=tit_cifar_pool
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/journal/%x_%A_%a.out
#SBATCH --error=logs/journal/%x_%A_%a.err

set -euo pipefail
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${CIFAR_MANIFEST:?Use journal/submit_cifar.sh pool-array}"
: "${CIFAR_OUTPUT:?Use journal/submit_cifar.sh pool-array}"
: "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"
: "${JOURNAL_GIT_SHA:?Submitted Git revision required}"
: "${JOURNAL_MANIFEST_SHA256:?Submitted manifest hash required}"
cd "${SLURM_SUBMIT_DIR}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${CIFAR_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tit-cifar-pool-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
if (( ${CIFAR_WORKERS:-4} * ${CIFAR_THREADS:-2} > SLURM_CPUS_PER_TASK )); then
    echo "CIFAR pool worker threads exceed allocated CPUs" >&2
    exit 2
fi
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=verbose,cores \
    /usr/bin/time -v python -u -m journal.batch_campaign \
    --manifest "$CIFAR_MANIFEST" --output-root "$CIFAR_OUTPUT" \
    --batch-index "$SLURM_ARRAY_TASK_ID" --batch-size 4 \
    --workers "${CIFAR_WORKERS:-4}" --threads "${CIFAR_THREADS:-2}"
