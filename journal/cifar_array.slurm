#!/bin/bash
# Short independent CPU trials can backfill separately on Sapphire.
#SBATCH --job-name=tit_cifar_array
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:25:00
#SBATCH --output=logs/journal/%x_%A_%a.out
#SBATCH --error=logs/journal/%x_%A_%a.err

set -euo pipefail
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${CIFAR_MANIFEST:?Use journal/submit_cifar.sh array}"
: "${CIFAR_OUTPUT:?Use journal/submit_cifar.sh array}"
: "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"
: "${JOURNAL_GIT_SHA:?Submitted Git revision required}"
: "${JOURNAL_MANIFEST_SHA256:?Submitted manifest hash required}"
cd "${SLURM_SUBMIT_DIR}"
# Use Slurm core binding without inherited OpenMP placement hints.
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tit-cifar-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
# run_task verifies the immutable submitted checkout/manifest, records the
# actual thread count, and refuses to overwrite an existing checkpoint.
# This stage trains only. Aggregate/evaluate later with campaign --resume and
# --threads matching this allocation (two for the default profile).
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=verbose,cores \
    /usr/bin/time -v python -u -m journal.run_task \
    --manifest "$CIFAR_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$CIFAR_OUTPUT" --device cpu
