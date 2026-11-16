#!/bin/bash
# Run in a separate clean checkout of codex/all-bayesian-bounds.
#SBATCH --job-name=bnn_bayesian
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/journal/%x_%j.out
#SBATCH --error=logs/journal/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${BAYESIAN_OUTPUT:?Set a fresh results directory}"
: "${JOURNAL_GIT_SHA:?Export the committed Git revision before submitting}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${BAYESIAN_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-bayesian-${SLURM_JOB_ID}"
if (( ${BAYESIAN_WORKERS:-32} * ${BAYESIAN_THREADS:-2} > SLURM_CPUS_PER_TASK )); then
    echo "Bayesian worker threads exceed allocated CPUs" >&2
    exit 2
fi
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    /usr/bin/time -v python -u -m journal.run_bayesian_campaign \
    --output "$BAYESIAN_OUTPUT" --workers "${BAYESIAN_WORKERS:-32}" \
    --threads "${BAYESIAN_THREADS:-2}" --confirm-draws "${BAYESIAN_CONFIRM_DRAWS:-20}" \
    --sweep-draws "${BAYESIAN_SWEEP_DRAWS:-10}"
