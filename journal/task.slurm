#!/bin/bash
# One independent configuration per task, with no nested process pool.
#SBATCH --job-name=tit_cpu
#SBATCH --output=logs/journal/%x_%A_%a.out
#SBATCH --error=logs/journal/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00

set -euo pipefail
# Noninteractive bash does not inherit the user's .zshrc environment paths.
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set DATASET to the existing shared dataset root}"
: "${JOURNAL_MANIFEST:?Submit with journal/submit.sh}"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tit-mpl-${SLURM_JOB_ID}"
task_device="${JOURNAL_DEVICE:-cpu}"
if [[ "${JOURNAL_PILOT:-0}" == 1 ]]; then
    # Sequential representative CPU trials inside one small allocation.
    for task_index in 0 1 2 3; do
        /usr/bin/time -v python -u -m journal.run_task --manifest "$JOURNAL_MANIFEST" \
            --index "$task_index" --device cpu
    done
else
    /usr/bin/time -v python -u -m journal.run_task --manifest "$JOURNAL_MANIFEST" \
        --device "$task_device"
fi
