#!/bin/bash
# Independent fixed p_train stage; publish only its completed whole cache.
#SBATCH --job-name=bnn_parallel_ptr
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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
: "${JOURNAL_GIT_SHA:?Record the external committed source revision}"
: "${PARALLEL_MAIN_ROOT:?Set the running main output root}"
: "${PARALLEL_OUTPUT:?Set a separate owned output root on the same filesystem}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-parallel-p-${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v python -u -m journal.run_parallel_p_train_sweep \
    --main-root "${PARALLEL_MAIN_ROOT}" --main-revision "${PARALLEL_MAIN_REVISION:-f602987}" \
    --output "${PARALLEL_OUTPUT}" --workers 16 --threads 2
