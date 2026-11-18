#!/bin/bash -l
#SBATCH --account=punim2792
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
: "${JOURNAL_GIT_SHA:?Missing submitted Git revision}"
: "${DATASET:?Set DATASET to the prepared dataset root}"
test "$(git rev-parse HEAD)" = "$JOURNAL_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28

unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export ATEN_CPU_CAPABILITY=avx2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-fading-${SLURM_JOB_ID}"

exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores "$@"
