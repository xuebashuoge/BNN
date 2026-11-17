#!/bin/bash
# Separate fair MNIST fixed-positive-Gaussian-scale validation control.
#SBATCH --job-name=bnn_mnist_scale
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=02:15:00
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
: "${JOURNAL_GIT_SHA:?Record the committed MNIST control checkout revision}"
: "${MNIST_SCALE_MAIN_ROOT:?Set the original running main campaign output root}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-mnist-scale-${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v python -u -m journal.run_bayesian_mnist_scale \
    --main-root "$MNIST_SCALE_MAIN_ROOT" --main-revision "${MNIST_SCALE_MAIN_REVISION:-f602987}" \
    --output "${MNIST_SCALE_OUTPUT:-results/journal_bayesian_mnist_scale/20261117}" --workers 16 --threads 2
