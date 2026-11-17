#!/bin/bash
# Read-only checkpoint/report assembly; no training or new Monte Carlo.
#SBATCH --job-name=bnn_final_report
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/journal/%x_%j.out
#SBATCH --error=logs/journal/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${FINAL_MAIN_ROOT:?Set the completed main campaign root}"
: "${FINAL_MNIST_ROOT:?Set the completed MNIST control root}"
: "${FINAL_OUTPUT:?Set a fresh report output root}"
: "${JOURNAL_GIT_SHA:?Record the committed reporting revision}"
[[ "$(git rev-parse HEAD)" == "$JOURNAL_GIT_SHA" ]] || { echo 'Report checkout changed after submission' >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-final-report-${SLURM_JOB_ID}"
/usr/bin/time -v python -u -m journal.assemble_bayesian_results \
    --main-root "$FINAL_MAIN_ROOT" --mnist-root "$FINAL_MNIST_ROOT" \
    --output "$FINAL_OUTPUT" --run-bounds --run-plots
