#!/bin/bash
# Separate small-model bound diagnostic; the main campaign runs elsewhere.
#SBATCH --job-name=bnn_bayes_cert
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:15:00
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
: "${JOURNAL_GIT_SHA:?Record submitted source revision}"
: "${JOURNAL_MANIFEST_SHA256:?Record submitted manifest hash}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-bayesian-cert-${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    /usr/bin/time -v python -u -m journal.run_bayesian_certificate \
    --output results/journal_bayesian_certificate/20261117 --workers 4 --threads 2
