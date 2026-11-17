#!/bin/bash
# Test-selected exploratory paired-seed examples; main five-seed means retained.
#SBATCH --job-name=bnn_seed_examples
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:45:00
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
: "${JOURNAL_GIT_SHA:?Record the exploratory evaluator execution commit}"
: "${SEED_EXAMPLES_SOURCE_ROOT:?Set the completed final assembled campaign root}"
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bnn-seed-examples-${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v python -u -m journal.verify_bayesian_seed_examples \
    --source-root "$SEED_EXAMPLES_SOURCE_ROOT" \
    --output "${SEED_EXAMPLES_OUTPUT:-results/journal_bayesian_seed_examples/20261117}" \
    --main-revision "${SEED_EXAMPLES_MAIN_REVISION:-f602987}" --workers 16 --all-settings
