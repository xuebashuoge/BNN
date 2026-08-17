#!/bin/bash

#=========================================================
# CPU-only Slurm job for paired ERM/proposed MNIST sweeps
#=========================================================

#SBATCH --job-name=mnist_erm_prop
#SBATCH --output=test17_erm_prop_%j.out
#SBATCH --error=test17_erm_prop_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=24:00:00

set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate torch28

if [[ -z "${DATASET:-}" ]]; then
    echo "ERROR: DATASET must point to the shared dataset directory." >&2
    exit 2
fi

# Every sweep worker is single-threaded; parallelism comes from the process pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "MNIST root: ${DATASET}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Starting paired ERM/proposed training-channel sweeps on CPU..."

python -u test17.py \
    --mode sweep \
    --num-seeds 20 \
    --max-workers "${SLURM_CPUS_PER_TASK}"

if command -v my-job-stats >/dev/null 2>&1; then
    my-job-stats -a -n -s
fi

echo "Job completed successfully."
