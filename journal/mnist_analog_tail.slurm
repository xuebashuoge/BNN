#!/bin/bash -l
#SBATCH --job-name=bnn_mnist_tail
#SBATCH --account=punim2792
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$(git rev-parse HEAD)" = "$JOURNAL_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
srun --ntasks=1 --cpus-per-task=24 --cpu-bind=cores     python -u -m journal.campaign --manifest "$BNN_TAIL_OUTPUT/manifest.jsonl"     --output-root "$BNN_TAIL_OUTPUT/runs" --workers 12 --threads 2     --evaluate-top 1 --eval-draws 10 --full-risks --resume
