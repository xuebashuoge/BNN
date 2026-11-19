#!/bin/bash -l
#SBATCH --job-name=bnn_cifar_ptr_m2000
#SBATCH --account=punim2792
#SBATCH --partition=sapphire
#SBATCH --array=0-12%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=00:35:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
data_dir=${1:?Pass the selected CIFAR input directory}
output=${2:?Pass the new run directory}
manifest="$data_dir/manifests/$SLURM_ARRAY_TASK_ID.jsonl"
: "${DATASET:?Set DATASET to the prepared datasets directory}"
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
read -r JOURNAL_GIT_SHA JOURNAL_MANIFEST_SHA256 < <(
    python -c 'import json,sys; protocol=json.load(open(sys.argv[1])); print(protocol["git_revision"],protocol["shards"][sys.argv[2]]["sha256"])' \
      "$data_dir/protocol.json" "$SLURM_ARRAY_TASK_ID"
)
export JOURNAL_GIT_SHA JOURNAL_MANIFEST_SHA256
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
srun --ntasks=1 --cpus-per-task=24 --cpu-bind=cores python -u -m journal.campaign \
    --manifest "$manifest" --output-root "$output/shards/$SLURM_ARRAY_TASK_ID" \
    --workers 12 --threads 2 --evaluate-top 1 --eval-draws 10 --full-risks --resume
