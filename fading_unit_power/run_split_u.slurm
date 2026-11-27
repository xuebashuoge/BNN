#!/bin/bash -l
#SBATCH --account=punim2792
#SBATCH --partition=sapphire
#SBATCH --qos=publiccpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
# Usage (via submit_split_u.sh): sbatch ... run_split_u.slurm STUDY ROOT
set -eo pipefail

study=${1:?Pass the study: performance, mismatch or bounds}
root=${2:?Pass the study output directory}
expected_revision=$(cat "$root/git-revision.txt")
if [[ "$(git rev-parse HEAD)" != "$expected_revision" ]]; then
    printf 'Expected worktree revision %s, found %s\n' "$expected_revision" "$(git rev-parse HEAD)" >&2
    exit 1
fi

export DATASET=${DATASET:-/data/gpfs/projects/punim2792/Datasets}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python_bin=${CONDA_PREFIX:?torch28 activation failed}/bin/python
set -u

unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2
export MPLCONFIGDIR=${TMPDIR:-/tmp}/bnn-split-u-${SLURM_JOB_ID}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

# One single-threaded training process per allocated CPU.
exec "$python_bin" -u -m fading_unit_power.split_u pipeline --study "$study" --root "$root" \
    --workers "$SLURM_CPUS_PER_TASK"
