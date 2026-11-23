#!/bin/bash -l
#SBATCH --account=punim2792
#SBATCH --partition=sapphire
#SBATCH --qos=publiccpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
set -eo pipefail

run_dir=${1:?Pass the campaign directory}
stage=${2:?Pass the job stage}
target=${3:-}
detail=${4:-}
expected_revision=$(cat "$run_dir/git-revision.txt")
actual_revision=$(git rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
    printf 'Expected worktree revision %s, found %s\n' "$expected_revision" "$actual_revision" >&2
    exit 1
fi

export DATASET=${DATASET:-/data/gpfs/projects/punim2792/Datasets}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
# Slurm inherits the submit shell's CONDA_PREFIX, but the login module can
# put base Python first on PATH before Conda sees that torch28 is active.
conda deactivate
conda activate torch28
python_bin=${CONDA_PREFIX:?torch28 activation failed}/bin/python
set -u

unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ATEN_CPU_CAPABILITY=avx2
export MPLCONFIGDIR=${TMPDIR:-/tmp}/bnn-fading-${SLURM_JOB_ID}

fixed_dir=$run_dir/fixed
performance_dir=$run_dir/performance
worker=(srun --export=ALL --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores)
if [[ "$stage" == performance || ( "$stage" == fixed && "$target" == mnist ) ]]; then
    sha256sum --check --status "$run_dir/ftz.sha256"
    worker+=(env "LD_PRELOAD=$run_dir/libbnn_ftz.so${LD_PRELOAD:+:$LD_PRELOAD}")
fi

case "$stage" in
    fixed)
        exec "${worker[@]}" "$python_bin" -u -m fading_unit_power.fixed_studies run "$target" \
            --index "$SLURM_ARRAY_TASK_ID" --output-root "$fixed_dir"
        ;;
    collect)
        exec "${worker[@]}" "$python_bin" -u -m fading_unit_power.fixed_studies collect --output-root "$fixed_dir"
        ;;
    performance)
        exec "${worker[@]}" "$python_bin" -u -m fading_unit_power.performance run --output "$performance_dir" \
            --manifest "$performance_dir/manifests/$target-$detail.jsonl" --index "$SLURM_ARRAY_TASK_ID"
        ;;
    select)
        "${worker[@]}" "$python_bin" -u -m fading_unit_power.performance select --output "$performance_dir" --dataset "$target"
        confirmation_count=$(awk 'END { print NR }' "$performance_dir/manifests/$target-confirmation.jsonl")
        if [[ "$confirmation_count" != "$detail" ]]; then
            printf '%s selection wrote %s tasks, expected %s\n' "$target" "$confirmation_count" "$detail" >&2
            exit 1
        fi
        ;;
    summarize)
        selection=()
        if [[ -n "$target" ]]; then
            selection=(--dataset "$target")
        fi
        exec "${worker[@]}" "$python_bin" -u -m fading_unit_power.performance summarize \
            --output "$performance_dir" "${selection[@]}"
        ;;
    *) printf 'Unknown job stage: %s\n' "$stage" >&2; exit 2 ;;
esac
