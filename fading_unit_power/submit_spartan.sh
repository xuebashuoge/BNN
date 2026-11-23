#!/bin/bash -l
set -eo pipefail

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s NEW_RUN_DIRECTORY [all|fixed|performance]\n' "$0"
    printf '       %s NEW_RUN_DIRECTORY refine SOURCE_PERFORMANCE_DIRECTORY DATASET\n' "$0"
    exit 0
fi
scope=${2:-all}
case "$scope" in
    all|fixed|performance) [[ $# -ge 1 && $# -le 2 ]] || exit 2 ;;
    refine) [[ $# == 4 ]] || exit 2 ;;
    *) printf 'Unknown run scope: %s\n' "$scope" >&2; exit 2 ;;
esac

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
export DATASET=${DATASET:-/data/gpfs/projects/punim2792/Datasets}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python_bin=${CONDA_PREFIX:?torch28 activation failed}/bin/python
set -u

[[ -d "$DATASET" ]] || { printf 'Dataset directory missing: %s\n' "$DATASET" >&2; exit 1; }
DATASET=$(cd -- "$DATASET" && pwd -P)
mkdir -- "$1"
run_dir=$(cd -- "$1" && pwd -P)
mkdir -- "$run_dir/logs"
cd -- "$repo"

git rev-parse HEAD > "$run_dir/git-revision.txt"
cc -O2 -shared -fPIC fading_unit_power/trainer/cpu_flush_bootstrap.c -o "$run_dir/libbnn_ftz.so"
sha256sum "$run_dir/libbnn_ftz.so" > "$run_dir/ftz.sha256"
printf 'stage\tjob_id\tdependency\tarray\n' > "$run_dir/jobs.tsv"

submit() {
    local name=$1 cpus=$2 time=$3 dependency=$4 array=$5
    shift 5
    local -a flags=(--job-name="$name" --cpus-per-task="$cpus" --time="$time")
    if [[ -n "$dependency" ]]; then
        flags+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    fi
    if [[ -n "$array" ]]; then
        flags+=(--array="$array" --output="$run_dir/logs/$name-%A_%a.log")
    else
        flags+=(--output="$run_dir/logs/$name-%j.log")
    fi

    local response job_id
    response=$(sbatch --parsable --chdir="$repo" "${flags[@]}" \
        fading_unit_power/run_spartan.slurm "$run_dir" "$@") || return
    job_id=${response%%;*}
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        printf 'Unexpected sbatch response for %s: %s\n' "$name" "$response" >&2
        return 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$name" "$job_id" "$dependency" "$array" | tee -a "$run_dir/jobs.tsv" >&2 || return
    printf '%s\n' "$job_id"
}

array_range() {
    local count=$1 concurrency=$2
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Expected a positive task count, got %s\n' "$count" >&2
        return 1
    fi
    printf '0-%s%%%s\n' "$(( count - 1 ))" "$concurrency"
}

if [[ "$scope" == all || "$scope" == fixed ]]; then
    mnist_count=$("$python_bin" -m fading_unit_power.fixed_studies count mnist)
    mnist_array=$(array_range "$mnist_count" 8)
fi

if [[ "$scope" != fixed ]]; then
    performance_dir=$run_dir/performance
    if [[ "$scope" == refine ]]; then
        datasets=("$4")
        "$python_bin" -m fading_unit_power.performance plan-refinement --output "$performance_dir" \
            --source "$3" --dataset "$4"
    else
        datasets=(moons mnist fashionmnist cifar10)
        "$python_bin" -m fading_unit_power.performance plan --output "$performance_dir"
    fi
    confirmation_count=$("$python_bin" -m fading_unit_power.performance confirmation-count)
fi

if [[ "$scope" == all || "$scope" == fixed ]]; then
    mnist_job=$(submit fixed-mnist 2 00:08:00 '' "$mnist_array" fixed mnist)
    submit fixed-collect 1 00:05:00 "$mnist_job" '' collect >/dev/null
fi

if [[ "$scope" != fixed ]]; then
    confirmations=()
    for dataset in "${datasets[@]}"; do
        case "$dataset" in
            moons) time=00:04:00; concurrency=3 ;;
            mnist|fashionmnist) time=00:08:00; concurrency=3 ;;
            cifar10) time=00:20:00; concurrency=12 ;;
        esac
        if [[ "$scope" == refine ]]; then
            concurrency=4
        fi
        search_count=$(awk 'END { print NR }' "$performance_dir/manifests/$dataset-search.jsonl")
        search_array=$(array_range "$search_count" "$concurrency")
        confirmation_array=$(array_range "$confirmation_count" "$concurrency")
        search_job=$(submit "$dataset-search" 2 "$time" '' "$search_array" performance "$dataset" search)
        select_job=$(submit "$dataset-select" 1 00:05:00 "$search_job" '' select "$dataset" "$confirmation_count")
        confirmations+=("$(submit "$dataset-confirm" 2 "$time" "$select_job" "$confirmation_array" performance "$dataset" confirmation)")
    done
    confirmation_dependencies=$(IFS=:; printf '%s' "${confirmations[*]}")
    summarize=(summarize)
    if [[ "$scope" == refine ]]; then
        summarize+=("$4")
    fi
    submit fading-summary 1 00:05:00 "$confirmation_dependencies" '' "${summarize[@]}" >/dev/null
fi

printf 'Submitted jobs recorded in %s/jobs.tsv\n' "$run_dir"
