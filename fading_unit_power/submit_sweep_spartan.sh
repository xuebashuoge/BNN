#!/bin/bash -l
set -eo pipefail

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s RUN_DIRECTORY [--resume]\n' "$0"
    exit 0
fi
if [[ $# != 1 && ( $# != 2 || $2 != --resume ) ]]; then
    printf 'Usage: %s RUN_DIRECTORY [--resume]\n' "$0" >&2
    exit 2
fi
resume=${2:-}

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
export DATASET=${DATASET:-/data/gpfs/projects/punim2792/Datasets}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python_bin=${CONDA_PREFIX:?torch28 activation failed}/bin/python
set -u

DATASET=$(cd -- "$DATASET" && pwd -P)
if [[ -z "$resume" ]]; then
    mkdir -- "$1"
fi
run_dir=$(cd -- "$1" && pwd -P)
performance_dir=$run_dir/performance
jobs_file=$run_dir/jobs.tsv
mkdir -p -- "$run_dir/logs"
exec 9>"$run_dir/.submit.lock"
flock -n 9 || { printf 'Submission already in progress for %s\n' "$run_dir" >&2; exit 1; }

cd -- "$repo"
revision=$(git rev-parse HEAD)
if [[ -s "$run_dir/git-revision.txt" && $(cat "$run_dir/git-revision.txt") != "$revision" ]]; then
    printf 'Use the campaign revision %s\n' "$(cat "$run_dir/git-revision.txt")" >&2
    exit 1
fi
if [[ -s "$run_dir/dataset-dir.txt" && $(cat "$run_dir/dataset-dir.txt") != "$DATASET" ]]; then
    printf 'Use the campaign dataset directory %s\n' "$(cat "$run_dir/dataset-dir.txt")" >&2
    exit 1
fi
if [[ -s "$jobs_file" ]]; then
    queued=$(squeue --noheader --user="${USER:?}" --format='%i')
    active=$(awk 'NR == FNR { if (FNR > 1) campaign[$2] = 1; next }
        { split($1, id, "_"); if (id[1] in campaign) print id[1] }' \
        "$jobs_file" - <<< "$queued" | sort -u)
    if [[ -n "$active" ]]; then
        printf 'Campaign jobs are still queued or running:\n%s\n' "$active" >&2
        exit 1
    fi
else
    printf 'stage\tjob_id\tdependency\tarray\n' > "$jobs_file"
fi

printf '%s\n' "$revision" > "$run_dir/git-revision.txt"
printf '%s\n' "$DATASET" > "$run_dir/dataset-dir.txt"
if [[ -f "$run_dir/ftz.sha256" ]]; then
    sha256sum --check --status "$run_dir/ftz.sha256"
else
    cc -O2 -shared -fPIC fading_unit_power/trainer/cpu_flush_bootstrap.c -o "$run_dir/libbnn_ftz.so"
    sha256sum "$run_dir/libbnn_ftz.so" > "$run_dir/ftz.sha256"
fi

sweep=("$python_bin" -m fading_unit_power.performance_sweep)
"${sweep[@]}" plan --output "$performance_dir"
refine_count=$("${sweep[@]}" count --phase refine)
confirmation_count=$("${sweep[@]}" count --phase confirmation)

submit() {
    local name=$1 cpus=$2 memory=$3 time=$4 dependency=$5 array=$6
    shift 6
    local -a flags=(--job-name="$name" --cpus-per-task="$cpus" --mem="$memory" --time="$time")
    if [[ -n "$dependency" ]]; then
        flags+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
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
    printf '%s\t%s\t%s\t%s\n' "$name" "$job_id" "$dependency" "$array" | tee -a "$jobs_file" >&2 || return
    printf '%s\n' "$job_id"
}

submit_array() {
    local dataset=$1 phase=$2 count=$3 dependency=$4 index array
    local -a pending=()
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Expected a positive %s task count for %s, got %s\n' "$phase" "$dataset" "$count" >&2
        return 1
    fi
    if [[ -z "$resume" ]]; then
        array="0-$((count - 1))"
    else
        for ((index = 0; index < count; index++)); do
            if [[ ! -f "$run_dir/done/$dataset/$phase/$index" ]]; then
                pending+=("$index")
            fi
        done
        if [[ ${#pending[@]} == 0 ]]; then
            return 0
        fi
        array=$(IFS=,; printf '%s' "${pending[*]}")
    fi
    submit "sweep-$dataset-$phase" 2 "$memory" "$time" "$dependency" "$array%$concurrency" \
        sweep-run "$dataset" "$phase"
}

confirmations=()
for dataset in moons mnist fashionmnist cifar10; do
    # The per-dataset stages run in series and share a 64-CPU limit.
    case "$dataset" in
        moons) time=00:05:00; memory=4G; concurrency=2 ;;
        mnist|fashionmnist) time=00:30:00; memory=4G; concurrency=4 ;;
        cifar10) time=00:50:00; memory=6G; concurrency=22 ;;
    esac
    search_count=$(awk 'END { print NR }' "$performance_dir/manifests/$dataset-search.jsonl")
    search_job=$(submit_array "$dataset" search "$search_count" '')
    # A successful rerun can leave an array's afterok dependency unsatisfied.
    # Consumers of arrays validate the complete set of results before proceeding.
    promote_job=$(submit "sweep-$dataset-promote" 1 4G 00:05:00 "${search_job:+afterany:$search_job}" '' \
        sweep-promote "$dataset" "$refine_count")
    refine_job=$(submit_array "$dataset" refine "$refine_count" "afterok:$promote_job")
    select_dependency="afterok:$promote_job"
    if [[ -n "$refine_job" ]]; then
        select_dependency="afterany:$refine_job"
    fi
    select_job=$(submit "sweep-$dataset-select" 1 4G 00:05:00 "$select_dependency" '' \
        sweep-select "$dataset" "$confirmation_count")
    confirmation_job=$(submit_array "$dataset" confirmation "$confirmation_count" "afterok:$select_job")
    confirmations+=("${confirmation_job:-$select_job}")
done

confirmation_dependencies=$(IFS=:; printf '%s' "${confirmations[*]}")
submit sweep-summary 1 4G 00:05:00 "afterany:$confirmation_dependencies" '' sweep-summarize >/dev/null
printf 'Submitted jobs recorded in %s\n' "$jobs_file"
