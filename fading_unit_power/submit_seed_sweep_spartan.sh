#!/bin/bash -l
set -euo pipefail

primary=$(cd -- "${1:?Pass the primary campaign directory}" && pwd -P)
output=${2:?Pass a new seed sweep directory}
repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
export DATASET
DATASET=$(cat "$primary/dataset-dir.txt")

mkdir -- "$output"
run_dir=$(cd -- "$output" && pwd -P)
mkdir -- "$run_dir/logs"
git -C "$repo" rev-parse HEAD > "$run_dir/git-revision.txt"
sha256sum --check --status "$primary/ftz.sha256"
cp -- "$primary/libbnn_ftz.so" "$run_dir/libbnn_ftz.so"
sha256sum "$run_dir/libbnn_ftz.so" > "$run_dir/ftz.sha256"
printf 'stage\tjob_id\tdependency\tarray\n' > "$run_dir/jobs.tsv"

submit() {
    local name=$1 dependency=$2 array=$3 cpus=$4 time=$5
    shift 5
    local -a flags=(--job-name="$name" --dependency="$dependency" --kill-on-invalid-dep=yes
                    --cpus-per-task="$cpus" --mem=6G --time="$time")
    if [[ -n "$array" ]]; then
        flags+=(--array="$array" --output="$run_dir/logs/$name-%A_%a.log")
    else
        flags+=(--output="$run_dir/logs/$name-%j.log")
    fi
    local response job_id
    response=$(sbatch --parsable --chdir="$repo" "${flags[@]}" \
        "$repo/fading_unit_power/run_spartan.slurm" "$run_dir" "$@") || return
    job_id=${response%%;*}
    [[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'Unexpected sbatch response: %s\n' "$response" >&2; return 1; }
    printf '%s\t%s\t%s\t%s\n' "$name" "$job_id" "$dependency" "$array" | tee -a "$run_dir/jobs.tsv" >&2
    printf '%s\n' "$job_id"
}

evaluations=()
for dataset in moons mnist fashionmnist cifar10; do
    # Wait for this dataset to release its CPU allocation in the primary sweep.
    confirmation=$(awk -v name="sweep-$dataset-confirmation" '$1 == name { id = $2 } END { print id }' "$primary/jobs.tsv")
    [[ "$confirmation" =~ ^[0-9]+$ ]] || { printf 'Missing primary confirmation job for %s\n' "$dataset" >&2; exit 1; }
    concurrency=4
    [[ "$dataset" != moons ]] || concurrency=2
    plan=$(submit "seed-$dataset-plan" "afterany:$confirmation" '' 1 00:05:00 seed-plan "$dataset" "$primary/performance")
    evaluations+=("$(submit "seed-$dataset-test" "afterok:$plan" "0-3%$concurrency" 2 00:15:00 seed-run "$dataset")")
done
dependencies=$(IFS=:; printf '%s' "${evaluations[*]}")
submit seed-summary "afterany:$dependencies" '' 1 00:05:00 seed-summary >/dev/null
printf 'Submitted seed sweep jobs recorded in %s/jobs.tsv\n' "$run_dir"
