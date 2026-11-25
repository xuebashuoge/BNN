#!/bin/bash -l
# Submit the four data-dependent-prior T1 studies; CPUs sum to the 64-CPU cap.
# Usage: bash -l fading_unit_power/submit_data_dependent.sh RUN_DIRECTORY [study ...]
# Rerunning with an existing RUN_DIRECTORY resumes: finished tasks are skipped.
# CONSTRUCTIONS="perfect" (say) reselects among those stage-1 constructions in an
# existing RUN_DIRECTORY, reusing its tuning fits, and publishes under only_<tag>.
set -euo pipefail

run_dir=${1:?Pass the campaign directory}
shift
studies=("$@")
[[ ${#studies[@]} -gt 0 ]] || studies=(performance cifar_mismatch mnist bounds)
declare -A cpus=([performance]=32 [cifar_mismatch]=16 [mnist]=12 [bounds]=4)
declare -A memory=([performance]=128G [cifar_mismatch]=64G [mnist]=32G [bounds]=8G)
declare -A hours=([performance]=48:00:00 [cifar_mismatch]=24:00:00 [mnist]=24:00:00 [bounds]=6:00:00)

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse HEAD)
extra=() tag="" revision_file=git-revision.txt
if [[ -n "${CONSTRUCTIONS:-}" ]]; then
    read -r -a constructions <<< "$CONSTRUCTIONS"
    extra=(--constructions "${constructions[@]}")
    tag=only_$(printf '%s\n' "${constructions[@]}" | sort | paste -sd_ -)
    revision_file=git-revision-$tag.txt
fi
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd -P)
[[ -f "$run_dir/jobs.tsv" ]] || printf 'study\tjob_id\trevision\n' > "$run_dir/jobs.tsv"
for study in "${studies[@]}"; do
    root=$run_dir/$study
    mkdir -p "$root/logs"
    if [[ -f "$root/$revision_file" && "$(cat "$root/$revision_file")" != "$revision" ]]; then
        printf '%s was started at revision %s; check out that revision to resume\n' \
            "$root" "$(cat "$root/$revision_file")" >&2
        exit 1
    fi
    printf '%s\n' "$revision" > "$root/$revision_file"
    job=$(sbatch --parsable --chdir="$repo" --job-name="dd-$study${tag:+-$tag}" --cpus-per-task="${cpus[$study]}" \
        --mem="${memory[$study]}" --time="${hours[$study]}" --output="$root/logs/%x-%j.log" \
        --export=ALL,REVISION_FILE="$revision_file" \
        "$repo/fading_unit_power/run_data_dependent.slurm" "$study" "$root" "${extra[@]}")
    printf '%s\t%s\t%s\n' "$study" "$job" "$revision" | tee -a "$run_dir/jobs.tsv"
done
