#!/bin/bash -l
# Submit the split-U studies from a detached worktree pinned to this checkout's HEAD.
# Usage: bash -l fading_unit_power/submit_split_u.sh RUN_DIRECTORY [study ...]
# The home quota on Spartan is full, so the worktree lives beside RUN_DIRECTORY.
# Rerunning with an existing RUN_DIRECTORY resumes: finished tasks are skipped.
set -euo pipefail

run_dir=${1:?Pass the campaign directory}
shift
studies=("$@")
[[ ${#studies[@]} -gt 0 ]] || studies=(performance mismatch bounds)
declare -A cpus=([performance]=40 [mismatch]=16 [bounds]=40 [boost]=64)
declare -A memory=([performance]=160G [mismatch]=64G [bounds]=96G [boost]=256G)
declare -A hours=([performance]=48:00:00 [mismatch]=24:00:00 [bounds]=24:00:00 [boost]=72:00:00)

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse HEAD)
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd -P)
worktree=$(dirname -- "$run_dir")/bnn-split-u-${revision:0:7}
[[ -d "$worktree" ]] || git -C "$repo" worktree add --detach "$worktree" "$revision"
[[ -f "$run_dir/jobs.tsv" ]] || printf 'study\tjob_id\trevision\n' > "$run_dir/jobs.tsv"
for study in "${studies[@]}"; do
    root=$run_dir/$study
    mkdir -p "$root/logs"
    if [[ -f "$root/git-revision.txt" && "$(cat "$root/git-revision.txt")" != "$revision" ]]; then
        printf '%s was started at revision %s; check out that revision to resume\n' \
            "$root" "$(cat "$root/git-revision.txt")" >&2
        exit 1
    fi
    printf '%s\n' "$revision" > "$root/git-revision.txt"
    job=$(sbatch --parsable --chdir="$worktree" --job-name="su-$study" --cpus-per-task="${cpus[$study]}" \
        --mem="${memory[$study]}" --time="${hours[$study]}" --output="$root/logs/%x-%j.log" \
        "$worktree/fading_unit_power/run_split_u.slurm" "$study" "$root")
    printf '%s\t%s\t%s\n' "$study" "$job" "$revision" | tee -a "$run_dir/jobs.tsv"
done
