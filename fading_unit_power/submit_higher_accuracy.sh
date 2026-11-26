#!/bin/bash -l
# Submit the smaller-shared-m deployment comparison (fading_unit_power/higher_accuracy.py).
# Usage: [M=200] bash -l fading_unit_power/submit_higher_accuracy.sh RUN_DIRECTORY [STAGE1_SOURCE ...]
# STAGE1_SOURCE roots hold finished fresh stage-1 fits to link (default: the published
# data-dependent-prior campaign). M is the shared bank size (default 20). Rerunning resumes.
set -euo pipefail

run_dir=${1:?Pass the campaign directory}
shift
stage1=("$@")
[[ ${#stage1[@]} -gt 0 ]] || stage1=(/data/gpfs/projects/punim2792/yangshuo/data-dependent-prior-20261126/performance)
m=${M:-20}
repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse HEAD)
mkdir -p "$run_dir/logs"
run_dir=$(cd -- "$run_dir" && pwd -P)
if [[ -f "$run_dir/git-revision.txt" && "$(cat "$run_dir/git-revision.txt")" != "$revision" ]]; then
    printf '%s was started at revision %s; check out that revision to resume\n' \
        "$run_dir" "$(cat "$run_dir/git-revision.txt")" >&2
    exit 1
fi
printf '%s\n' "$revision" > "$run_dir/git-revision.txt"
[[ -f "$run_dir/jobs.tsv" ]] || printf 'job_id\trevision\n' > "$run_dir/jobs.tsv"
job=$(sbatch --parsable --chdir="$repo" --job-name="higher-accuracy-m$m" --cpus-per-task=64 --mem=256G \
    --time=24:00:00 --output="$run_dir/logs/%x-%j.log" \
    "$repo/fading_unit_power/run_higher_accuracy.slurm" "$run_dir" --m "$m" --reuse-stage1 "${stage1[@]}")
printf '%s\t%s\n' "$job" "$revision" | tee -a "$run_dir/jobs.tsv"
