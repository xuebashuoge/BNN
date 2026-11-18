#!/usr/bin/env bash
# Paired transfer of a completed mild-law cohort to the stronger link.
set -euo pipefail
source_manifest=${1:?Supply original confirmation manifest}
root=${2:?Supply absolute output directory}
threads=${3:-1}
[[ $source_manifest = /* && $root = /* && -f $source_manifest && $threads =~ ^[1-9][0-9]*$ ]]
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)}
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
: "${DATASET:?Set DATASET}"
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
mkdir -p "$root/logs"
exec 9>"$root/submit.lock"
flock 9
"${DIAGONAL_PYTHON:-python}" -m journal.fading_severity transfer --source "$source_manifest" --output-dir "$root"
manifest="$root/manifest.jsonl"
count=$(wc -l < "$manifest")
export JOURNAL_MANIFEST_SHA256
JOURNAL_MANIFEST_SHA256=$(sha256sum "$manifest" | cut -d' ' -f1)
worker=(journal/full_fading_spartan.slurm python -u -m journal.fading_severity)
common=(--partition=sapphire --chdir="$PWD" --cpus-per-task="$threads" --mem=4G --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err")
submit() {
    local record=$1
    shift
    if [[ -s $record ]]; then cat "$record"; return; fi
    local id
    id=$(sbatch --parsable "${common[@]}" "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\n' "$id" | tee "$record"
}
job=$(submit "$root/train.job" --job-name=severe_train --time=00:35:00 --array="0-$((count-1))%${DIAGONAL_PARALLELISM:-64}" \
      "${worker[@]}" task --manifest "$manifest" --output-root "$root/runs" --threads "$threads" --device cpu)
submit "$root/report.job" --job-name=severe_report --time=00:05:00 --kill-on-invalid-dep=yes --dependency="afterok:$job" \
       "${worker[@]}" report --manifest "$manifest" --output-root "$root/runs"
