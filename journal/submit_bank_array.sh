#!/usr/bin/env bash
# Execute one dataset/bank-size comparison and aggregate its complete confirmation.
set -euo pipefail
module=${1:?Supply a comparison module}
root=${2:?Supply the absolute result directory}
threads=${3:?Supply CPU thread count}
label=${4:?Supply job label}
manifest="$root/manifest.jsonl"
count=$(wc -l < "$manifest")
export JOURNAL_MANIFEST_SHA256
JOURNAL_MANIFEST_SHA256=$(sha256sum "$manifest" | cut -d' ' -f1)
worker=(journal/full_fading_spartan.slurm python -u -m "$module")
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
job=$(submit "$root/train.job" --job-name="${label}_train" --time="${BANK_TIME_LIMIT:-00:35:00}" --array="0-$((count-1))%${DIAGONAL_PARALLELISM:-64}" \
      "${worker[@]}" task --manifest "$manifest" --output-root "$root/runs" --threads "$threads" --device cpu)
submit "$root/report.job" --job-name="${label}_report" --time=00:05:00 --kill-on-invalid-dep=yes --dependency="afterok:$job" \
       "${worker[@]}" report --manifest "$manifest" --output-root "$root/runs"
