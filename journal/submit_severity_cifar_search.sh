#!/usr/bin/env bash
# Screen nine configurations per label, then confirm the frozen selection on five fresh seeds.
set -euo pipefail
phase=${1:?Use screen or confirm}
source_dir=${2:?Supply the transfer directory}
dest=${3:?Supply the retuning output directory}
threads=${4:-1}
[[ "$source_dir" = /* && "$dest" = /* && -s "$source_dir/transfer.json" && $threads =~ ^[1-9][0-9]*$ ]]
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)}
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
: "${DATASET:?Set DATASET to the prepared dataset root}"
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
mkdir -p "$dest/logs"
exec 9>"$dest/submit.lock"
flock 9
worker=journal/full_fading_spartan.slurm
module=journal.fading_severity
common=(--partition=sapphire --chdir="$PWD" --cpus-per-task="$threads" --mem=4G --output="$dest/logs/%x_%A_%a.out" --error="$dest/logs/%x_%A_%a.err")
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
if [[ $phase == screen ]]; then
    "${DIAGONAL_PYTHON:-python}" -m journal.fading_severity_cifar_search screen --transfer-dir "$source_dir" --root "$dest"
    manifest="$dest/screen.jsonl"
    runs="$dest/screen-runs"
elif [[ $phase == confirm ]]; then
    "${DIAGONAL_PYTHON:-python}" -m journal.fading_severity_cifar_search select --transfer-dir "$source_dir" --root "$dest"
    manifest="$dest/manifest.jsonl"
    runs="$dest/runs"
else
    echo "Expected screen or confirm" >&2
    exit 2
fi
count=$(wc -l < "$manifest")
export JOURNAL_MANIFEST_SHA256
JOURNAL_MANIFEST_SHA256=$(sha256sum "$manifest" | cut -d' ' -f1)
job=$(submit "$dest/$phase.job" --job-name="severe_cf_$phase" --time=00:35:00 --array="0-$((count-1))%${DIAGONAL_PARALLELISM:-96}" \
      "$worker" python -u -m "$module" task --manifest "$manifest" --output-root "$runs" --threads "$threads" --device cpu)
if [[ $phase == screen ]]; then
    submit "$dest/select.job" --job-name=severe_cf_select --time=00:05:00 --kill-on-invalid-dep=yes --dependency="afterok:$job" \
           "$worker" bash journal/submit_severity_cifar_search.sh confirm "$source_dir" "$dest" "$threads"
else
    submit "$dest/report.job" --job-name=severe_cf_report --time=00:05:00 --kill-on-invalid-dep=yes --dependency="afterok:$job" \
           "$worker" python -u -m "$module" report --manifest "$manifest" --output-root "$runs"
fi
