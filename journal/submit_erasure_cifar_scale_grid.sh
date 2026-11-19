#!/bin/bash
# The supplemental grid adds four distinct cases to each nine-candidate erasure screen.
set -euo pipefail
phase=${1:?Use screen or confirm}
prior=${2:?Supply the primary erasure directory with a completed retuning selection}
dest="$prior/scale-grid"
: "${DATASET:?Set DATASET}"
: "${JOURNAL_GIT_SHA:?Set the pinned Git revision}"
cd "$(git rev-parse --show-toplevel)"
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
[[ $prior = /* && -r "$prior/retuned/selection.json" ]]
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
mkdir -p "$dest/logs"
exec 8>"$dest/submit.lock"; flock 8
py=${DIAGONAL_PYTHON:-python}
module=journal.erasure_bank_comparison
worker=journal/full_fading_spartan.slurm
opts=(--partition=sapphire --chdir="$PWD" --exclude=spartan-bm055,spartan-bm142 --cpus-per-task=1 --mem=3G
      --output="$dest/logs/%x_%A_%a.out" --error="$dest/logs/%x_%A_%a.err")
submit() {
    local ref=$1
    shift
    if [[ -s "$ref" ]]; then cat "$ref";return;fi
    local job
    job=$(sbatch --parsable "${opts[@]}" "$@") || return
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]]
    printf '%s\n' "$job" | tee "$ref"
}
if [[ $phase == screen ]];then
    "$py" -m journal.erasure_cifar_scale_grid screen --transfer "$prior" --directory "$dest"
    manifest="$dest/screen.jsonl"; runs="$dest/screen-runs"
elif [[ $phase == confirm ]];then
    "$py" -m journal.erasure_cifar_scale_grid select --transfer "$prior" --directory "$dest"
    if [[ ! -s "$dest/manifest.jsonl" ]];then exit 0;fi
    manifest="$dest/manifest.jsonl";runs="$dest/runs"
else
    exit 2
fi
count=$(wc -l < "$manifest")
export JOURNAL_MANIFEST_SHA256
JOURNAL_MANIFEST_SHA256=$(sha256sum "$manifest" | cut -d' ' -f1)
arr=$(submit "$dest/$phase.job" --job-name="er_cf_scale_$phase" --time=00:35:00 --time-min=00:20:00 \
      --array="0-$((count-1))%${DIAGONAL_PARALLELISM:-64}" "$worker" python -u -m "$module" task \
      --manifest "$manifest" --output-root "$runs" --threads 1 --device cpu)
if [[ $phase == screen ]];then
    submit "$dest/select.job" --job-name=er_cf_scale_select --time=00:05:00 --dependency="afterok:$arr" --kill-on-invalid-dep=yes \
           "$worker" bash journal/submit_erasure_cifar_scale_grid.sh confirm "$prior"
else
    submit "$dest/report.job" --job-name=er_cf_scale_report --time=00:05:00 --dependency="afterok:$arr" --kill-on-invalid-dep=yes \
           "$worker" python -m "$module" report --manifest "$manifest" --output-root "$runs"
fi
