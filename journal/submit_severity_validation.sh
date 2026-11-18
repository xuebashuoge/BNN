#!/usr/bin/env bash
# Run additional hard-law validation and a fresh paired confirmation.
set -euo pipefail
source_selection=${1:?Supply a mild-law selection JSON or the additional grid name erasure or mnist_hard}
primary=${2:?Supply the hard primary-transfer directory}
phase=${3:-screen}
threads=${4:-1}
[[ $primary == /* && $threads =~ ^[1-9][0-9]*$ ]]
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)}
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
: "${DATASET:?Set DATASET to the prepared data}"
py=${DIAGONAL_PYTHON:-python}
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
case $source_selection in
    erasure) runner=journal.erasure_bank_comparison; base="$primary"; limit=00:10:00; job_label=erasure ;;
    mnist_hard) runner=journal.fading_severity; base="$primary/expansion"; limit=00:08:00; job_label=mnist_hard ;;
    /*) runner=journal.fading_severity; base="$primary"; limit=00:35:00; job_label=hard ;;
    *) echo "Expected an absolute source selection or a declared expansion name" >&2; exit 2 ;;
esac
validation="$base/validation"
case $phase in
    screen) out=$validation ;;
    confirm) out="$base/retuned" ;;
    *) exit 2 ;;
esac
mkdir -p "$out/logs"
exec 9>"$out/submit.lock"
flock 9
manifest="$out/manifest.jsonl"
selector=journal.severity_validation
if [[ $source_selection == erasure || $source_selection == mnist_hard ]]; then
    "$py" -m journal.additional_bank_validation "$source_selection" "$phase" --root "$primary"
elif [[ $phase == screen ]]; then
    "$py" -m "$selector" screen --source-selection "$source_selection" --primary "$primary/manifest.jsonl" --output "$manifest"
else
    "$py" -m "$selector" confirm --screen "$validation/manifest.jsonl" --runs "$validation/runs" --output "$manifest"
fi
export JOURNAL_MANIFEST_SHA256="$(sha256sum "$manifest" | cut -d' ' -f1)"
common=(--partition=sapphire --chdir="$PWD" --cpus-per-task="$threads" --mem=4G --kill-on-invalid-dep=yes
        --output="$out/logs/%x_%A_%a.out" --error="$out/logs/%x_%A_%a.err")
submit() {
    local record=$1 id
    shift
    if [[ -s $record ]]; then
        id=$(cat "$record")
        [[ $id =~ ^[0-9]+$ ]] || return 1
    else
        id=$(sbatch --parsable "${common[@]}" "$@") || return 1
        id=${id%%;*}
        [[ $id =~ ^[0-9]+$ ]] || return 1
        printf '%s\n' "$id" > "$record"
    fi
    printf '%s\n' "$id"
}
worker=journal/full_fading_spartan.slurm
n=$(wc -l < "$manifest")
job=$(submit "$out/train.job" --job-name="${job_label}_$phase" --time="$limit" --array="0-$((n-1))%${DIAGONAL_PARALLELISM:-64}" \
    "$worker" python -u -m "$runner" task --manifest "$manifest" --output-root "$out/runs" --threads "$threads" --device cpu)
if [[ $phase == screen ]]; then
    submit "$out/select.job" --job-name="${job_label}_select" --time=00:07:00 --dependency="afterok:$job" \
        "$worker" bash journal/submit_severity_validation.sh "$source_selection" "$primary" confirm "$threads"
else
    submit "$out/report.job" --job-name="${job_label}_report" --time=00:05:00 --dependency="afterok:$job" \
        "$worker" python -u -m "$runner" report --manifest "$manifest" --output-root "$out/runs"
fi
