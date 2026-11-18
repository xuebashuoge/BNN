#!/usr/bin/env bash
# Run additional hard-law validation and a fresh paired confirmation.
set -euo pipefail
source_selection=${1:?Supply the mild-law selection JSON}
primary=${2:?Supply the hard primary-transfer directory}
phase=${3:-screen}
threads=${4:-1}
[[ $source_selection == /* && $primary == /* && $threads =~ ^[1-9][0-9]*$ ]]
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)}
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
: "${DATASET:?Set DATASET to the prepared data}"
py=${DIAGONAL_PYTHON:-python}
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
validation="$primary/validation"
case $phase in
    screen) out=$validation ;;
    confirm) out="$primary/retuned" ;;
    *) exit 2 ;;
esac
mkdir -p "$out/logs"
exec 9>"$out/submit.lock"
flock 9
manifest="$out/manifest.jsonl"
selector=journal.severity_validation
runner=journal.fading_severity
if [[ $phase == screen ]]; then
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
job=$(submit "$out/train.job" --job-name="hard_$phase" --time=00:35:00 --array="0-$((n-1))%${DIAGONAL_PARALLELISM:-64}" \
    "$worker" python -u -m "$runner" task --manifest "$manifest" --output-root "$out/runs" --threads "$threads" --device cpu)
if [[ $phase == screen ]]; then
    submit "$out/select.job" --job-name=hard_select --time=00:07:00 --dependency="afterok:$job" \
        "$worker" bash journal/submit_severity_validation.sh "$source_selection" "$primary" confirm "$threads"
else
    submit "$out/report.job" --job-name=hard_report --time=00:05:00 --dependency="afterok:$job" \
        "$worker" python -u -m "$runner" report --manifest "$manifest" --output-root "$out/runs"
fi
