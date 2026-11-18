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
exec bash journal/submit_bank_array.sh journal.fading_severity "$root" "$threads" severe
