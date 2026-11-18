#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?Supply dataset}
m=${2:?Supply channel-bank size}
root=${3:?Supply absolute output directory}
threads=${4:-1}
[[ $root = /* && $threads =~ ^[1-9][0-9]*$ ]]
: "${DATASET:?Set the dataset root}"
: "${ERASURE_SOURCE:?Set the original all-Bayesian confirmation manifest}"
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)} OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
mkdir -p "$root/logs"
exec 9>"$root/submit.lock"
flock 9
"${DIAGONAL_PYTHON:-python}" -m journal.erasure_bank_comparison build --source "$ERASURE_SOURCE" --dataset "$dataset" --m "$m" --output-dir "$root"
if [[ $dataset != cifar10 ]]; then export BANK_TIME_LIMIT=${BANK_TIME_LIMIT:-00:10:00}; fi
exec bash journal/submit_bank_array.sh journal.erasure_bank_comparison "$root" "$threads" erasure
