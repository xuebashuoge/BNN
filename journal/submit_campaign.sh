#!/bin/bash
# Submit one staged search from committed TIT_gpt code on Spartan.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash journal/submit_campaign.sh MANIFEST OUTPUT [--test-only]" >&2
    exit 2
fi
[[ "${3:-}" == "" || "${3:-}" == --test-only ]] || exit 2
[[ "$(git branch --show-current)" == TIT_gpt ]] || { echo "Use TIT_gpt" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit tracked changes first" >&2; exit 2; }
[[ -f "$1" ]] || { echo "Manifest not found" >&2; exit 2; }
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
python -m journal.run_task --manifest "$1" --validate-only
mkdir -p logs/journal
export CAMPAIGN_MANIFEST="$1"
export CAMPAIGN_OUTPUT="$2"
export JOURNAL_GIT_SHA="$(git rev-parse HEAD)"
export JOURNAL_MANIFEST_SHA256="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1")"
options=(--job-name="${CAMPAIGN_JOB_NAME:-tit_screen}" --time="${CAMPAIGN_TIME:-00:40:00}" \
    --cpus-per-task="${CAMPAIGN_CPUS:-8}" --mem="${CAMPAIGN_MEM:-12G}")
[[ "${3:-}" != --test-only ]] || options+=(--test-only)
sbatch "${options[@]}" journal/campaign.slurm
