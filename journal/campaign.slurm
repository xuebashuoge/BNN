#!/bin/bash
# A bounded pool avoids waiting for many short array tasks individually.
#SBATCH --job-name=tit_screen
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=logs/journal/%x_%j.out
#SBATCH --error=logs/journal/%x_%j.err

set -euo pipefail
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/data/gpfs/projects/punim2792/anaconda3/envs}"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28
: "${DATASET:?Set the shared dataset root}"
: "${CAMPAIGN_MANIFEST:?Use journal/submit_campaign.sh}"
: "${CAMPAIGN_OUTPUT:?Use journal/submit_campaign.sh}"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${CAMPAIGN_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tit-campaign-${SLURM_JOB_ID}"
python -c 'import os; from journal.run_task import verify_submission_snapshot; verify_submission_snapshot(os.environ["CAMPAIGN_MANIFEST"])'
if [[ -n "${CAMPAIGN_PREPARE_DATASETS:-}" ]]; then
    python -u -m journal.prepare_data --datasets "$CAMPAIGN_PREPARE_DATASETS"
fi
if (( ${CAMPAIGN_WORKERS:-4} * ${CAMPAIGN_THREADS:-2} > SLURM_CPUS_PER_TASK )); then
    echo "Worker thread budget exceeds allocated CPUs" >&2
    exit 2
fi
/usr/bin/time -v python -u -m journal.campaign \
    --manifest "$CAMPAIGN_MANIFEST" --workers "${CAMPAIGN_WORKERS:-4}" \
    --threads "${CAMPAIGN_THREADS:-2}" --output-root "$CAMPAIGN_OUTPUT" \
    --evaluate-top "${CAMPAIGN_EVALUATE_TOP:-1}" --eval-draws "${CAMPAIGN_EVAL_DRAWS:-5}" --resume
