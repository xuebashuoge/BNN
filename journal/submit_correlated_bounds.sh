#!/usr/bin/env bash
# One independent training job for each objective, law and width.
set -euo pipefail
root=${1:?Supply an absolute output directory}
rho=${2:-0.15}
scope=${3:-fading}
case $scope in
    fading|erasure) last=9; plot_args=(--family "$scope") ;;
    both) last=19; plot_args=(--both-channels) ;;
    *) echo "Supply fading, erasure or both as the channel scope" >&2; exit 2 ;;
esac
[[ $root = /* ]]
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=$(git rev-parse HEAD)
export CORR_BOUNDS_OUTPUT="$root" CORR_BOUNDS_TRAIN_RHO="$rho" CORR_BOUNDS_SCOPE="$scope"
: "${DATASET:?Set DATASET to the prepared dataset location}"
[[ -z $(git status --porcelain --untracked-files=no) ]]
mkdir -p "$root/logs"
id=$(sbatch --parsable --partition=sapphire --cpus-per-task=1 --mem=3G --time=00:05:00 --array="0-$last" \
    --chdir="$PWD" --job-name=corr_t1_t2 --output="$root/logs/train_%A_%a.out" --error="$root/logs/train_%A_%a.err" \
    journal/full_fading_spartan.slurm bash -c '
        set -eu
        cases=(perfect t1_matched t2_matched t1_mismatch t2_mismatch)
        widths=(16 8)
        i=$SLURM_ARRAY_TASK_ID
        families=(fading erasure)
        family=$CORR_BOUNDS_SCOPE
        args=()
        if [[ $family == both ]]; then
            family=${families[i/10]}
            args=(--both-channels)
        fi
        exec python -u -m journal.correlated_bound_study --width "${widths[(i%10)/5]}" --case "${cases[i%5]}" \
            --output "$CORR_BOUNDS_OUTPUT" --mismatch-rho "$CORR_BOUNDS_TRAIN_RHO" --family "$family" "${args[@]}"')
printf '%s\n' "${id%%;*}" | tee "$root/train.job"
report=$(sbatch --parsable --partition=sapphire --cpus-per-task=1 --mem=3G --time=00:04:00 \
    --chdir="$PWD" --job-name=corr_bound_report --dependency="afterok:${id%%;*}" --kill-on-invalid-dep=yes \
    --output="$root/logs/report_%j.out" --error="$root/logs/report_%j.err" \
    journal/full_fading_spartan.slurm python -m journal.plot_correlated_bound_study --root "$root" --output "$root/figures" "${plot_args[@]}")
printf '%s\n' "${report%%;*}" | tee "$root/report.job"
