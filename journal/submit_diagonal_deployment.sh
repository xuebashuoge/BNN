#!/usr/bin/env bash
# Run on Spartan from a clean checkout with DATASET exported.
#   bash journal/submit_diagonal_deployment.sh frozen /absolute/stress /old_study /three_bank_study
#   bash journal/submit_diagonal_deployment.sh recovery /absolute/stress /severe_transfer_study/matched
set -euo pipefail
phase=$1
out=$2
shift 2
: "${DATASET:?Set the dataset root}"
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=$(git rev-parse HEAD)
[[ -z $(git status --porcelain --untracked-files=no) && -d $DATASET && $out = /* ]]
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
worker=journal/full_fading_spartan.slurm
script=journal/submit_diagonal_deployment.sh
p=$out/logs
if [[ $phase == *recovery ]]; then p=$out/recovery/logs; fi
mkdir -p "$p"
submit() {
    local name=$1
    shift
    (
        flock -x 9
        local record="$p/$name.job" id
        if [[ -s $record ]]; then cat "$record"; exit 0; fi
        id=$(sbatch --parsable --partition=sapphire --kill-on-invalid-dep=yes --chdir="$PWD" \
             --output="$p/%x_%A_%a.out" --error="$p/%x_%A_%a.err" "$@")
        id=${id%%;*}
        [[ $id =~ ^[0-9]+$ ]]
        echo "$id" | tee "$record"
    ) 9>"$p/$name.job.lock"
}
case "$phase" in
    frozen|recovery)
        submit configure --job-name="stress_$phase" --cpus-per-task=1 --mem=3G --time=00:08:00 \
            "$worker" bash "$script" "schedule-$phase" "$out" "$@"
        ;;
    schedule-frozen)
        old=$1; grid=$2
        python -m unittest journal.test_diagonal_fading_stress journal.test_diagonal_fading journal.test_cifar_search_models journal.test_theorem1
        python -m journal.diagonal_fading_stress prepare --old "$old" --grid "$grid" --output "$out"
        count=$(jq '.tasks|length' "$out/protocol.json")
        job=$(submit eval --job-name=frozen_stress --cpus-per-task=2 --mem=4G --time=00:12:00 \
            --array="0-$((count-1))%112" "$worker" python -m journal.diagonal_fading_stress run --threads 2 --output "$out")
        submit report --job-name=stress_report --cpus-per-task=1 --mem=3G --time=00:06:00 \
            --dependency="afterok:$job" "$worker" python -m journal.diagonal_fading_stress aggregate --output "$out"
        ;;
    schedule-recovery)
        matched=$1
        python -m journal.diagonal_fading_recovery prepare --stress "$out" --matched "$matched"
        jobs=()
        while read -r m dataset; do
            stage=$matched/m$m/$dataset; dep=()
            if [[ ! -s $stage/runs/accuracy_summary.json ]]; then dep=(--dependency="afterok:$(cat "$stage/report.job")"); fi
            count=$(wc -l < "$stage/manifest.jsonl")
            id=$(submit "m$m-$dataset" --job-name="reco-$m-$dataset" --cpus-per-task=2 --mem=4G --time=00:08:00 \
                --array="0-$((count-1))%14" "${dep[@]}" "$worker" python -m journal.diagonal_fading_recovery run \
                --stress "$out" --m "$m" --dataset "$dataset" --threads 2)
            jobs+=("$id")
        done < <(jq -r '.sources[]|[.m,.dataset]|@tsv' "$out/recovery/protocol.json")
        if [[ ! -s $out/summary.json ]]; then jobs+=("$(cat "$out/logs/report.job")"); fi
        deps=$(IFS=:; echo "${jobs[*]}")
        submit report --job-name=recovery_report --cpus-per-task=1 --mem=3G --time=00:06:00 \
            --dependency="afterok:$deps" "$worker" python -m journal.diagonal_fading_recovery aggregate --stress "$out"
        ;;
    *) echo "Expected frozen or recovery" >&2; exit 2 ;;
esac
