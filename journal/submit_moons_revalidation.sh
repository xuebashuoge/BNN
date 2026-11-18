#!/bin/bash
# Score frozen moons checkpoints, select all seven methods on validation, then evaluate fresh runs.
set -euo pipefail
phase=$1
root=$2
mkdir -p "$root/revalidation/logs"
worker=journal/full_fading_spartan.slurm
script=journal/submit_moons_revalidation.sh
module=journal.diagonal_moons_revalidation
compare=journal.diagonal_fading_comparison
cohort=confirm-revalidated
submit() {
    local label=$1
    shift
    local id
    id=$(sbatch --parsable --partition=sapphire --cpus-per-task=1 --mem=3G --time=00:10:00 \
        --kill-on-invalid-dep=yes --job-name="moon_rv_$label" \
        --output="$root/revalidation/logs/%x_%A_%a.out" --error="$root/revalidation/logs/%x_%A_%a.err" "$@")
    echo "$id" | tee "$root/revalidation/$label.job"
}
case $phase in
    start)
        check=$(submit check "$worker" python -m unittest journal.test_diagonal_moons_revalidation)
        scores=$(submit score --dependency="afterok:$check" --array=0-34%20 "$worker" \
            bash -c 'exec python -m journal.diagonal_moons_revalidation score --index "$SLURM_ARRAY_TASK_ID" --root "$1"' _ "$root")
        submit freeze --dependency="afterok:$scores" "$worker" bash "$script" freeze "$root"
        ;;
    freeze)
        python -m "$module" freeze --root "$root"
        export JOURNAL_MANIFEST_SHA256
        JOURNAL_MANIFEST_SHA256=$(sha256sum "$root/$cohort.jsonl" | cut -d' ' -f1)
        jobs=$(submit confirm --array=0-20 "$worker" python -m "$compare" task \
            --manifest "$root/$cohort.jsonl" --output-root "$root/$cohort" --threads 1)
        submit report --dependency="afterok:$jobs" "$worker" python -m "$compare" report \
            --manifest "$root/$cohort.jsonl" --output-root "$root/$cohort"
        ;;
    *) echo "Use start or freeze" >&2; exit 2 ;;
esac
