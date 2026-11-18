#!/bin/bash
# Usage: bash journal/submit_erasure_mnist_precision.sh schedule ERASURE_ROOT
set -euo pipefail
phase=$1; base=$2; out=$base/penalty_precision/mnist
cd "$(git rev-parse --show-toplevel)"
: "${DATASET:?Set dataset root}"
export JOURNAL_GIT_SHA=$(git rev-parse HEAD) OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
[[ $base = /* && -d $DATASET && -s $out/declaration.json && -z $(git status --porcelain --untracked-files=no) ]]
w=journal/full_fading_spartan.slurm
module=journal.erasure_mnist_precision
submit(){
    local name=$1
    shift
    (
        flock -x 9
        local rec="$out/logs/$name.job" id
        if [[ -s $rec ]]; then cat "$rec"; exit 0; fi
        id=$(sbatch --parsable --kill-on-invalid-dep=yes --partition=sapphire --cpus-per-task=2 --mem=3G \
            --open-mode=append --job-name="pr50_$name" --output="$out/logs/$name.out" --error="$out/logs/$name.err" "$@")
        id=${id%%;*}; [[ $id =~ ^[0-9]+$ ]]; echo "$id"|tee "$rec"
    ) 9>"$out/logs/$name.job.lock"
}
case "$phase" in
 schedule)
    for m in 200 500 1000 2000; do
        root=$base/p050/m$m/mnist/penalty60
        dep=()
        if [[ ! -s $root/retuned/manifest.jsonl || ! -s $root/retuned/train.job ]]; then
            dep=(--dependency="afterok:$(cat "$root/validation/select.job")")
        fi
        submit "pin$m" --time=00:03:00 "${dep[@]}" "$w" bash journal/submit_erasure_mnist_precision.sh pin "$base" "$m"
    done
    ;;
 pin)
    m=$3
    root=$base/p050/m$m/mnist/penalty60/retuned
    python -m "$module" pin --base "$base" --out "$out" --m "$m"
    dep=()
    if [[ ! -s $root/runs/accuracy_summary.json ]]; then dep=(--dependency="afterok:$(cat "$root/report.job")"); fi
    id=$(submit "eval$m" --time=00:04:00 --array=0-62%24 "${dep[@]}" "$w" python -m "$module" run --out "$out" --m "$m")
    submit "report$m" --time=00:02:00 --dependency="afterok:$id" "$w" python -m "$module" report --out "$out" --m "$m"
    ;;
 *) exit 2 ;;
esac
