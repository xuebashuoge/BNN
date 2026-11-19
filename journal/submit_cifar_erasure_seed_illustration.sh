#!/bin/bash
# Run from a clean Spartan checkout after all four p=.50 validation choices freeze.
# bash journal/submit_cifar_erasure_seed_illustration.sh start ERASURE_ROOT
set -euo pipefail
phase=$1; base=$2; out=$base/seed_illustrations/cifar10
cd "$(git rev-parse --show-toplevel)"
: "${DATASET:?Set dataset root}"
export JOURNAL_GIT_SHA=$(git rev-parse HEAD) OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
export FULL_FADING_FTZ_LIBRARY=${CIFAR_SEED_FTZ_LIBRARY:-/data/gpfs/projects/punim2792/yangshuo/matrix_cpu_runtime_20261118/libbnn_ftz.so}
export FULL_FADING_FTZ_SHA256=5c2a8753f9eb46cdf1d039266ee3463c4bcff4534ba1a4d652c3a61cbb79cc3f
[[ $(sha256sum "$FULL_FADING_FTZ_LIBRARY" | cut -d' ' -f1) == "$FULL_FADING_FTZ_SHA256" ]]
[[ $base = /* && -d $DATASET && -z $(git status --porcelain --untracked-files=no) ]]
[[ -L $out && $(readlink -f "$out") == /home/yangshuo/Git/bnn_task_inode_recovery_20261119/new_seed_runs/cifar10 ]]
mkdir -p "$out/logs"
limit=${CIFAR_SEED_ARRAY_CAP:-56}
[[ $limit =~ ^[0-9]+$ ]] && (( limit >= 1 && limit <= 168 ))
worker=journal/full_fading_spartan.slurm
module=journal.erasure_validation_seed_illustration
submit(){
    local name=$1; shift
    (
        flock -x 9
        local path="$out/logs/$name.job" id
        if [[ -s $path ]]; then cat "$path"; exit 0; fi
        id=$(sbatch --parsable --kill-on-invalid-dep=yes --partition=sapphire --cpus-per-task=1 --mem=3G \
            --job-name="er_cf_seed_$name" --output="$out/logs/%x_%A_%a.out" --error="$out/logs/%x_%A_%a.err" "$@")
        id=${id%%;*}; [[ $id =~ ^[0-9]+$ ]]; echo "$id" | tee "$path"
    ) 9>"$out/logs/$name.job.lock"
}
# Use native confirmation only for the optional-work stopping condition.
# Configurations, the eight seeds and all four sizes are fixed independently.
all_confirmations_lead=true
for size in 200 500 1000 2000; do
    prior=$base/p050/m$size/cifar10/retuned/runs/accuracy_summary.json
    if [[ ! -s $prior ]] || ! jq -e '.confirmation_seeds==[174101,174102,174103,174104,174105]
            and (.groups|length)==1 and ( .groups[0].mean_ranking.margin > 1e-12 )' "$prior" >/dev/null; then
        all_confirmations_lead=false
    fi
done
if [[ $all_confirmations_lead == true ]] && [[ ! -s $out/logs/train.job ]]; then
    echo "Skipped unstarted illustration: all four five-seed confirmations already led."
    exit 0
fi
case "$phase" in
 start)
    for m in 200 500 1000 2000; do
        test -s "$base/p050/m$m/cifar10/retuned/manifest.jsonl"
        test -s "$base/p050/m$m/cifar10/retuned/selection.json"
    done
    submit prepare --time=00:04:00 "$worker" bash journal/submit_cifar_erasure_seed_illustration.sh dispatch "$base"
    ;;
 dispatch)
    src=()
    for m in 200 500 1000 2000; do
        path=$base/p050/m$m/cifar10/retuned
        # This illustration uses exactly nine validation candidates per label and five original confirmation seeds.
        jq -e '(.deployment_law=={"kind":"outage","outage_prob":0.5}) and .candidates_per_method==9 and
               .screening_seeds==[173101,173102,173103] and .confirmation_seeds==[174101,174102,174103,174104,174105] and
               (.selected_variants|length)==7 and (.validation_trials|group_by(.method_label)|length)==7 and
               all((.validation_trials|group_by(.method_label))[]; length==9)' "$path/selection.json" >/dev/null
        jq -se --argjson m "$m" 'length==35 and all(.[]; .stage=="erasure_cifar_retuned" and .role=="erasure_confirmation"
            and .config.dataset=="cifar10" and .config.m==$m and .config.epochs==25 and .config.cpu_flush_denormal==true)' "$path/manifest.jsonl" >/dev/null
        src+=(--source "$path/manifest.jsonl")
    done
    python -m "$module" prepare --pool original --output "$out" "${src[@]}"
    python - "$out" "$limit" <<'PY'
import json, os, sys
from pathlib import Path
from journal.full_fading_comparison import write_once
from journal.run_experiment import file_fingerprint
out=Path(sys.argv[1]); library=Path(os.environ["FULL_FADING_FTZ_LIBRARY"])
runtime=dict(code=os.environ["JOURNAL_GIT_SHA"], backend="cpu", cores_per_task=1, array_cap=int(sys.argv[2]),
             verified_ftz_library=str(library), ftz_library=file_fingerprint(library),
             partition="sapphire", cpu_capability="avx2", OMP_WAIT_POLICY=os.environ["OMP_WAIT_POLICY"],
             KMP_BLOCKTIME=os.environ["KMP_BLOCKTIME"], train_time="00:35:00", train_backfill_floor="00:25:00",
             test_time="00:12:00")
write_once(out/"runtime.json",json.dumps(runtime,indent=2)+"\n")
PY
    test "$(wc -l < "$out/pool.jsonl")" = 224
    export JOURNAL_MANIFEST_SHA256=$(sha256sum "$out/pool.jsonl"|cut -d' ' -f1)
    # The flush-to-zero bootstrap checksum is also enforced by the worker at each start.
    job=$(submit train --time=00:35:00 --time-min=00:25:00 --array="0-223%$limit" "$worker" python -m "$module" train --output "$out" --threads 1)
    freeze=$(submit freeze --time=00:05:00 --dependency="afterok:$job" "$worker" python -m "$module" freeze --output "$out")
    test_job=$(submit evaluate --time=00:12:00 --array="0-223%$limit" --dependency="afterok:$freeze" "$worker" python -m "$module" evaluate --output "$out")
    submit report --time=00:06:00 --dependency="afterok:$test_job" "$worker" python -m "$module" report --output "$out"
    ;;
 *) exit 2 ;;
esac
