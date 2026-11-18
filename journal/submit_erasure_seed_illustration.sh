#!/usr/bin/env bash
# Start only when this optional validation-selected illustration is explicitly requested.
# bash journal/submit_erasure_seed_illustration.sh start OUT SOURCE_MANIFEST [SOURCE_MANIFEST ...]
set -euo pipefail
phase=$1; out=$2; shift 2
: "${DATASET:?Set dataset directory}"
cd "$(git rev-parse --show-toplevel)"
export JOURNAL_GIT_SHA=$(git rev-parse HEAD) OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
[[ $out = /* && -d $DATASET && -z $(git status --porcelain --untracked-files=no) ]]
mkdir -p "$out/logs"
worker=journal/full_fading_spartan.slurm
module=journal.erasure_validation_seed_illustration
submit(){
    local name=$1; shift
    (
        flock -x 9
        local path="$out/logs/$name.job" id
        if [[ -s $path ]]; then cat "$path"; exit 0; fi
        id=$(sbatch --parsable --kill-on-invalid-dep=yes --partition=sapphire --cpus-per-task=2 --mem=4G \
            --job-name="er_seed_$name" --output="$out/logs/%x_%A_%a.out" --error="$out/logs/%x_%A_%a.err" "$@")
        id=${id%%;*}; [[ $id =~ ^[0-9]+$ ]]; echo "$id" | tee "$path"
    ) 9>"$out/logs/$name.job.lock"
}
case "$phase" in
 start)
    [[ $# -ge 1 ]]
    submit prepare --time=00:05:00 "$worker" bash journal/submit_erasure_seed_illustration.sh dispatch "$out" "$@"
    ;;
 dispatch)
    src=()
    for path in "$@"; do src+=(--source "$path"); done
    python -m "$module" prepare --output "$out" "${src[@]}"
    n=$(wc -l < "$out/pool.jsonl")
    export JOURNAL_MANIFEST_SHA256=$(sha256sum "$out/pool.jsonl"|cut -d' ' -f1)
    train=$(submit train --time=00:30:00 --array="0-$((n-1))%56" "$worker" python -m "$module" train --output "$out" --threads 2)
    freeze=$(submit freeze --time=00:05:00 --dependency="afterok:$train" "$worker" python -m "$module" freeze --output "$out")
    test=$(submit evaluate --time=00:10:00 --dependency="afterok:$freeze" --array="0-$((n-1))%56" "$worker" python -m "$module" evaluate --output "$out")
    submit report --time=00:08:00 --dependency="afterok:$test" "$worker" python -m "$module" report --output "$out"
    ;;
 *) echo "Use start with an output folder and explicitly designated source confirmation manifests" >&2; exit 2 ;;
esac
