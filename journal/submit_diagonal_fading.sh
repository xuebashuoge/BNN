#!/usr/bin/env bash
# Launch independent simulations in Slurm arrays and aggregate only complete cohorts.
set -euo pipefail
phase=${1:-all}
dataset=${2:-}
cd "$(git rev-parse --show-toplevel)"
: "${DIAGONAL_OUTPUT:?Set DIAGONAL_OUTPUT to an absolute shared results directory}"
: "${DATASET:?Set DATASET to the prepared dataset directory}"
[[ $DIAGONAL_OUTPUT = /* && -d $DATASET ]]
export JOURNAL_GIT_SHA=${JOURNAL_GIT_SHA:-$(git rev-parse HEAD)}
[[ $(git rev-parse HEAD) == "$JOURNAL_GIT_SHA" && -z $(git status --porcelain --untracked-files=no) ]]
root=$DIAGONAL_OUTPUT
mkdir -p "$root/logs"
py=${DIAGONAL_PYTHON:-python}
threads=${DIAGONAL_THREADS:-2}
parallelism=${DIAGONAL_PARALLELISM:-64}
worker=journal/full_fading_spartan.slurm
launcher=journal/submit_diagonal_fading.sh
mnist=journal.diagonal_fading_mnist_sweep
bounds=journal.diagonal_fading_bounds
compare=journal.diagonal_fading_comparison
cpu=(--partition=sapphire --cpus-per-task="$threads" --mem=4G)
compute=("${cpu[@]}")
device=cpu
if [[ $dataset == cifar10 && ${DIAGONAL_CIFAR_DEVICE:-cpu} == cuda ]]; then
    device=cuda
    compute=(--partition=feit-gpu-a100 --qos=feit --gres=gpu:1 --cpus-per-task="$threads" --mem=6G)
fi

submit() {
    local record=$1 job=$2
    shift 2
    if [[ -s $record ]]; then cat "$record"; return; fi
    local id
    id=$(sbatch --parsable --chdir="$PWD" --job-name="diag_$job" --kill-on-invalid-dep=yes \
        --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err" "$@")
    id=${id%%;*}
    printf '%s\n' "$id" | tee "$record"
}

array() {
    local stage=$1 label=$2 module=$3 dest=$4 limit=$5
    local manifest="$stage/$label.jsonl" count sha
    count=$(wc -l < "$manifest")
    sha=$(sha256sum "$manifest")
    JOURNAL_MANIFEST_SHA256=${sha%% *} submit "$stage/$label.job" "${label}_${dataset:-mnist}" \
        "${compute[@]}" --time="$limit" --array="0-$((count-1))%$parallelism" "$worker" \
        python -u -m "$module" task --manifest "$manifest" --output-root "$dest" --threads "$threads" --device "$device"
}

case $phase in
    all)
        bash "$launcher" sweep
        bash "$launcher" bounds
        for ds in mnist fashionmnist cifar10; do bash "$launcher" screen "$ds"; done
        ;;
    sweep|extended)
        stage="$root/$phase"
        mkdir -p "$stage"
        options=() inputs=() deps=()
        if [[ $phase == extended ]]; then
            options=(--extended)
            inputs=(--manifest "$root/sweep/sweep.jsonl" --stage-root "$root/sweep")
            deps=("$(cat "$root/sweep/sweep.job")")
        fi
        manifest="$stage/$phase.jsonl"
        if [[ ! -s $manifest ]]; then "$py" -m "$mnist" build --output "$manifest" "${options[@]}"; fi
        job=$(array "$stage" "$phase" "$mnist" "$stage" 00:08:00)
        deps+=("$job")
        wait_for=$(IFS=:; echo "${deps[*]}")
        submit "$stage/report.job" mnist_plot "${cpu[@]}" --time=00:05:00 --dependency="afterok:$wait_for" \
            "$worker" python -u -m "$mnist" report "${inputs[@]}" --manifest "$manifest" --stage-root "$stage" --output "$stage/figures"
        ;;
    bounds)
        stage="$root/bounds"
        mkdir -p "$stage"
        job=$(submit "$stage/tasks.job" bounds "${cpu[@]}" --time=00:08:00 --array=0-5 \
            "$worker" bash "$launcher" bound-task)
        submit "$stage/report.job" bounds_plot "${cpu[@]}" --time=00:15:00 --dependency="afterok:$job" \
            "$worker" bash "$launcher" bound-report
        ;;
    bound-task)
        families=(fading erasure)
        regimes=(perfect matched mismatch)
        i=${SLURM_ARRAY_TASK_ID:?}
        python -u -m "$bounds" run --output "$root/bounds" --width 16 --m 2000 \
            --family "${families[i/3]}" --regime "${regimes[i%3]}"
        ;;
    bound-report)
        python -u -m "$bounds" aggregate --output "$root/bounds" --width 16
        python -u -m "$bounds" fallback --output "$root/bounds"
        ;;
    screen|refine|confirm|confirm-refine)
        case $dataset in mnist|fashionmnist|cifar10|moons) ;; *) echo "Supply comparison dataset" >&2; exit 2;; esac
        stage="$root/comparison/$dataset"
        mkdir -p "$stage"
        limit=00:12:00
        if [[ $dataset == cifar10 ]]; then limit=00:30:00; fi
        if [[ $phase == screen || $phase == refine ]]; then
            if [[ $phase == screen ]]; then
                "$py" -m "$compare" screen --datasets "$dataset" --m "${DIAGONAL_M:-2000}" --output "$stage/screen.jsonl"
                next=confirm
            else
                "$py" -m "$compare" refine-screen --screen-manifest "$stage/screen.jsonl" --output "$stage/refine.jsonl"
                next=confirm-refine
            fi
            job=$(array "$stage" "$phase" "$compare" "$stage/$phase" "$limit")
            depends="$job"
            if [[ $phase == refine ]]; then depends+=":$(cat "$stage/screen.job")"; fi
            submit "$stage/$phase-select.job" "select_$dataset" "${cpu[@]}" --time=00:05:00 --dependency="afterok:$depends" \
                "$worker" bash "$launcher" "$next" "$dataset"
        else
            extra=()
            if [[ $phase == confirm-refine ]]; then
                extra=(--refinement-manifest "$stage/refine.jsonl" --refinement-root "$stage/refine")
            fi
            python -m "$compare" confirm --screen-manifest "$stage/screen.jsonl" --screen-root "$stage/screen" \
                "${extra[@]}" --output "$stage/$phase.jsonl"
            job=$(array "$stage" "$phase" "$compare" "$stage/$phase" "$limit")
            submit "$stage/$phase-report.job" "report_$dataset" "${cpu[@]}" --time=00:05:00 --dependency="afterok:$job" \
                "$worker" python -m "$compare" report --manifest "$stage/$phase.jsonl" --output-root "$stage/$phase"
        fi
        ;;
    *) echo "Unknown phase $phase" >&2; exit 2;;
esac
