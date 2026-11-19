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
# Let OpenMP workers sleep between small tensor operations on shared nodes.
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE} KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
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
    local id
    if [[ -s $record ]]; then
        id=$(cat "$record")
        if [[ $id =~ ^[0-9]+$ ]]; then printf '%s\n' "$id"; return; fi
        if [[ -n $id ]]; then echo "Invalid Slurm ID in $record" >&2; return 1; fi
    fi
    id=$(sbatch --parsable --chdir="$PWD" --job-name="diag_$job" --kill-on-invalid-dep=yes \
        --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err" "$@") || return 1
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID: $id" >&2; return 1; }
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
        for ds in mnist fashionmnist cifar10 moons; do bash "$launcher" screen "$ds"; done
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
    bounds|bounds-by-method)
        bash journal/submit_correlated_bounds.sh "$root/bounds-by-method" "${DIAGONAL_MISMATCH_RHO:-0.15}" both
        ;;
    bounds-legacy)
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
    screen|refine|balanced|upper|posterior|finalists|confirm|confirm-refine|confirm-balanced|confirm-upper|confirm-posterior|confirm-finalists)
        case $dataset in mnist|fashionmnist|cifar10|moons) ;; *) echo "Supply comparison dataset" >&2; exit 2;; esac
        stage="$root/comparison/$dataset"
        mkdir -p "$stage"
        limit=00:12:00
        if [[ $dataset == cifar10 ]]; then limit=00:30:00; fi
        if [[ $phase == *upper && $dataset != cifar10 ]]; then
            echo "Upper-rate refinement requires CIFAR-10" >&2; exit 2
        fi
        if [[ $phase == *balanced && $dataset != cifar10 && $dataset != moons ]]; then
            echo "Balanced refinement supports CIFAR-10 and two moons" >&2; exit 2
        fi
        if [[ ( $phase == *posterior || $phase == *finalists ) && $dataset != fashionmnist ]]; then
            echo "Posterior refinement requires Fashion-MNIST" >&2; exit 2
        fi
        if [[ $phase == screen || $phase == refine || $phase == balanced || $phase == upper || $phase == posterior || $phase == finalists ]]; then
            if [[ $phase == screen ]]; then
                options=()
                if [[ $dataset == moons && -n ${DIAGONAL_MOONS_TRAIN_SIZE:-} ]]; then
                    options=(--moons-train-size "$DIAGONAL_MOONS_TRAIN_SIZE")
                fi
                "$py" -m "$compare" screen --datasets "$dataset" --m "${DIAGONAL_M:-2000}" \
                    "${options[@]}" --output "$stage/screen.jsonl"
                next=confirm
            elif [[ $phase == balanced || $phase == upper ]]; then
                options=()
                if [[ $phase == upper ]]; then options=(--upper); fi
                "$py" -m "$compare" balanced-screen "${options[@]}" --screen-manifest "$stage/screen.jsonl" \
                    --screen-root "$stage/screen" --output "$stage/$phase.jsonl"
                next=confirm-$phase
            elif [[ $phase == posterior ]]; then
                "$py" -m "$compare" posterior-screen --screen-manifest "$stage/screen.jsonl" --screen-root "$stage/screen" \
                    --refinement-manifest "$stage/refine.jsonl" --refinement-root "$stage/refine" --output "$stage/posterior.jsonl"
                next=confirm-posterior
            elif [[ $phase == finalists ]]; then
                "$py" -m "$compare" finalists-screen --screen-manifest "$stage/screen.jsonl" --screen-root "$stage/screen" \
                    --refinement-manifest "$stage/refine.jsonl" --refinement-root "$stage/refine" \
                    --posterior-manifest "$stage/posterior.jsonl" --posterior-root "$stage/posterior" --output "$stage/finalists.jsonl"
                next=confirm-finalists
            else
                "$py" -m "$compare" refine-screen --screen-manifest "$stage/screen.jsonl" --output "$stage/refine.jsonl"
                next=confirm-refine
            fi
            job=$(array "$stage" "$phase" "$compare" "$stage/$phase" "$limit")
            depends="$job"
            prerequisites=()
            case $phase in
                refine) prerequisites=(screen) ;;
                balanced) prerequisites=(refine) ;;
                upper) prerequisites=(refine balanced) ;;
            esac
            for prerequisite in "${prerequisites[@]}"; do
                # Slurm can forget a finished array by the time a new grid is submitted.
                # The confirmation controller validates each expected report before selection.
                reports="$stage/$prerequisite/reports"
                done_count=0
                if [[ -d $reports ]]; then done_count=$(find "$reports" -name '*.json' | wc -l); fi
                if (( done_count < $(wc -l < "$stage/$prerequisite.jsonl") )); then
                    depends+=":$(cat "$stage/$prerequisite.job")"
                fi
            done
            submit "$stage/$phase-select.job" "select_$dataset" "${cpu[@]}" --time=00:05:00 --dependency="afterok:$depends" \
                "$worker" bash "$launcher" "$next" "$dataset"
        else
            extra=()
            if [[ $phase == confirm-refine || $phase == confirm-balanced || $phase == confirm-upper || $phase == confirm-posterior || $phase == confirm-finalists ]]; then
                extra=(--refinement-manifest "$stage/refine.jsonl" --refinement-root "$stage/refine")
            fi
            if [[ $phase == confirm-upper ]]; then
                extra+=(--upper-manifest "$stage/upper.jsonl" --upper-root "$stage/upper")
            fi
            if [[ $phase == confirm-balanced || $phase == confirm-upper ]]; then
                extra+=(--balanced-manifest "$stage/balanced.jsonl" --balanced-root "$stage/balanced")
                if [[ $dataset == moons ]]; then extra+=(--eval-draws 100); fi
            fi
            if [[ $phase == confirm-posterior || $phase == confirm-finalists ]]; then
                extra+=(--posterior-manifest "$stage/posterior.jsonl" --posterior-root "$stage/posterior")
            fi
            if [[ $phase == confirm-finalists ]]; then
                extra+=(--finalists-manifest "$stage/finalists.jsonl" --finalists-root "$stage/finalists")
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
