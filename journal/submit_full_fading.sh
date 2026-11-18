#!/usr/bin/env bash
# Submit independently backfillable Spartan jobs from the pinned Git checkout.
set -euo pipefail

phase=${1:?Use pilot, sweep, sweep-extended, bounds, comparison, or all}
: "${FULL_FADING_OUTPUT:?Set an absolute results directory on project storage}"
: "${DATASET:?Set the prepared dataset root}"
: "${JOURNAL_GIT_SHA:?Export the committed Git revision}"
[[ "$FULL_FADING_OUTPUT" = /* && -d "$DATASET" ]]
cd "$(git rev-parse --show-toplevel)"
test "$(git rev-parse HEAD)" = "$JOURNAL_GIT_SHA"
test -z "$(git status --porcelain --untracked-files=no)"

root="$FULL_FADING_OUTPUT"
mkdir -p "$root/logs"
launcher=journal/full_fading_spartan.slurm
python_bin="${FULL_FADING_PYTHON:-python}"
image_device="${FULL_FADING_IMAGE_DEVICE:-cuda}"
comparison_datasets="${FULL_FADING_COMPARISON_DATASETS:-mnist,fashionmnist,cifar10}"
array_parallelism="${FULL_FADING_PARALLELISM:-16}"
threads=2
cpu=(--partition="${FULL_FADING_CPU_PARTITION:-sapphire}" --cpus-per-task="$threads" --mem=5G)
gpu=(--partition="${FULL_FADING_GPU_PARTITION:-feit-gpu-a100}" --qos="${FULL_FADING_GPU_QOS:-feit}"
     --gres=gpu:1 --cpus-per-task="$threads" --mem=8G)
case "$image_device" in
    cuda) image=("${gpu[@]}") ;;
    cpu) image=("${cpu[@]}") ;;
    *) echo "FULL_FADING_IMAGE_DEVICE must be cpu or cuda" >&2; exit 2 ;;
esac

submit() {
    local job_file=$1 job_name=$2
    shift 2
    if [[ -s "$job_file" ]]; then
        cat "$job_file"
        return
    fi
    local options=()
    while [[ "$1" != "--" ]]; do
        options+=("$1")
        shift
    done
    shift
    local job
    job=$(sbatch --parsable --chdir="$PWD" --job-name="$job_name" \
          --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err" \
          "${options[@]}" "$launcher" "$@")
    job=${job%%;*}
    printf '%s\n' "$job" | tee "$job_file"
}

manifest_submit() {
    local manifest=$1
    shift
    local checksum
    checksum=$(sha256sum "$manifest")
    JOURNAL_MANIFEST_SHA256="${checksum%% *}" submit "$@"
}

run_pilot() {
    local pilot="$root/pilot" manifest="$root/pilot/manifest.jsonl"
    mkdir -p "$pilot/cpu" "$pilot/$image_device"
    if [[ ! -f "$manifest" ]]; then
        "$python_bin" -m journal.full_fading_mnist_sweep build --output "$manifest" --pilot
    fi
    for device in cpu "$image_device"; do
        local resources=("${cpu[@]}")
        if [[ "$device" == cuda ]]; then
            resources=("${gpu[@]}")
        fi
        manifest_submit "$manifest" "$pilot/$device/job_id" "fading_pilot_$device" \
            "${resources[@]}" --time=00:06:00 -- \
            python -u -m journal.full_fading_mnist_sweep task --manifest "$manifest" \
            --output-root "$pilot/$device" --index 0 --device "$device" --threads "$threads"
    done
}

run_sweep() {
    local stage="$root/$1" manifest="$root/$1/manifest.jsonl"
    local extra=() report_inputs=() dependencies=()
    mkdir -p "$stage"
    if [[ "$1" == sweep-extended ]]; then
        extra=(--extended)
        [[ -s "$root/sweep/tasks.job" ]]
        report_inputs=(--manifest "$root/sweep/manifest.jsonl" --stage-root "$root/sweep")
        dependencies+=("$(cat "$root/sweep/tasks.job")")
    fi
    if [[ ! -f "$manifest" ]]; then
        "$python_bin" -m journal.full_fading_mnist_sweep build --output "$manifest" "${extra[@]}"
    fi
    local last job wait_for
    last=$(($(wc -l < "$manifest") - 1))
    job=$(manifest_submit "$manifest" "$stage/tasks.job" fading_mnist_m \
        "${image[@]}" --time=00:15:00 --array="0-$last%$array_parallelism" -- \
        python -u -m journal.full_fading_mnist_sweep task --manifest "$manifest" \
        --output-root "$stage" --device "$image_device" --threads "$threads")
    dependencies+=("$job")
    wait_for=$(IFS=:; printf '%s' "${dependencies[*]}")
    submit "$stage/report.job" fading_mnist_plot "${cpu[@]}" --time=00:05:00 \
        --dependency="afterok:$wait_for" -- python -u -m journal.full_fading_mnist_sweep report \
        "${report_inputs[@]}" --manifest "$manifest" --stage-root "$stage" --output "$stage/figures"
    printf 'MNIST training array %s\n' "$job"
}

run_bounds() {
    local stage="$root/bounds" moon_job image_job
    mkdir -p "$stage"
    moon_job=$(submit "$stage/moons.job" fading_bounds_moons "${cpu[@]}" --time=00:08:00 \
        --array=0-11%12 -- bash journal/submit_full_fading.sh bound-task moons)
    image_job=$(submit "$stage/images.job" fading_bounds_images "${image[@]}" --time=00:25:00 \
        --array="0-17%$array_parallelism" -- bash journal/submit_full_fading.sh bound-task images)
    submit "$stage/report.job" fading_bounds_plot "${cpu[@]}" --time=00:08:00 \
        --dependency="afterok:$moon_job:$image_job" -- \
        python -u -m journal.full_fading_bounds aggregate --output "$stage"
    printf 'Bound arrays %s %s\n' "$moon_job" "$image_job"
}

run_bound_task() {
    local regime=(perfect matched mismatch) family=(fading erasure)
    local datasets=(mnist fashionmnist cifar10) index="$SLURM_ARRAY_TASK_ID"
    local device="$image_device" dataset extra=()
    if [[ "$1" == moons ]]; then
        dataset=moons
        device=cpu
        if (( index >= 6 )); then
            extra=(--fallback)
        fi
    else
        dataset=${datasets[index / 6]}
    fi
    exec python -u -m journal.full_fading_bounds run --output "$root/bounds" \
        --dataset "$dataset" --family "${family[(index % 6) / 3]}" --regime "${regime[index % 3]}" \
        --device "$device" --threads "$threads" --m 2000 "${extra[@]}"
}

run_comparison() {
    local dataset=$1 stage="$root/comparison/$1" resources=("${image[@]}")
    local device="$image_device" time_limit=00:20:00
    if [[ "$dataset" == moons ]]; then
        resources=("${cpu[@]}")
        device=cpu
        time_limit=00:08:00
    fi
    if [[ "$dataset" == cifar10 ]]; then
        time_limit=00:35:00
    fi
    local manifest="$stage/screen.jsonl"
    mkdir -p "$stage"
    if [[ ! -f "$manifest" ]]; then
        "$python_bin" -m journal.full_fading_comparison screen --output "$manifest" \
            --datasets "$dataset" --m 2000
    fi
    local last screen_job
    last=$(($(wc -l < "$manifest") - 1))
    screen_job=$(manifest_submit "$manifest" "$stage/screen.job" "fading_screen_$dataset" \
        "${resources[@]}" --time="$time_limit" --array="0-$last%$array_parallelism" -- \
        python -u -m journal.full_fading_comparison task --manifest "$manifest" \
        --output-root "$stage/screen" --device "$device" --threads "$threads")
    submit "$stage/select.job" "fading_select_$dataset" "${cpu[@]}" --time=00:05:00 \
        --dependency="afterok:$screen_job" -- bash journal/submit_full_fading.sh confirm "$dataset"
    printf '%s comparison screen array %s\n' "$dataset" "$screen_job"
}

run_confirmation() {
    local dataset=$1 stage="$root/comparison/$1" resources=("${image[@]}")
    local device="$image_device" time_limit=00:25:00
    if [[ "$dataset" == moons ]]; then
        resources=("${cpu[@]}")
        device=cpu
        time_limit=00:08:00
    fi
    if [[ "$dataset" == cifar10 ]]; then
        time_limit=00:40:00
    fi
    local manifest="$stage/confirm.jsonl"
    if [[ ! -f "$manifest" ]]; then
        "$python_bin" -m journal.full_fading_comparison confirm --screen-manifest "$stage/screen.jsonl" \
            --screen-root "$stage/screen" --output "$manifest"
    fi
    local last confirm_job
    last=$(($(wc -l < "$manifest") - 1))
    confirm_job=$(manifest_submit "$manifest" "$stage/confirm.job" "fading_confirm_$dataset" \
        "${resources[@]}" --time="$time_limit" --array="0-$last%$array_parallelism" -- \
        python -u -m journal.full_fading_comparison task --manifest "$manifest" \
        --output-root "$stage/confirm" --device "$device" --threads "$threads")
    submit "$stage/report.job" "fading_accuracy_$dataset" "${cpu[@]}" --time=00:05:00 \
        --dependency="afterok:$confirm_job" -- python -u -m journal.full_fading_comparison report \
        --manifest "$manifest" --output-root "$stage/confirm"
    printf '%s comparison confirmation array %s\n' "$dataset" "$confirm_job"
}

case "$phase" in
    pilot) run_pilot ;;
    sweep|sweep-extended) run_sweep "$phase" ;;
    bounds) run_bounds ;;
    bound-task) run_bound_task "$2" ;;
    confirm) run_confirmation "$2" ;;
    comparison|all)
        if [[ "$phase" == all ]]; then
            run_sweep sweep
            run_bounds
        fi
        IFS=, read -ra datasets <<< "$comparison_datasets"
        for dataset in "${datasets[@]}"; do
            run_comparison "$dataset"
        done
        ;;
    *) echo "Unknown study phase $phase" >&2; exit 2 ;;
esac
