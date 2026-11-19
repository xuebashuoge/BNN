# MNIST channel-bank size

The chosen figure uses the 54 T1 observations in `observations.csv`, with three seeds (74101–74103) at each of 18 bank sizes from 2 to 25,600. The original observations file also contains T2 results from the same sweep. Matched diagonal fading uses correlation 0.35 at training and deployment. Deployment accuracy and the signed bounded-loss gap were verified against each selected task's test report.

`manifest.jsonl` contains the exact T1 task entries, in original order, from the base and extended manifests. `run_index.csv` connects each plotted task to the source-code revision, weight fingerprint, and evaluation record. `run_records.tar.gz` stores the original per-task training report, evaluation, metadata, history, and config. Historical absolute paths in the original reports are provenance. Use the archive member paths from the index to find a file in this checkout.

On Spartan the 54 checkpoints are in `../../models/mnist_bank/checkpoints.tar.gz` relative to this directory. Run `sha256sum -c archives.sha256` here to verify the archives and original CPU library. Extracting the records and weights archives into the same directory recreates the task directories under `runs/<task_id>/`.

## Repeat the training on Spartan

Run the following from the repository root with `DATASET` set. It uses the training revision and floating-point settings recorded by Spartan for both original arrays, 35355614 and 35355935. The commands compile the CPU library from the matching source in Git and use the original checksum. The worker activates `torch28`.

```bash
selected="$PWD/results/final_chosen_results"
run_parent="$selected/models/mnist_bank"
mkdir -p "$run_parent"
run=$(mktemp -d "$run_parent/rerun.XXXXXXXX")
export JOURNAL_GIT_SHA=9616de066250ccec3c0b4115808885b64ebd54cc
git worktree add --detach "$run/source" "$JOURNAL_GIT_SHA"
manifest="$selected/data/mnist_bank/manifest.jsonl"
export JOURNAL_MANIFEST_SHA256=$(sha256sum "$manifest" | cut -d' ' -f1)
export FULL_FADING_FTZ_LIBRARY="$run/libbnn_ftz.so"
export FULL_FADING_FTZ_SHA256=5c2a8753f9eb46cdf1d039266ee3463c4bcff4534ba1a4d652c3a61cbb79cc3f
(
    cd "$run/source"
    cc -O2 -shared -fPIC journal/cpu_flush_bootstrap.c -o "$FULL_FADING_FTZ_LIBRARY"
    sbatch --array=0-53 --partition=sapphire --time=00:08:00 --cpus-per-task=2 --mem=4G \
        --output="$run/slurm-%A_%a.log" journal/full_fading_spartan.slurm \
        python -u -m journal.diagonal_fading_mnist_sweep task --manifest "$manifest" \
        --output-root "$run" --threads 2 --device cpu
)
```
