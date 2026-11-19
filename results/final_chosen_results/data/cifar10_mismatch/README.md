# CIFAR-10 erasure mismatch

The chosen figure uses all 156 rows in `observations.csv`. Each of four learners trains at thirteen erasure probabilities on seeds 75101–75103 with 2000 channel draws. Every learner deploys at erasure probability 0.50. The plotted accuracy and bounded test risk were verified against each task's held-out evaluation.

`manifests/0.jsonl` through `manifests/12.jsonl` and `protocol.json` preserve the original configurations and shard checksums. `run_index.csv` connects each task to its training revision, checkpoint fingerprint, and evaluation record. `run_records.tar.gz` contains the original per-task evaluation, metadata, training history, and config. Paths embedded in the original reports record the historical experiment location. The archive member paths in the index locate those files here.

On Spartan the weights are in `../../models/cifar10_mismatch/checkpoints.tar.gz` relative to this directory. Run `sha256sum -c archives.sha256` in this directory to verify both archives. Extracting them into the same directory recreates each task under `runs/<task_id>/`.

## Repeat the training on Spartan

The following commands run the original simulation at revision `f602987`. Run from the repository root with `DATASET` pointing to the prepared dataset directory. Each invocation creates a new result directory under the Git checkout.

```bash
selected="$PWD/results/final_chosen_results"
run_parent="$selected/models/cifar10_mismatch"
mkdir -p "$run_parent"
run=$(mktemp -d "$run_parent/rerun.XXXXXXXX")
git worktree add --detach "$run/source" f602987cceff4a078d12fd33cb6eec8582a1cd3c
(
    cd "$run/source"
    sbatch --output="$run/slurm-%A_%a.log" "$selected/data/cifar10_mismatch/run_spartan.slurm" \
        "$selected/data/cifar10_mismatch" "$run"
)
```

The thirteen Slurm tasks reproduce the original ten-draw full-risk evaluation protocol. The historical run was Slurm array 35329292.
