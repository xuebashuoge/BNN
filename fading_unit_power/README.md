# Fading experiments with unit transmit power

This campaign reruns the fading results selected in [`results/final_chosen_results`](../results/final_chosen_results). Every transmitted feature vector has squared norm equal to its number of real channel symbols. The normalization happens before channel gain and receiver noise for each input and posterior draw. An all-zero encoder output transmits ones. The perfect-training comparator uses the same transmitter with a noiseless identity channel.

| Study | Protocol |
|---|---|
| MNIST generalization and deployment accuracy vs. $m$ | Replay the 54 archived T1 configurations at 18 bank sizes using the three archived paired seeds and the archived evaluation seeds. |
| Two-moons bound decomposition | Replay the five fading configurations at width 8 and $m=2000$. The C1(ii) shift uses the normalized radius $\sqrt{8}$ in the Lipschitz and Gaussian MGF calculation. The recorded empirical Monte Carlo seed is reused with separate streams. The held-out calculation uses the configured evaluation seed. |
| Deployment accuracy at $m=2000$ | Compare three configurations per method on two paired training seeds using deployment accuracy on validation. Confirm each method's selected configuration on three fresh paired seeds. Report the gap from the better of T1 and T2 to the strongest baseline, in percentage points. |

The trainer and channel/model dependencies are adapted from the separate source archive in `results/final_chosen_results/data/performance/training_source.tar.gz`, revision `20f5fc96642a1dc50bc234b03940d9cc95712e13`. The MNIST and bound archive records supply their training and evaluation configurations. The bound records do not specify the original Monte Carlo substream offsets or the held-out evaluation seed.

## Spartan

Use a dedicated checkout and create a new campaign directory on project storage. The `DATASET` environment variable can override Spartan's prepared dataset directory.

```bash
cd ~/Git/bnn-unit-power
bash -l fading_unit_power/submit_spartan.sh /data/gpfs/projects/punim2792/yangshuo/unit-power-fading-RUN
```

The launcher submits CPU arrays on `sapphire`, followed by dependent validation selection, held-out confirmation and reporting jobs. `jobs.tsv` records the Slurm job IDs and dependencies, and `logs/` contains the per-task logs. Use `fixed` or `performance` as a second argument to submit just that part in a new campaign directory.

The fixed-study figures and source records land in `fixed/fixed_results/`. The comparison writes `performance/performance_table.tex`, `performance/deployment_by_seed.csv` and `performance/performance_summary.json`. The comparison table and bound figure draw their erasure data from the original archive. The fitted weights, full evaluation records and frozen manifests are under the campaign directory.

A dataset can be refined after its validation selection finishes. The new search uses each method's validation winner and nearby learning rates, regularization strengths or posterior scales. The planner reads the search manifest and validation selection record. Use a checkout separate from the primary run while any primary job is active.

```bash
bash -l fading_unit_power/submit_spartan.sh "$refinement_run" refine "$primary_run/performance" moons
```

When both runs finish, `python -m fading_unit_power.performance summarize --output "$primary_run/performance" --source "moons=$refinement_run/performance"` uses the refined confirmation for Two moons in the combined table.
