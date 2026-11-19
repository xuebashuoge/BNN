# Deployment accuracy at m = 2000

Run `python results/final_chosen_results/rebuild_performance_table.py` from the repository root with the torch28 environment. It reads the archived test evaluations and writes `performance_table.tex` and `data/performance/deployment_by_seed.csv`. Pass `--check` to verify both outputs. A clean checkout has all required inputs.

## Trials

The table includes the four matched methods and the proposed perfect-training baseline. Each percentage averages test accuracy across paired training seeds, with the sample standard deviation across those seeds. Within a seed, reported accuracy averages the Gibbs deployment replicates. Fading uses diagonal gains with spatial correlation 0.35, marginal mean 0.5, unit marginal variance and independent unit-variance Gaussian noise. Erasure uses independent feature-erasure probability 0.50.

| Dataset | Fading confirmation | Paired seeds | Erasure confirmation | Paired seeds |
|---|---|---:|---|---:|
| Two moons | confirm | 3 | transfer | 5 |
| MNIST | confirm | 3 | prior-width reference | 5 |
| Fashion-MNIST | confirm-refine | 3 | common retuned | 9 |
| CIFAR-10 | confirm-balanced | 3 | scale grid | 5 |

The 40 displayed cells use 180 trained predictors. Bold marks the highest unrounded row mean. For two-moons fading, T2 averages 74.260500% and T1 averages 74.260333%.

## Records and weights

`deployment_by_seed.csv` maps each training seed and method to its task ID, training commit, evaluation, and checkpoint SHA-256. The `archive_key` column identifies the same directory in each trial archive.

- `selected_trials.tar.gz` contains the original test reports (including individual accuracy replicates), evaluation settings, checkpoint metadata, and training configs and manifest entries for the 180 selected fits. The eight cohort directories contain the original confirmation manifest, selection/protocol when available, summary and seed scores. Each original cohort records all methods considered, including auxiliary perfect-training controls.
- `training_source.tar.gz` contains the historical trainer and evaluator and their project dependencies by training commit, including the Spartan worker and floating-point bootstrap source. The file list for each version and the SHA-256 digest of each archive are in `provenance.json`.
- On Spartan, `results/final_chosen_results/models/performance/weights.tar` holds the original 180 PyTorch checkpoints. Run `python results/final_chosen_results/rebuild_performance_table.py --check --verify-models` to verify every weight file against its independent deployment report.
- The run records preserve the original directory strings for provenance. The table builder resolves all inputs within this checkout.

For inspection or replay, choose a row in `deployment_by_seed.csv`. Its original weight is at `<archive_key>/weights.pt` in the model archive. Its training configuration, metadata and evaluation overrides are at `<archive_key>/task_config.json`, `<archive_key>/metadata.json` and `<archive_key>/test_deployment/overrides.json` in the trial archive. Extract the weight and metadata into the same checkpoint directory. The historical entry point is `python -m journal.run_experiment` under the row's training revision in `training_source.tar.gz`. The original run used `--mode train` with the archived training configuration, followed by `--mode evaluate --split test --risk-scope deployment` with the saved weight, metadata and overrides. Set `DATASET` to the prepared dataset root when running the trainer or evaluator.
