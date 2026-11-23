# Deployment accuracy on ten fresh paired seeds

This experiment measures how the accuracy gaps between T1, T2, ERM and L2 vary across fresh training runs. It repeats the 40 configurations from the [selected table](../performance_table.tex) on four datasets under independent 50% feature erasure and spatially correlated diagonal fading. The fading channel has adjacent-gain correlation 0.35 and unit noise variance. The training bank size is 2000.

## Design

The ten training seeds are 761001 through 761010, inclusive. Every dataset, channel and method uses every seed. We fixed the list before launching the simulations. Each comparison includes matched ERM, L2, T1, T2 and the shared T1/T2 perfect-training control. Each run retains its archived hyperparameters, training implementation, data split and evaluation budget. The training initialization, finite channel bank and evaluation streams change with the seed. The image experiments reuse the official test sets.

The unit of comparison is a training seed. For each dataset and channel, we subtract each baseline's accuracy from T1's or T2's accuracy at the same seed, then report the mean difference in percentage points. The range and the number of wins describe how much the result depends on the draw. The paired-contrast CSV compares each fixed method pair across the two cohorts.

## Results

T1 or T2 leads the stronger matched baseline in five of eight settings. The table compares the higher mean accuracy of T1 and T2 with the higher mean accuracy of ERM and L2 within each cohort. All gaps are percentage points. The method names identify the pair in the ten-seed replication.

| Dataset | Channel | Proposed | Baseline | Earlier gap | Ten-seed gap |
|---|---|---|---|---:|---:|
| Two moons | Fading | T2 | ERM | +0.006 | +0.021 |
| Two moons | Erasure | T2 | ERM | +0.283 | −0.101 |
| MNIST | Fading | T1 | ERM | +0.120 | +0.325 |
| MNIST | Erasure | T1 | L2 | +0.051 | +0.016 |
| Fashion-MNIST | Fading | T1 | L2 | +0.225 | +0.106 |
| Fashion-MNIST | Erasure | T2 | L2 | +0.022 | +0.152 |
| CIFAR-10 | Fading | T2 | L2 | +1.523 | −0.473 |
| CIFAR-10 | Erasure | T1 | L2 | +0.109 | −0.689 |

On MNIST fading, T1 exceeds ERM by 0.325 points and L2 by 0.659 points on average. It exceeds both on seven of ten seeds. Under Fashion-MNIST erasure, T2 exceeds ERM by 0.153 points and L2 by 0.152 points, beating both on eight seeds. On CIFAR-10 fading, T2 leads ERM by 3.055 points and trails L2 by 0.473 points. Its paired differences from L2 range from −4.2420 to +2.7785 points.

The two paper tables contain all five method means and all four fixed T1/T2 comparisons with the matched baselines. [Deployment accuracy](report/performance_table.tex) · [Paired differences](report/paired_differences_table.tex)

The seed-level release identifies every checkpoint. For each proposed method the summaries give its number of strict wins against both baselines at the same seed. Each fixed-pair contrast includes the mean, sample SD, extrema, wins, ties and losses across the ten seeds, together with the earlier cohort's mean gap for that same pair. [400 seed-level scores](report/deployment_by_seed.csv) · [40 method summaries](report/method_summary.csv) · [32 paired comparisons](report/paired_differences.csv)

## Reproduction on Spartan

From the checkout containing the submitted code, set `DATASET` to the prepared dataset directory and launch the CPU arrays with the following command.

```sh
export DATASET=/data/gpfs/projects/punim2792/Datasets
export FULL_FADING_FTZ_LIBRARY=/data/gpfs/projects/punim2792/yangshuo/matrix_cpu_runtime_20261118/libbnn_ftz.so
export FULL_FADING_FTZ_SHA256=5c2a8753f9eb46cdf1d039266ee3463c4bcff4534ba1a4d652c3a61cbb79cc3f
bash journal/submit_paired_accuracy_replication.sh /data/gpfs/projects/punim2792/yangshuo/paired_accuracy_seeds_20261123
```

The Slurm jobs use the `torch28` environment and run with one CPU thread for erasure and two for fading, matching the archived trainers. The experiment directory on Spartan holds the individual checkpoints and evaluation records.

## Saved runs

The Slurm fading and erasure arrays were 35541625 and 35541639. The dependent report was 35541640. All 100 array elements and the report completed with exit code 0. The submission used Git revision `e8a8d9c`. Each manifest entry records the historical source revision used to fit that configuration. The trainer runs from the corresponding archived source snapshot.

The [run records](run_records.tar.gz) contain the frozen manifest, job IDs, training configurations and histories, checkpoint metadata, and per-draw deployment evaluations. The [release inventory](release.json) gives SHA-256 hashes for both archives, the manifest and the five report files. The weights are stored in `models/fresh_paired_seeds/weights.tar.gz` under the selected-results directory locally and in `archives/weights.tar.gz` under the experiment directory on Spartan.

To regenerate the report from the downloaded files, run the following commands from the repository root.

```sh
bundle=results/final_chosen_results/fresh_paired_seeds
replay=$(mktemp -d)
tar -xzf "$bundle/run_records.tar.gz" -C "$replay"
tar -xzf "$bundle/../models/fresh_paired_seeds/weights.tar.gz" -C "$replay"
conda run -n torch28 python -m journal.report_paired_accuracy_replication "$replay"
diff -r "$bundle/report" "$replay/report"
```
