# Deployment accuracy on ten fresh paired seeds

This experiment measures how the accuracy gaps between T1, T2, ERM and L2 vary across fresh training runs. It repeats the 40 configurations from the [selected table](../performance_table.tex) on four datasets under independent 50% feature erasure and spatially correlated diagonal fading. The fading channel has adjacent-gain correlation 0.35 and unit noise variance. The training bank size is 2000.

## Design

The ten training seeds are 761001 through 761010, inclusive. Every dataset, channel and method uses every seed. We fixed the list before launching the simulations. Each comparison includes matched ERM, L2, T1, T2 and the shared T1/T2 perfect-training control. Each run retains its archived hyperparameters, training implementation, data split and evaluation budget. The training initialization, finite channel bank and evaluation streams change with the seed. The image experiments reuse the official test sets.

The unit of comparison is a training seed. For each dataset and channel, we subtract each baseline's accuracy from T1's or T2's accuracy at the same seed, then report the mean difference in percentage points. The range and the number of wins describe how much the result depends on the draw. The comparison with the earlier table uses a separate cohort and the same fixed pair of methods.

## Reproduction on Spartan

From the checkout containing the submitted code, set `DATASET` to the prepared dataset directory and launch the CPU arrays with the following command.

```sh
export DATASET=/data/gpfs/projects/punim2792/Datasets
export FULL_FADING_FTZ_LIBRARY=/data/gpfs/projects/punim2792/yangshuo/matrix_cpu_runtime_20261118/libbnn_ftz.so
export FULL_FADING_FTZ_SHA256=5c2a8753f9eb46cdf1d039266ee3463c4bcff4534ba1a4d652c3a61cbb79cc3f
bash journal/submit_paired_accuracy_replication.sh /data/gpfs/projects/punim2792/yangshuo/paired_accuracy_seeds_20261123
```

The Slurm jobs use the `torch28` environment and run with one CPU thread for erasure and two for fading, matching the archived trainers. The experiment directory on Spartan holds the individual checkpoints and evaluation records.
