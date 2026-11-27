# Selected journal results

The three figures and the two deployment accuracy tables below are the manuscript selection. Their plotting and table scripts resolve inputs relative to this directory. A fresh checkout contains everything needed to regenerate the publication outputs.

| Output | Inputs and experiment |
|---|---|
| [MNIST finite-channel generalization](mnist_analog_finite_channel_generalization.pdf) | [Observations](fading_unit_power/data/mnist_bank/observations.csv) from 54 T1 fits, three paired seeds at each of 18 channel-set sizes from 2 to 25,600, with unit transmit power ([details](fading_unit_power/README.md)). Measured feature power per coordinate is 1 to within 5e-8. T1 uses the fixed prior Q_W = N(0, I) in both complexity terms, so Q_{W\|S} = Q_W. Training and deployment use diagonal fading with correlation 0.35. The mean Δ(U,S) falls from 0.401 at m = 2 to 0.012 at m = 25,600, while deployment accuracy rises from 36.8% to 73.9%. The earlier figure, which predates unit transmit power, is kept as [`data/mnist_bank/mnist_analog_finite_channel_generalization_archived.pdf`](data/mnist_bank/mnist_analog_finite_channel_generalization_archived.pdf) and is redrawn by `draw_mnist_analog_finite_channel_generalization.py --archived`. |
| [CIFAR-10 erasure mismatch](cifar10_channel_mismatch_all_methods.pdf) | [Observations](split_u/cifar10_mismatch/observations.csv) from 156 fits, four methods and three paired seeds at each of 13 training erasure probabilities. T1 uses the split-U prior Q_{W\|S,U_1}, trained on the first part of the channel bank ([details](split_u/README.md)). ERM, L2 and T2 are the archived fits. Deployment erases features with probability 0.50 and the bank size is 2000. T1 is highest at 11 of 13 training probabilities; L2 leads at 0.05 and T2 at 0.10. Earlier versions are kept in `data/cifar10_mismatch/`: T1 with the fresh prior (`cifar10_channel_mismatch_fresh.pdf`, redrawn by `draw_cifar10_channel_mismatch.py --fresh`) and with the fixed prior Q_W (`cifar10_channel_mismatch_fixed_prior.pdf`, `--fixed-prior`). |
| [Two-moons bound decomposition](bound_decomposition_d8_t1_t2_both_channels.pdf) | [Ten certified case reports](tighter_bound/bounds/tightest/) for width 8 and bank size 2000, from the tightest selection rule ([details](tighter_bound/README.md)); T1 uses the fresh Q_{W\|S}. Panel (a) uses fading with deployment correlation 0.35 and mismatch training at 0.15. Panel (b) uses erasure with deployment probability 0.50 and mismatch training at 0.45. |
| [Deployment accuracy table](performance_table.tex) | [Per-seed accuracy](split_u/performance_m2000/deployment_by_seed.csv) from 180 fits at bank size 2000, with T1 using the split-U prior Q_{W\|S,U_1} ([details](split_u/README.md)). Deployment uses correlated fading with correlation 0.35 and unit noise variance, or independent feature erasure with probability 0.50. |
| [Bank-size table](bank_size_table.tex) | Best baseline max(ERM, L2) against best proposed max(T1, T2) at m = 20, 200, 1000 and 2000, with T1 using the split-U prior tuned at each m ([`split_u/performance_m*`](split_u/README.md)). |

## Regenerating figures and table

From the repository root, run the following command. It uses the saved results and takes no training time.

```sh
conda run -n torch28 bash results/final_chosen_results/rebuild.sh
```

Each figure script also runs independently and writes its PDF and PNG beside the script. The selected PDF and PNG files are included as originally supplied. The width-8 bar labels sum the five displayed components. Each full certificate in `tighter_bound/bounds/tightest/<channel>/<case>/bound.json` includes an additional Monte Carlo correction of 0.0012927 (2,000,000 draws). The diamonds show the held-out Gibbs loss. Both perfect predictors use Corollary 1(i); for fading, the expected distance is bounded with Jensen's inequality. Each C1, T1 and T2 bar uses the posterior with the smallest certified bound among those trained with the corresponding objective, so these posteriors are close to their priors and near chance accuracy. The earlier figure, with posteriors fitted for accuracy and a Monte Carlo correction of 0.0129273, is kept as [`data/bounds/bound_decomposition_d8_t1_t2_both_channels_archived.pdf`](data/bounds/bound_decomposition_d8_t1_t2_both_channels_archived.pdf) and is redrawn by `plot_bound_decomposition_d8.py --archived`.

Both tables come from `rebuild_final_tables.py`, which reads the published per-seed CSVs. The earlier tables with T1 using the fresh prior are kept as [`data/performance/performance_table_fresh.tex`](data/performance/performance_table_fresh.tex) and [`data/performance/bank_size_table_fresh.tex`](data/performance/bank_size_table_fresh.tex) and are rebuilt by `rebuild_final_tables.py --fresh`. The m = 2000 table with the fixed prior Q_W is kept as [`data/performance/performance_table_fixed_prior.tex`](data/performance/performance_table_fixed_prior.tex) and is rebuilt from its archived evaluations by `rebuild_performance_table.py`.

The m = 2000 table uses the sample standard deviation across paired training seeds. Each fading row uses three seeds. The erasure rows use five for Two moons, five for MNIST, nine for Fashion-MNIST, and five for CIFAR-10. Bold marks the largest mean before rounding. A proposed method leads every row at m = 2000: T1 in seven rows and T2 in Two moons erasure. Leads over max(ERM, L2) range from 0.03 points (MNIST erasure) to 1.21 points (CIFAR-10 erasure), with a mean of 0.44. T1's prior is trained on the first part of the channel bank U and its channel term uses the rest, so T1 sees no channels beyond the bank that ERM, L2 and T2 train on.

In the bank-size table, the proposed method wins 8 of 8 rows at m = 2000, 7 at m = 1000, 6 at m = 200 and 4 at m = 20, with mean leads of +0.44, +0.46, +0.18 and −0.61 points. The rows it loses are fading rows at small m (MNIST and Fashion-MNIST, and CIFAR-10 at m = 20) and Two moons erasure at m = 20 by 0.04 points. There the first part of the bank holds only a few channel draws, so the learned prior adds little. The CIFAR-10 mismatch figure and the CIFAR-10 erasure table row come from separate experiments with their own configurations and seeds.

## Checkpoints and experimental records

The `data/` subdirectories contain the run records, configurations and indices for the publication outputs. Each bound case includes its original configuration, empirical observations or summaries, test measurements and checkpoint fingerprint. The bound routines live in [the training and evaluation module](../../journal/correlated_bound_study.py). The recorded Git revisions in the case metadata identify the original training implementations.

The 400 selected fit records have checkpoint copies under `models/` in the Spartan and Mac working directories. Git ignores this directory. The archives preserve the original weights. The associated run indices identify their members. To verify the archives on Spartan, run `sha256sum -c models.sha256` from this directory. On macOS, use `shasum -a 256 -c models.sha256`.

The canonical Spartan checkout is `~/Git/bnn` on the `correlated_diagonal_fading` branch. On a new machine, Git provides the figures, table, scripts and inputs. To obtain the saved models as well, run this from the root of the new checkout after pulling the branch:

```sh
rsync -a unimelb:Git/bnn/results/final_chosen_results/models/ results/final_chosen_results/models/
```
