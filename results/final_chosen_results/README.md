# Selected journal results

The three figures and the two deployment accuracy tables below are the manuscript selection. Their plotting and table scripts resolve inputs relative to this directory. A fresh checkout contains everything needed to regenerate the publication outputs.

| Output | Inputs and experiment |
|---|---|
| [MNIST finite-channel generalization](mnist_analog_finite_channel_generalization.pdf) | [Observations](data/mnist_bank/observations.csv) from 54 T1 fits, three paired seeds at each of 18 channel-set sizes from 2 to 25,600. Training and deployment use diagonal fading with correlation 0.35. |
| [CIFAR-10 erasure mismatch](cifar10_channel_mismatch_all_methods.pdf) | [Observations](data_dependent/only_fresh/cifar10_mismatch/observations.csv) from 156 fits, four methods and three paired seeds at each of 13 training erasure probabilities. T1 uses the fresh Q_{W\|S} for every training erasure probability above 0 ([details](data_dependent/README.md)). At 0 the bank has no channel complexity, so the archived T1 fits are reused. ERM, L2 and T2 are the archived fits. Deployment erases features with probability 0.50 and the bank size is 2000. The earlier figure with the fixed prior Q_W is kept as [`data/cifar10_mismatch/cifar10_channel_mismatch_fixed_prior.pdf`](data/cifar10_mismatch/cifar10_channel_mismatch_fixed_prior.pdf) and is redrawn by `draw_cifar10_channel_mismatch.py --fixed-prior`. |
| [Two-moons bound decomposition](bound_decomposition_d8_t1_t2_both_channels.pdf) | [Ten fitted case reports](data/bounds/) for width 8 and bank size 2000. Panel (a) uses fading with deployment correlation 0.35 and mismatch training at 0.15. Panel (b) uses erasure with deployment probability 0.50 and mismatch training at 0.45. |
| [Deployment accuracy table](performance_table.tex) | [Per-seed accuracy](data_dependent/only_fresh/performance/deployment_by_seed.csv) from 180 fits at bank size 2000, with T1 using the fresh data-dependent prior Q_{W\|S} ([details](data_dependent/README.md)). Deployment uses correlated fading with correlation 0.35 and unit noise variance, or independent feature erasure with probability 0.50. |
| [Bank-size table](bank_size_table.tex) | Best baseline max(ERM, L2) against best proposed max(T1, T2) at m = 20, 200, 1000 ([campaigns](higher_accuracy/README.md)) and 2000 (the table above). |

## Regenerating figures and table

From the repository root, run the following command. It uses the saved results and takes no training time.

```sh
conda run -n torch28 bash results/final_chosen_results/rebuild.sh
```

Each figure script also runs independently and writes its PDF and PNG beside the script. The selected PDF and PNG files are included as originally supplied. The width-8 bar labels sum the five displayed components. Each full certificate in `data/bounds/<channel>/<case>/bound.json` includes an additional Monte Carlo correction of 0.0129273. The diamonds show the held-out Gibbs loss. The fading perfect predictor uses Corollary 1(ii), and the erasure perfect predictor uses Corollary 1(i). Each T1 and T2 bar uses a predictor fitted with the corresponding objective.

Both tables come from `rebuild_final_tables.py`, which reads the published per-seed CSVs. The earlier m = 2000 table with the fixed prior Q_W is kept as [`data/performance/performance_table_fixed_prior.tex`](data/performance/performance_table_fixed_prior.tex) and is rebuilt from its archived evaluations by `rebuild_performance_table.py`.

The m = 2000 table uses the sample standard deviation across paired training seeds. Each fading row uses three seeds. The erasure rows use five for Two moons, five for MNIST, nine for Fashion-MNIST, and five for CIFAR-10. Bold marks the largest mean before rounding. T1 is best in seven rows and T2 in Two moons erasure, so a proposed method leads every row. Leads over max(ERM, L2) range from 0.13 points (MNIST erasure) to 2.62 points (CIFAR-10 fading), with a mean of 0.75. Stage 1 of T1 sees fresh simulated channel draws that ERM, L2 and T2 do not, and T1 trains twice; state this with the table.

In the bank-size table, the proposed method's lead grows as m shrinks: mean +0.75, +1.02, +2.46 and +8.04 points at m = 2000, 1000, 200 and 20. T1 stays near its m = 2000 accuracy at every size while ERM and L2 degrade. It wins 7 of 8 rows at m = 200 and m = 1000, losing one Two moons row by less than 0.1 point. The m = 20, 200 and 1000 columns were tuned at their own m with this branch's wider protocol, while the m = 2000 columns repeat the main table. The CIFAR-10 mismatch figure and the CIFAR-10 erasure table row come from separate experiments with their own configurations and seeds.

## Checkpoints and experimental records

The `data/` subdirectories contain the run records, configurations and indices for the publication outputs. Each bound case includes its original configuration, empirical observations or summaries, test measurements and checkpoint fingerprint. The bound routines live in [the training and evaluation module](../../journal/correlated_bound_study.py). The recorded Git revisions in the case metadata identify the original training implementations.

The 400 selected fit records have checkpoint copies under `models/` in the Spartan and Mac working directories. Git ignores this directory. The archives preserve the original weights. The associated run indices identify their members. To verify the archives on Spartan, run `sha256sum -c models.sha256` from this directory. On macOS, use `shasum -a 256 -c models.sha256`.

The canonical Spartan checkout is `~/Git/bnn` on the `correlated_diagonal_fading` branch. On a new machine, Git provides the figures, table, scripts and inputs. To obtain the saved models as well, run this from the root of the new checkout after pulling the branch:

```sh
rsync -a unimelb:Git/bnn/results/final_chosen_results/models/ results/final_chosen_results/models/
```
