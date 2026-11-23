# Selected journal results

The three figures and the deployment accuracy table below are the manuscript selection. Their plotting and table scripts resolve inputs relative to this directory. A fresh checkout contains everything needed to regenerate the publication outputs.

| Output | Inputs and experiment |
|---|---|
| [MNIST finite-channel generalization](mnist_analog_finite_channel_generalization.pdf) | [Observations](data/mnist_bank/observations.csv) from 54 T1 fits, three paired seeds at each of 18 channel-set sizes from 2 to 25,600. Training and deployment use diagonal fading with correlation 0.35. |
| [CIFAR-10 erasure mismatch](cifar10_channel_mismatch_all_methods.pdf) | [Observations](data/cifar10_mismatch/observations.csv) from 156 fits, four methods and three paired seeds at each of 13 training erasure probabilities. Deployment erases features with probability 0.50 and the bank size is 2000. |
| [Two-moons bound decomposition](bound_decomposition_d8_t1_t2_both_channels.pdf) | [Ten fitted case reports](data/bounds/) for width 8 and bank size 2000. Panel (a) uses fading with deployment correlation 0.35 and mismatch training at 0.15. Panel (b) uses erasure with deployment probability 0.50 and mismatch training at 0.45. |
| [Deployment accuracy table](performance_table.tex) | [Per-seed accuracy](data/performance/deployment_by_seed.csv) and [evaluation records](data/performance/selected_trials.tar.gz) from 180 fits at bank size 2000. Deployment uses correlated fading with correlation 0.35 and unit noise variance, or independent feature erasure with probability 0.50. |

## Regenerating figures and table

From the repository root, run the following command. It uses the saved results and takes no training time.

```sh
conda run -n torch28 bash results/final_chosen_results/rebuild.sh
```

Each figure script also runs independently and writes its PDF and PNG beside the script. The selected PDF and PNG files are included as originally supplied. The width-8 bar labels sum the five displayed components. Each full certificate in `data/bounds/<channel>/<case>/bound.json` includes an additional Monte Carlo correction of 0.0129273. The diamonds show the held-out Gibbs loss. The fading perfect predictor uses Corollary 1(ii), and the erasure perfect predictor uses Corollary 1(i). Each T1 and T2 bar uses a predictor fitted with the corresponding objective.

The table uses the sample standard deviation across paired training seeds. Each fading row uses three seeds. The erasure rows use five for Two moons, five for MNIST, nine for Fashion-MNIST, and five for CIFAR-10. Bold marks the largest mean before rounding. For Two moons fading, T2 averages 74.260500% and T1 averages 74.260333%. The CIFAR-10 mismatch figure and the CIFAR-10 erasure table row come from separate experiments with their own configurations and seeds.

## Checkpoints and experimental records

The `data/` subdirectories contain the run records, configurations and indices for the publication outputs. Each bound case includes its original configuration, empirical observations or summaries, test measurements and checkpoint fingerprint. The bound routines live in [the training and evaluation module](../../journal/correlated_bound_study.py). The recorded Git revisions in the case metadata identify the original training implementations.

The 400 selected fit records have checkpoint copies under `models/` in the Spartan and Mac working directories. Git ignores this directory. The archives preserve the original weights. The associated run indices identify their members. To verify the archives on Spartan, run `sha256sum -c models.sha256` from this directory. On macOS, use `shasum -a 256 -c models.sha256`.

The canonical Spartan checkout is `~/Git/bnn` on the `correlated_diagonal_fading` branch. On a new machine, Git provides the figures, table, scripts and inputs. To obtain the saved models as well, run this from the root of the new checkout after pulling the branch:

```sh
rsync -a unimelb:Git/bnn/results/final_chosen_results/models/ results/final_chosen_results/models/
```
