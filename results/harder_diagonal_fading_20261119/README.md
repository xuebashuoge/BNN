# More severe correlated diagonal fading

The mild channel uses adjacent-gain correlation 0.35 and noise standard deviation 1. The combined hard channel uses correlation 0.85 and noise standard deviation 2. In every condition the gain matrix is diagonal, each gain is Gaussian with mean 0.5 and variance 1, and additive white Gaussian noise is independent of the gains. At a fixed transmit power the noise-power increase is 6.02 dB.

## Frozen models and matched retraining

A cell gives the accuracy drop when both changes are applied to the same mild-trained checkpoints, followed by the accuracy recovered by retraining on the hard law. Both values are percentage points. Four matched-training labels (ERM, L2, T1 and T2) receive equal weight within each paired seed. All figures in this comparison use the same 24 posterior and standardized channel streams across conditions.

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 100 | — | — | -11.31 / +5.99 | — |
| 200 | -8.21 / +3.31 | -11.79 / +4.35 | -11.83 / +5.57 | -7.90 / -0.81 |
| 500 | -8.79 / +3.77 | -11.02 / +4.83 | -8.05 / +4.82 | -8.32 / -0.35 |
| 1000 | -9.99 / +4.59 | -11.00 / +4.68 | -6.40 / +1.93 | -8.27 / -0.19 |
| 2000 | -10.10 / +4.73 | -10.37 / +4.69 | -5.36 / -0.81 | -8.29 / -0.26 |

[Four-law comparison and recovery figure](channel_severity.pdf) · [All channel means and paired changes](channel_effects.csv) · [Seed-level changes](channel_effects_by_seed.csv)

Correlation-only and noise-only evaluations isolate each change on the original classifiers. The trainable transmitter can change its feature power during retraining. The measured mean squared output per coordinate is recorded in the [transmit-power table](matched_transmit_power.csv). The effects CSV also gives the difference between the joint change and the sum of the two separate changes as interaction_pp. The retrained_hard_pct and recovery_pp columns compare freshly trained models with the original models under the same hard deployment channel.

## Deployment accuracy by training objective

The primary comparison transfers each mild-law validation-selected configuration and paired training seed. Matched methods retrain with a hard-channel bank. Perfect-trained methods train with the identity channel and use hard-channel validation to select the checkpoint. All seven are evaluated on the hard deployment law. Each cell below gives the more accurate matched T1/T2 algorithm and its mean gap in percentage points over the strongest of the five competing labels. Positive gaps indicate a proposed-method lead.

### Transfer

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 100 | — | — | T2 46.878% (+1.328) | — |
| 200 | T2 73.776% (+0.132) | T2 61.539% (+0.243) | T1 43.828% (-2.830) | T2 64.934% (-0.196) |
| 500 | T1 76.479% (-0.203) | T1 66.229% (+0.090) | T2 52.358% (+1.346) | T2 65.789% (-0.063) |
| 1000 | T1 77.820% (+0.061) | T1 67.797% (+0.170) | T2 53.101% (+0.940) | T1 65.837% (+0.017) |
| 2000 | T2 78.333% (-0.218) | T2 68.575% (+0.090) | T1 50.546% (+0.441) | T1 65.671% (-0.311) |

### Expansion/retuned

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 200 | T2 73.828% (+0.164) | — | — | — |
| 500 | T1 76.296% (+0.209) | — | — | — |
| 1000 | T1 77.556% (-0.075) | — | — | — |

### Retuned

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 200 | T1 73.379% (+0.038) | — | T1 44.535% (+2.640) | T1 65.461% (+0.169) |
| 500 | T1 76.129% (-0.028) | — | T2 52.625% (+0.157) | T2 65.571% (+0.019) |
| 1000 | T2 77.755% (+0.264) | — | T2 53.789% (+0.463) | T2 65.515% (-0.024) |
| 2000 | T1 78.392% (-0.039) | — | — | T1 65.610% (-0.095) |

The primary retraining table uses the native Gibbs-draw budget of its original cohort. The fixed-model and recovery tables use 24 fresh paired draws and can differ slightly from the native values. Validation-selected follow-ups, where present, use separate confirmation seeds. All completed cohorts remain in the numerical tables.

[Rankings, gaps and seed counts](method_comparisons.csv) · [All seven method means](method_means.csv) · [Per-seed accuracies](accuracy_by_seed.csv) · [Mild versus hard retraining, paired by seed](paired_retraining_vs_original.csv)

The complete seven-method transferred-configuration figures are available for [MNIST](mnist_transferred_methods.pdf), [Fashion-MNIST](fashionmnist_transferred_methods.pdf), [CIFAR-10](cifar10_transferred_methods.pdf), [Two moons](moons_transferred_methods.pdf).

The principal bank sizes are 200, 500 and 1000. The larger 2000 reference includes all four datasets. The earlier 100 reference is available for CIFAR-10. The sources file identifies each original confirmation cohort. Comparisons between datasets and between the core and reference studies can use different hyperparameters, seed cohorts and native Gibbs-draw budgets.
