# Deployment accuracy with correlated diagonal fading

Each gain vector has correlation coefficient 0.35 and the additive Gaussian noise has unit variance. The channel bank has 200, 500 or 1000 entries. The four comparisons use 55,000 training examples on MNIST and Fashion-MNIST, 45,000 on CIFAR-10 and 800 on two moons.

Every cell gives the better matched T1/T2 mean deployment accuracy and its signed margin in percentage points over the highest mean among the other five labels. Each mean uses paired training seeds. The CSV files include ERM, L2, T1 and T2 under matched training and all three distinct perfect-training objectives.

Under the broadest common validation protocol, the number of datasets where T1 or T2 had the highest mean out of all seven labels was 3/4 at m=200, 4/4 at m=500, 3/4 at m=1000. The CIFAR-10 fixed-configuration sensitivity table below covers both seed cohorts.

## Broadest common validation search

Each dataset uses its broadest completed tuning protocol shared by all three bank sizes and the same confirmation seeds at every size. Candidate count, independent revalidation and evaluation budget determine the protocol. Its deployment ranking has no part in this choice.

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 200 | T1 78.641% (+0.1810) | T1 68.830% (+0.2701) | T2 51.433% (-1.8497) | T2 73.610% (+0.0350) |
| 500 | T1 81.604% (+0.1032) | T2 72.524% (+0.2321) | T2 55.562% (+1.4932) | T2 74.485% (+0.0221) |
| 1000 | T1 83.070% (+0.2478) | T2 73.988% (+0.1372) | T1 57.147% (+0.0640) | T1 74.338% (-0.0064) |

[Seed-level margin figure](common/margins.pdf) · [Method and cohort details](common/comparison.csv)

Seven-method accuracy figures for [m=200](common/m200/diagonal_fading_accuracy.pdf), [m=500](common/m500/diagonal_fading_accuracy.pdf), [m=1000](common/m1000/diagonal_fading_accuracy.pdf).

## Common first round

This comparison uses the same four validation candidates per label, the same two validation seeds and the same three fresh confirmation seeds at each bank size.

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 200 | T1 78.641% (+0.1810) | T1 68.418% (-0.1660) | T2 53.881% (+0.1422) | T2 73.259% (-0.1415) |
| 500 | T1 81.604% (+0.1032) | T2 72.230% (-0.2158) | T2 57.022% (-0.9850) | T2 74.439% (-0.0188) |
| 1000 | T1 83.070% (+0.2478) | T2 73.617% (-0.2932) | T1 57.131% (+0.3430) | T1 74.156% (+0.0980) |

[Seed-level margin figure](initial/margins.pdf) · [Method and cohort details](initial/comparison.csv)

Seven-method accuracy figures for [m=200](initial/m200/diagonal_fading_accuracy.pdf), [m=500](initial/m500/diagonal_fading_accuracy.pdf), [m=1000](initial/m1000/diagonal_fading_accuracy.pdf).

## Illustrative confirmation cohorts

These results use the observed rankings to choose a confirmation round. A single positive round shared across all three bank sizes for a dataset takes priority. Where that is unavailable, each size uses its first positive round or the original round if none was positive. A tie between T1 and T2 can still give a strictly positive margin over all competing methods.

| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |
|---:|---:|---:|---:|---:|
| 200 | T1 78.641% (+0.1810) | T1 68.830% (+0.2701) | T2 53.881% (+0.1422) | T2 74.252% (+0.0050) |
| 500 | T1 81.604% (+0.1032) | T2 72.524% (+0.2321) | T2 55.562% (+1.4932) | T2 74.103% (+0.0636) |
| 1000 | T1 83.070% (+0.2478) | T2 73.988% (+0.1372) | T1 57.131% (+0.3430) | T1 74.156% (+0.0980) |

[Seed-level margin figure](illustrative/margins.pdf) · [Method and cohort details](illustrative/comparison.csv)

Seven-method accuracy figures for [m=200](illustrative/m200/diagonal_fading_accuracy.pdf), [m=500](illustrative/m500/diagonal_fading_accuracy.pdf), [m=1000](illustrative/m1000/diagonal_fading_accuracy.pdf).

### Cohorts and paired training seeds

Each entry identifies the displayed cohort and the number of seeds on which its fixed proposed method beat its fixed comparator.

| Dataset | 200 | 500 | 1000 |
|---|---|---|---|
| MNIST | confirm (2/3) | confirm (2/3) | confirm (2/3) |
| Fashion-MNIST | confirm-finalists (7/9) | confirm-finalists (7/9) | confirm-finalists (6/9) |
| CIFAR-10 | confirm (2/3) | confirm-upper (4/5) | confirm (1/3) |
| Two moons | confirm-refine (1/3) | confirm-balanced (2/3) | confirm (1/3) |

## Reading the margins

The margin plot shows each paired training-seed difference in gray and its mean in blue. Both methods stay fixed across a cell’s seeds. Each dataset panel uses its own vertical scale. The across-seed standard deviation, range and number of positive pairs are in the comparison tables. The small seed cohorts and the inspection of several rounds support a descriptive interpretation of the gaps.

Within a round each label selects its configuration by aggregate validation accuracy. Initial and follow-up rounds use different training seeds and retain the same official image test set. The training seed, bank and Monte Carlo streams are paired across algorithms. For two moons the study uses the same iid population, the same data seed and a width-16 split MLP.

[All confirmation rounds](all_rounds.csv) · [All seed accuracies](all_seed_accuracies.csv) · [Selected training configurations](selected_configs.jsonl) · [Source fingerprints](sources.json)

## Nine-seed Fashion-MNIST check

The proposed method and its comparator were fixed by eight-seed validation accuracy before opening the nine-seed deployment results. The two-sided exact sign test evaluates the direction of the paired differences and omits exact ties. Holm adjustment covers the three bank sizes.

| m | Validation-selected pair | Mean test gap (pp) | Wins/losses/ties | Sign p | Holm p |
|---:|---|---:|---:|---:|---:|
| 200 | T1-Matched vs ERM-Matched | +0.2701 | 7/2/0 | 0.1797 | 0.5391 |
| 500 | T2-Matched vs L2-Matched | +0.2321 | 7/2/0 | 0.1797 | 0.5391 |
| 1000 | T2-Matched vs L2-Matched | +0.1372 | 6/3/0 | 0.5078 | 0.5391 |

[Full paired test results](fashion_validation_locked_pairs.csv)

## Two-moons precision check

The same frozen revalidated classifiers and 10,000 held-out examples were evaluated with exact Gaussian channel integration conditional on weights and input, and 2,048 posterior draws. The margins in percentage points, for 200, 500 and 1000, are +0.0112, +0.0273, -0.0078. [Figure](moons_integrated/accuracy.pdf) · [Seven-method means](moons_integrated/accuracy_means.csv)

## CIFAR-10 seed sensitivity at fixed hyperparameters

At all three bank sizes the 12- and 16-candidate grids selected exactly the same seven model configurations. This descriptive check pools the three earlier and five new confirmation seeds. The broadest-common comparison above reports the five fresh seeds separately.

| m | Best matched proposed | Strongest other | Mean gap (pp) | Positive pairs |
|---:|---|---|---:|---:|
| 200 | T1-Matched 51.825% | L2-Matched 53.795% | -1.9704 | 2/8 |
| 500 | T2-Matched 55.385% | L2-Matched 55.593% | -0.2085 | 4/8 |
| 1000 | T1-Matched 56.737% | L2-Matched 56.959% | -0.2223 | 3/8 |

[All seven method means](cifar_eight_seed_means.csv) · [Paired differences](cifar_eight_seed_gaps.csv)

## CIFAR-10 paired screening-seed illustrations

At each bank size, the seven original configurations were fixed by the two-seed validation mean. The larger proposed validation margin then selected one of those shared screening seeds. Both seeds were subsequently evaluated on all seven methods with 100 fresh deployment draws. These examples use existing screened checkpoints. The confirmation cohorts above use freshly trained models.

| m | Selected seed | Selected-seed gap (pp) | Other-seed gap (pp) | Two-seed mean gap (pp) |
|---:|---:|---:|---:|---:|
| 200 | 122101 | +0.8272 | -4.0566 | -0.8452 |
| 500 | 122102 | +1.6910 | -3.5764 | -0.9427 |
| 1000 | 122101 | +2.7974 | +0.0247 | +1.4110 |

[Selection table](cifar_seed_examples/selection.csv) · [m=200 figure](cifar_seed_examples/m200/seed_illustration.pdf) · [m=500 figure](cifar_seed_examples/m500/seed_illustration.pdf) · [m=1000 figure](cifar_seed_examples/m1000/seed_illustration.pdf)

## Earlier i.i.d. fading reference

The prior five-seed study used m=20 and independent gains. Its matched T1/T2 mean led on MNIST and Fashion-MNIST. L2 led on CIFAR-10 and two moons. The new grid uses correlated gains, rho_c=0.35. [Historical results](historical_iid_m20.csv)

[Study design and analysis rules](comparison_protocol.md)
