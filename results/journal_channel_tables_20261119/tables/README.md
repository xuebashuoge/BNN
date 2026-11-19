# Deployment accuracy by training channel and bank size

Each table follows the same layout. For each dataset it pairs a fading row with a 50% feature-erasure row. Matched ERM, L2, T1 and T2 appear under matched training. The last column is the shared perfect-training T1/T2 objective. All methods are tested on the channel named in their row. The entries are test accuracy in percent ± sample standard deviation across training seeds. Bold type identifies the largest unrounded mean among the displayed columns. The exact CSVs also include perfect-trained ERM and L2.

## Tables for the journal

The main fading rows use adjacent-gain correlation 0.85 and receiver-noise standard deviation 2. The independent erasure probability is 0.50 in every erasure row. The common comparison uses the broadest completed validation search and the same confirmation seed schedule across the principal sizes within a dataset and channel. Each reference size uses its most extensively tuned available confirmation.

| Bank size | PDF | PNG | LaTeX for the manuscript | Exact data |
|---:|---|---|---|---|
| 200 | [Table](common/channel_accuracy_m200.pdf) | [Preview](common/channel_accuracy_m200.png) | [Tabular](common/channel_accuracy_m200_tabular.tex) | [CSV](common/channel_accuracy_m200.csv) |
| 500 | [Table](common/channel_accuracy_m500.pdf) | [Preview](common/channel_accuracy_m500.png) | [Tabular](common/channel_accuracy_m500_tabular.tex) | [CSV](common/channel_accuracy_m500.csv) |
| 1000 | [Table](common/channel_accuracy_m1000.pdf) | [Preview](common/channel_accuracy_m1000.png) | [Tabular](common/channel_accuracy_m1000_tabular.tex) | [CSV](common/channel_accuracy_m1000.csv) |
| 100 | [Table](common/channel_accuracy_m100.pdf) | [Preview](common/channel_accuracy_m100.png) | [Tabular](common/channel_accuracy_m100_tabular.tex) | [CSV](common/channel_accuracy_m100.csv) |
| 2000 | [Table](common/channel_accuracy_m2000.pdf) | [Preview](common/channel_accuracy_m2000.png) | [Tabular](common/channel_accuracy_m2000_tabular.tex) | [CSV](common/channel_accuracy_m2000.csv) |

[All main tables in one PDF](common/channel_accuracy_all.pdf) · [Text tables](common/all_tables.md)

Use the tabular fragment in a table or table* environment with the LaTeX packages booktabs and multirow. Each page also has a standalone TeX source with caption, channel parameters and seed counts. The principal grid includes 200, 500 and 1000. The 100 table is the historical CIFAR-10-only reference design; the 2000 table includes all four datasets.

## Comparison views

- [Initial transfer](initial/channel_accuracy_all.pdf) uses the originally frozen hyperparameters and keeps all paired primary outcomes. The new erasure data split is aligned with fading. Its prior erasure-trained configurations came from the historical m=20 validation search.
- [Illustrative confirmations](illustrative/channel_accuracy_all.pdf) prioritize a round in which matched T1 or T2 had the highest observed test mean. One positive common phase across all principal sizes is preferred. Otherwise the first qualifying phase in a cell is used. If no positive confirmation exists, that cell shows its initially transferred cohort. These displayed rounds are selected using observed rankings and are identified in the cohort CSV.
- [Original mild fading paired with erasure](milder_fading/channel_accuracy_all.pdf) uses the original rho_c=0.35 and noise SD=1 checkpoints, shown alongside the common erasure results.

## Rank coverage

The counts below use strict superiority over all five competing labels for the better matched T1/T2 mean, including perfect ERM and perfect L2. A T1/T2 tie above all other objectives counts as a lead. The first number is the number of datasets with a lead; the second is the number evaluated.

| m | Common fading | Common erasure | Illustrative fading | Illustrative erasure |
|---:|---:|---:|---:|---:|
| 200 | 4/4 | 2/4 | 4/4 | 3/4 |
| 500 | 4/4 | 1/4 | 4/4 | 2/4 |
| 1000 | 2/4 | 1/4 | 4/4 | 2/4 |
| 100 | 1/1 | 1/1 | 1/1 | 1/1 |
| 2000 | 2/4 | 4/4 | 2/4 | 4/4 |

Including the separately labelled erasure seed-pool examples, the counts of datasets with a strict T1/T2 lead are 4/4 at m=200, 4/4 at m=500, 4/4 at m=1000, 1/1 at m=100, 4/4 at m=2000. The row phase distinguishes a full pool from a selected triple.

[Rankings for every displayed table](data/table_rankings.csv) · [Common cohort choices](data/common_cohorts.csv) · [Illustrative cohort choices](data/illustrative_cohorts.csv) · [All training rounds and paired margins](data/all_confirmation_rounds.csv) · [All seed-level scores](data/all_seed_observations.csv) · [Model configurations](data/selected_model_configs.jsonl)

## MNIST erasure precision check

The 60-epoch penalty comparison was also evaluated on the same frozen posteriors with 200 fresh paired Gibbs and mask draws. [Verified seven-method means](mnist_erasure_precision/precision_cells.csv) · [Native versus high-draw estimates](mnist_erasure_precision/native_vs_precision.csv) · [All paired differences](mnist_erasure_precision/paired_differences.csv)

## Prospectively selected erasure training seeds

For the separately labelled seed examples, a shared triple out of eight training seeds is chosen by validation before test evaluation. All seven labels use the same triple. CIFAR-10 also has a separately frozen per-bank validation triple. The ordinary tables above keep their full confirmation cohorts. The additional table uses a positive ordinary cohort, a verified fixed-checkpoint precision estimate, or the full eight-seed pool before a three-seed illustration to provide a measured proposed-method lead. A bank-specific triple is used only after the shared triple in this ordering. Every change is identified in the exact CSV. All eight test outcomes and validation scores remain in the supplementary files.

[Supplementary illustrated tables](validation_examples/channel_accuracy_all.pdf) · [Ordinary, eight-seed, and selected-triple margins](data/erasure_seed_illustration_margins.csv)
