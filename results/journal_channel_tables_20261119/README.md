# Harder fading and 50% erasure

[Common-protocol bank-size tables (one PDF)](tables/common/channel_accuracy_all.pdf) · [Illustrations including validation-selected seeds (one PDF)](tables/validation_examples/channel_accuracy_all.pdf) · [Table index, exact CSVs and LaTeX fragments](tables/README.md)

Fading uses correlation 0.85, mean gain 0.5, unit gain variance and independent Gaussian noise of standard deviation 2. Erasure independently drops each coordinate with probability 0.50. The journal tables give test accuracy in percent ± sample SD over paired training seeds. The largest unrounded displayed mean is bold.

The three principal sizes are 200, 500 and 1000. The additional 100 reference covers CIFAR-10, and 2000 covers all four datasets. The common-protocol tables preserve the broadest shared validation design within each dataset and channel. The illustrative tables use observed rankings across trials. The seed-illustration tables retain every full-pool result and label any table row that uses three seeds selected by validation.

| m | Fading, common | Erasure, common | Fading, illustrative | Erasure, illustrative including seed examples |
|---:|---:|---:|---:|---:|
| 200 | 4/4 | 2/4 | 4/4 | 4/4 |
| 500 | 4/4 | 1/4 | 4/4 | 4/4 |
| 1000 | 2/4 | 1/4 | 4/4 | 4/4 |
| 100 | 1/1 | 1/1 | 1/1 | 1/1 |
| 2000 | 2/4 | 4/4 | 2/4 | 4/4 |

Counts give the datasets in which matched T1 or T2 strictly exceeds the other five method labels. [All means, paired margins, phases, seeds and sources](tables/data/table_rankings.csv)

## Channel effects

[Paired channel effects figure](fading_stress/channel_severity.pdf) · [All effect sizes](fading_stress/channel_effects.csv) · [Degradation and recovery report](fading_stress/README.md)

The same 525 original checkpoints were evaluated under the original law, stronger correlation only, higher noise only, and both changes together. The matched retraining recovery used the same 24 weight/channel random streams. The frozen comparison holds each transmitter fixed. The retraining report also gives the observed feature power per coordinate.

## Provenance

The complete study includes 81 ordinary confirmation cohorts. The archive in audit/ contains their manifests, validation choices and numerical deployment reports. The separate precision checks and illustrative seed pools retain their own protocols and their per-seed observations. An earlier project file-count quota interrupted some tasks. Completed results were retained; affected indices were retried with the original configurations and seeds. [Archive and working-copy locations](audit/spartan_storage_notes.md) · [Final verification](audit/release_check.txt)
