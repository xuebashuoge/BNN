# Two-moons erasure bound decomposition at width 8

All five predictors deploy at a feature-erasure probability of 0.50. The mismatch learners train at 0.35. The model has two fully connected layers and width 8. Training uses 800 examples and a bank of 2000 channel draws.

| Training law | Training erasure probability | Certificate | Channel-shift term | Uncapped upper bound | Held-out Gibbs loss | Deployment accuracy |
|---|---:|---|---:|---:|---:|---:|
| Perfect | 0 (ideal) | Corollary 1(i) | 0.0452 | 0.3704 | 0.14652 | 74.49% |
| Matched | 0.50 | Theorem 1 | 0.0000 | 0.4185 | 0.14688 | 68.18% |
| Matched | 0.50 | Theorem 2 | 0.0000 | 0.3588 | 0.14669 | 72.01% |
| Mismatch | 0.35 | Theorem 1 | 0.4343 | 0.8573 | 0.14677 | 70.99% |
| Mismatch | 0.35 | Theorem 2 | 0.4343 | 0.7953 | 0.14664 | 72.48% |

All five upper bounds are below the unit ceiling of the bounded loss. For the 0.35 to 0.50 mismatch, the eight-component Bernoulli law gives a deployment-to-training KL divergence of 0.377243. The shift in each theorem is therefore 0.434306. In the earlier [0.25 training case](../bounds_fading_erasure_ptr025/README.md), the shift was 0.758528 and the width-8 bounds were 1.1844 for T1 and 1.1204 for T2.

[Decomposition figure (PDF)](erasure_bound_decomposition_d8.pdf) · [PNG](erasure_bound_decomposition_d8.png) · [Numerical components and accuracies (CSV)](erasure_bound_decomposition_d8.csv) · [Case records and provenance (JSON)](erasure_bound_decomposition_d8.json)

## Training and evaluation

Each learner uses its own training objective and theorem. Perfect training uses Corollary 1(i). T1 and T2 train on the same erasure bank with separate fits, paired data and evaluation seeds. All five certificates use the earlier joint-study allocation of 0.00125 for each theorem event and each Monte Carlo event. This preserves the numerical comparability of the bound components. The empirical Gibbs loss uses 20,000 independent observations. Deployment accuracy and held-out Gibbs loss average 20 draws on 10,000 held-out examples. Accuracy is estimated for one trained posterior per row.

## Reproduction

From the Spartan checkout with `DATASET` set, run `bash journal/submit_correlated_bounds.sh /absolute/shared/output 0.15 erasure 0.35 8`. The correlation argument applies to fading runs. This command creates five width-8 erasure cases with the confidence allocation used above.

In this update, Slurm array 35406036 trained the two mismatch learners. The three existing control fits and their certificates were reused. Audit 35406040 compared their reports, weights, metadata and histories byte for byte, regenerated the two training banks, reproduced both empirical losses to within 0.000000003 and verified all 40 held-out accuracy replicates. Report job 35406053 produced the figure and tables. Raw checkpoints and logs are in `/data/gpfs/projects/punim2792/yangshuo/erasure_bound_d8_ptr035_20261119`.
