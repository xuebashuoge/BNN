# Two-moons fading certificates

The test channel has adjacent gain correlation 0.35. Matched training uses 0.35, mismatched training uses 0.15, and perfect training uses the identity channel and zero noise. Each stochastic gain has mean 0.5 and variance one and the additive noise has unit variance. Both widths use 800 i.i.d. training examples and a bank size of 2000.

Each row reports the certificate evaluated on the posterior produced by its own stated training objective. Bounds apply to normalized smooth bounded Gibbs loss on a 0–1 scale. Test accuracies are descriptive and use 10,000 held-out examples and 20 Gibbs draws. The confidence budget is simultaneous over the ten reported bounds.

| Training and evaluation | Bound at width 16 | Test accuracy (%) | Bound at width 8 | Test accuracy (%) |
|---|---:|---:|---:|---:|
| Perfect → C1(ii) | 1.53955 | 56.3705 | 0.91561 | 56.1365 |
| Matched T1 → Theorem 1 | 0.47835 | 55.4105 | 0.40276 | 53.6200 |
| Matched T2 → Theorem 2 | 0.40141 | 55.7885 | 0.35858 | 56.0565 |
| Mismatched T1 → Theorem 1 | 0.89547 | 55.2825 | 0.68812 | 53.6110 |
| Mismatched T2 → Theorem 2 | 0.82021 | 55.8270 | 0.64325 | 55.9735 |

The mismatch adds 0.417763 to the width-16 bounds and 0.285387 to the width-8 bounds. Within each stochastic law and width, T1 and T2 use the same training-bank realization, data split and random seed. Their fitted posterior means differ. Perfect training uses the i.i.d. reference only in C1(ii)'s analytic calculation.

[Stacked-bar figure (PDF)](correlated_bound_decomposition.pdf) · [PNG](correlated_bound_decomposition.png) · [Full-precision per-case decomposition (CSV)](correlated_bound_decomposition.csv) · [Case provenance (JSON)](correlated_bound_decomposition.json)

The [study derivation](../../../journal/DIAGONAL_FADING_BOUNDS.md) gives the objectives, complexity terms and confidence allocation. Spartan training array `35397366` completed all ten fits. Independent verification job `35397412` checked checkpoint fingerprints, channel banks, posterior KLs, directed channel KLs and bound arithmetic. Figure and test job `35397439` produced the checked release. The checkpoints and individual reports reside at `/data/gpfs/projects/punim2792/yangshuo/bound_corr_mismatch_20261119/by_method`.
