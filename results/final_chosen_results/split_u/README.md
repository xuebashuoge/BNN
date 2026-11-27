# Split-U data-dependent prior (branch `claude-split-U`)

T1's channel prior is learned from part of the channel bank instead of fresh simulated channels.
The bank U of size m is split into U_1 = U[:m_1] and U_2 = U[m_1:], with m_2 = m − m_1.

* **Stage 1** fits the prior Q_{W|S,U_1} by PAC-Bayes on S × U_1.
* **Stage 2** fits the posterior. Its channel complexity term runs over U_2 only, so it uses m_2, and the bound's empirical risk is measured on S × U_2.
* The certificate takes a union bound over every split fraction and prior scale that was tried.
* T2 keeps the fixed prior Q_W.

Code: [`split_u.py`](../../../fading_unit_power/split_u.py) for the performance, mismatch and second-round studies, and [`split_bound.py`](../../../fading_unit_power/split_bound.py) for the bound study. Both are run by `submit_split_u.sh` / `run_split_u.slurm`.

## Deployment accuracy (`performance_m{20,200,1000,2000}/`)

Each folder has `comparison.json`, `deployment_by_seed.csv` and `performance_table.tex`. `performance_selection.json` records the chosen T1 variant and its validation leaderboard for every row, and `performance_overview.json` has the gaps per m.

Two tuning rounds were run.

1. **Round 1.** Splits of 25% and 50%, with a grid over prior scale and learning rates. Its outputs are kept in `round1/performance_*`.
2. **Second round** (`boost`), only on the rows round 1 lost. It widened the grid to:
   * splits of 25%, 50% and 75%;
   * prior scale s ∈ {10, 100, 1000};
   * channel step ×{0.1, 1, 10} and data step ×{1, 10, 100};
   * stage 2 initialized either from the prior mean or from the base initialization (`-base`).

   The variant was chosen on validation and then evaluated on the paired test seeds.

Gap = max(T1, T2) − max(ERM, L2), in percentage points.

| Row | m = 20 | m = 200 | m = 1000 | m = 2000 |
|---|---|---|---|---|
| Two moons fading | +0.02 | +0.01 | +0.08 | +0.23 |
| Two moons erasure | −0.04 | +0.31 | +0.02 | +0.28 |
| MNIST fading | −3.26 | −1.67 | −0.64 | +0.38 |
| MNIST erasure | +0.33 | +0.05 | +0.09 | +0.03 |
| Fashion-MNIST fading | −1.16 | −0.55 | +0.47 | +0.29 |
| Fashion-MNIST erasure | +1.24 | +0.44 | +0.40 | +0.46 |
| CIFAR-10 fading | −3.89 | +1.27 | +2.51 | +0.63 |
| CIFAR-10 erasure | +1.86 | +1.59 | +0.78 | +1.21 |
| **Rows won** | 4 / 8 | 6 / 8 | 7 / 8 | **8 / 8** |

* **m = 2000 meets the goal in every row.** The second round turned MNIST fading from −0.40 to +0.38.
* **The losses are fading rows at small m.** There U_1 holds only a handful of channel realizations, so the split prior learns little.
  * The validation leaderboards are flat: the top candidates are within about 0.3 points of each other.
  * Many winning variants ignore the prior mean (`-base`).
* **The fresh prior is much stronger at small m** because stage 1 sees unlimited simulated channels. It reaches 78.1% on MNIST fading at m = 20, against 64.9% for the best baseline (`t1_fresh` in `comparison.json`).

## CIFAR-10 erasure mismatch (`cifar10_channel_mismatch_all_methods.*`, `cifar10_mismatch/`)

The setup uses bank size 2000 with deployment erasure probability 0.50. ERM, L2 and T2 are the archived fits, and T1 uses the split-U prior. The selected variant is `split25-s30-e1`.

* T1 has the highest mean accuracy at 11 of 13 training erasure probabilities.
* At p_tr = 0.10, T2 is highest (43.70 vs T1 43.32).
* At p_tr = 0.05, L2 is highest (40.78 vs T1 39.71 and T2 39.73).
* At the deployment-matched p_tr = 0.50, T1 reaches 54.12% against 52.05% for L2.

## Two-moons bound decomposition (`bounds/`)

The study uses width 8 and m = 2000. The candidate grid covers six receiver caps, L ∈ {0.075, 0.25, 0.5, 1, 2, 4}, and two feature caps:

* F = 1, the archived setting, where per-coordinate power is at most 1/8;
* F = √8, where per-coordinate power is at most 1.

The union bound is therefore over G = 12 architectures. Every pooled candidate is rescored at G = 12.

At F = 1 the per-coordinate power is 1/8, about −8 dB under unit-variance fading noise. Two-moons fading then stays near 58% accuracy and near-chance loss whatever the posterior. Allowing F = √8 keeps unit transmit power and lifts fading to 62–66%.

Three selection rules are saved. `balanced` is recommended.

* **`tightest`:** smallest certified bound, with no accuracy constraint. Every case lands at chance (50% accuracy), so it is kept for reference only.
* **`balanced`:** smallest bound among posteriors within 3 points of the best validation accuracy.
* **`accurate`:** the same with a 1-point tolerance. Fading reaches about 66%, but erasure T1 mismatch becomes vacuous (2.24).

The `balanced` results are below. Chance bounded loss is 0.1484.

| Channel | Case | F | L | Bound | Test loss | Accuracy |
|---|---|---|---|---|---|---|
| Fading | C1 | √8 | 0.075 | 0.772 | 0.143 | 63.9% |
| Fading | T1 matched | √8 | 2 | 0.461 | 0.139 | 62.0% |
| Fading | T1 mismatch | √8 | 4 | 0.748 | 0.140 | 61.9% |
| Fading | T2 matched | √8 | 0.5 | 0.491 | 0.133 | 64.9% |
| Fading | T2 mismatch | √8 | 0.5 | 0.785 | 0.132 | 65.2% |
| Erasure | C1 | √8 | 0.075 | 0.783 | 0.141 | 81.9% |
| Erasure | T1 matched | 1 | 2 | 0.404 | 0.109 | 81.2% |
| Erasure | T1 mismatch | 1 | 2 | 0.550 | 0.108 | 81.3% |
| Erasure | T2 matched | 1 | 2 | 0.431 | 0.103 | 83.2% |
| Erasure | T2 mismatch | 1 | 2 | 0.573 | 0.103 | 82.7% |

* All ten bounds are non-vacuous.
* All ten test losses are below chance.
* T1 is tighter than T2 in all four matched/mismatch pairs.

Each `bounds/<rule>/<channel>/<case>/` folder holds `bound.json`, the full certificate with components, configuration and test measurements, and `posterior.pt`. `summary.json` collects every rule. The figures are `bound_decomposition_d8_t1_t2_both_channels_<rule>.{pdf,png}`.

The C1 shift term grows with F and L, so C1 picks L = 0.075. `round1/bounds/` keeps the earlier F = 1, G = 6 run, where fading stays near chance.

## Provenance

| Study | Spartan run directory | Revision |
|---|---|---|
| Performance round 1, mismatch | `/data/gpfs/projects/punim2792/yangshuo/split-u-20261127` | 07cf053 |
| Performance second round (`boost`) | same, root `boost/` (job 35881436) | f0ebe55 |
| Bounds (F, L grid) | `/data/gpfs/projects/punim2792/yangshuo/split-u-20261128` | 5f04af0 |

Each revision ran from a detached worktree `/data/gpfs/projects/punim2792/yangshuo/bnn-split-u-<sha7>`. The trained models for the performance studies stay on Spartan. The bound posteriors are small and are committed here.
