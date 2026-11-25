# T1 with a data-dependent channel prior Q_{W|S}

These are the four paper studies rerun with the revised Theorem 1. Its channel complexity uses
D(P_{W|US} || Q_{W|S}) with Q_{W|S} = N(mu_S, s^2 I), and its sample complexity keeps the fixed prior
Q_W = N(0, prior_std^2 I). Each T1 fit has two stages:

1. **Stage 1** trains a Gaussian posterior on S **without the bank U**. It uses only the sample term,
   and its mean is mu_S.
2. **Stage 2** is T1 with prior Q_{W|S}, starting from mu_S.

Stage 1 has three constructions:

| construction | what stage 1 sees besides S | extra channel draws beyond U |
|---|---|---|
| `perfect` | the noiseless channel | none |
| `independent_bank` | a bank U' of size m (channel seed + 500000) | m |
| `fresh` | new P_tr draws every minibatch | unlimited |

All three are independent of U, so all are valid priors for the theorem.

The prior scale s ∈ {3, 10, 30, 100} × posterior_std and the channel weight η ∈ {1, 10, 100} × the
archived value are picked by validation accuracy. Tuning uses seeds offset by +10M and +20M, which are
disjoint from the confirmation seeds. Confirmation reuses the archived paired seeds and evaluation
seeds, so ERM, L2 and T2 come unchanged from `../fading_unit_power/performance` and `../data`. Every
fading run keeps unit transmit power per feature, and the summaries check this.

Code: `fading_unit_power/data_dependent.py` (campaign), `fading_unit_power/dd_bounds.py` (certificates),
`fading_unit_power/trainer/theorem1.py` and `run_experiment.py` (the `channel_prior*` config keys).

## Two selections

* **This directory.** The selection may use any construction. Validation picks `fresh` almost
  everywhere. T1 then gets unlimited extra channel draws that ERM, L2 and T2 never see, so its accuracy
  lead is not a like-for-like comparison. The MNIST m-sweep is also degenerate: accuracy is about 74.5%
  and Δ(U,S) about 0.01 at every m, because stage 1 already generalizes over the channel law.
* **`only_perfect/`.** The selection is restricted to `perfect`, so T1 uses no channels beyond U. This
  is the fair comparison. It reuses the same tuning fits, and its union bound still counts all 12
  priors, which is conservative.

### Deployment accuracy at m = 2000 (`performance/`, mean of the archived seeds, %)

| row | ERM | L2 | T2 | T1 fixed Q_W | T1 any construction | T1 `perfect` only |
|---|---|---|---|---|---|---|
| Moons fading | 70.49 | 70.79 | 70.75 | 70.75 | **70.96** | 70.59 |
| Moons erasure | 86.87 | 86.87 | **87.15** | 87.10 | 86.78 | 86.67 |
| MNIST fading | 77.57 | 77.63 | 77.23 | 77.35 | **78.26** | 76.75 |
| MNIST erasure | 96.24 | 96.24 | 96.23 | 96.29 | **96.37** | 96.31 |
| F-MNIST fading | 70.08 | 70.17 | 70.12 | 70.12 | **70.97** | 70.19 |
| F-MNIST erasure | 86.30 | 86.23 | 86.32 | 86.27 | **86.74** | 86.41 |
| CIFAR-10 fading | 54.60 | 55.69 | 55.24 | 55.13 | 58.31 | **58.65** |
| CIFAR-10 erasure | 65.23 | 65.10 | 65.34 | 65.23 | **66.40** | 65.22 |

With any construction, T1 is best in 7 of 8 rows. With `perfect` only, it is best in 4 of 8 rows
(F-MNIST ×2, MNIST erasure, CIFAR-10 fading) and behind in the other 4.

### MNIST bank sweep (`mnist_bank/`, Δ(U,S) / accuracy %)

| m | fixed Q_W | `perfect` | any (`fresh`) |
|---|---|---|---|
| 2 | 0.482 / 31.3 | 0.306 / 48.0 | 0.008 / 74.4 |
| 20 | 0.278 / 59.7 | 0.259 / 56.9 | 0.009 / 74.5 |
| 100 | 0.160 / 70.3 | 0.165 / 61.9 | 0.012 / 74.5 |
| 800 | 0.040 / 77.2 | 0.051 / 72.3 | 0.012 / 74.6 |
| 25600 | 0.010 / 77.8 | 0.011 / 74.6 | 0.011 / 74.8 |

One variant is selected for all m: `perfect-s100-e1` and `fresh-s3-e10`. The `perfect` figure keeps the
paper's message.

### CIFAR-10 erasure mismatch (`cifar10_mismatch/`)

* **Any construction:** T1 is the top curve for every p_tr > 0.
* **`perfect` only:** T1 is the top curve for p_tr ≥ 0.5, peaking at 55.65% at p_tr = 0.6. At
  p_tr = 0.4 it ties L2 (51.85 vs 52.02). For 0 < p_tr ≤ 0.3 it is 1.6–3.1 points behind the best
  baseline.
* **p_tr = 0:** this bank has no channel complexity, so the archived T1 fits are reused there.

### Two-moons certificates (`bounds/`, width 8, m = 2000)

The certificate family is the three stage-1 means × s ∈ {1, 1.25, 1.5, 2, 3, 5} × posterior_std, so
K = 18. The union bound uses ε1/K. The posterior is chosen with one Monte Carlo seed (271828) and
certified with an independent one (314159). The confidence allocation is unchanged: 0.00125 for each
theorem and for each Monte Carlo estimate.

| case | T1 fixed Q_W | T1 Q_{W\|S} | T2 |
|---|---|---|---|
| fading matched | 0.406 | 0.377 | 0.361 |
| fading mismatch | 0.691 | 0.661 | 0.646 |
| erasure matched | 0.419 | 0.380 | 0.359 |
| erasure mismatch | 0.564 | 0.522 | 0.503 |

The channel complexity falls from about 0.10 to about 0.06 in every case, so T1 is tighter than before.
T2 is still tighter. The sample term (about 0.155) is unchanged, and the channel term cannot drop below
about 0.059 because of its log(√m K / ε1) part.

The figure labels leave out the Monte Carlo radius of 0.0129, as the archived figure does. The
`uncapped_risk_upper` field in each `bound.json` includes it.

## Reproduce

```bash
bash -l fading_unit_power/submit_data_dependent.sh RUN_DIR                 # all four studies
CONSTRUCTIONS=perfect bash -l fading_unit_power/submit_data_dependent.sh RUN_DIR performance cifar_mismatch mnist
```

The published runs used `RUN_DIR=/data/gpfs/projects/punim2792/yangshuo/data-dependent-prior-20261126`
on Spartan:

* revision 3551a87 for the any-construction selection;
* revision 82b6a2f for `only_perfect`.

Checkpoints stay on Spartan under `RUN_DIR/<study>/runs/`. Their SHA-256 hashes are listed in each
`observations.csv`, `deployment_by_seed.csv` and `bound.json`.
