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

## Each construction separately (`only_perfect/`, `only_independent_bank/`, `only_fresh/`)

Each folder selects s and η by the same validation rule, but only among one construction. It reuses
the shared tuning fits.

### Deployment accuracy at m = 2000 (%)

`*` marks a row where T1 beats the best of ERM, L2 and T2. The selected (s, η) is in brackets.

| row | ERM | L2 | T2 | T1 fixed Q_W | `perfect` | `independent_bank` | `fresh` |
|---|---|---|---|---|---|---|---|
| Moons fading | 70.49 | 70.79 | 70.75 | 70.75 | 70.59 (s10, e10) | **71.18*** (s10, e10) | 70.96* (s100, e100) |
| Moons erasure | 86.87 | 86.87 | **87.15** | 87.10 | 86.67 (s100, e1) | 86.78 (s100, e100) | 86.88 (s30, e10) |
| MNIST fading | 77.57 | 77.63 | 77.23 | 77.35 | 76.75 (s100, e1) | 77.46 (s10, e1) | **78.26*** (s3, e1) |
| MNIST erasure | 96.24 | 96.24 | 96.23 | 96.29 | 96.31* (s100, e1) | 96.32* (s30, e100) | **96.37*** (s10, e10) |
| F-MNIST fading | 70.08 | 70.17 | 70.12 | 70.12 | 70.19* (s100, e1) | 70.79* (s30, e1) | **70.97*** (s3, e100) |
| F-MNIST erasure | 86.30 | 86.23 | 86.32 | 86.27 | 86.41* (s100, e1) | 86.67* (s100, e10) | **86.74*** (s100, e1) |
| CIFAR-10 fading | 54.60 | 55.69 | 55.24 | 55.13 | **58.65*** (s30, e10) | 56.25* (s100, e100) | 58.31* (s10, e10) |
| CIFAR-10 erasure | 65.23 | 65.10 | 65.34 | 65.23 | 65.22 (s100, e1) | **66.40*** (s100, e1) | 66.16* (s100, e1) |
| **rows won** | | | | | 4 / 8 | 6 / 8 | 7 / 8 |

### MNIST bank sweep (Δ(U,S) / accuracy %)

The selected variants are `perfect-s100-e1`, `independent_bank-s30-e100` and `fresh-s3-e10`.

| m | fixed Q_W | `perfect` | `independent_bank` | `fresh` |
|---|---|---|---|---|
| 2 | 0.482 / 31.3 | 0.306 / 48.0 | 0.257 / 43.7 | 0.008 / 74.4 |
| 5 | 0.379 / 44.6 | 0.296 / 51.0 | 0.220 / 51.2 | 0.006 / 74.4 |
| 20 | 0.278 / 59.7 | 0.259 / 56.9 | 0.170 / 56.5 | 0.009 / 74.5 |
| 100 | 0.160 / 70.3 | 0.165 / 61.9 | 0.099 / 63.0 | 0.012 / 74.5 |
| 400 | 0.067 / 75.8 | 0.094 / 69.8 | 0.040 / 70.6 | 0.015 / 74.6 |
| 800 | 0.040 / 77.2 | 0.051 / 72.3 | 0.025 / 73.0 | 0.012 / 74.6 |
| 3200 | 0.014 / 78.3 | 0.020 / 74.1 | 0.013 / 74.2 | 0.011 / 74.9 |
| 25600 | 0.010 / 77.8 | 0.011 / 74.6 | 0.012 / 74.4 | 0.011 / 74.8 |

`independent_bank` gives the smallest Δ(U,S) for every m from 5 to 800 while keeping the trend over m.
`fresh` is flat over m.

### CIFAR-10 erasure mismatch (accuracy %)

The selected variants are `perfect-s100-e1`, `independent_bank-s100-e1` and `fresh-s100-e10`.

| p_tr | ERM | L2 | T2 | `perfect` | `independent_bank` | `fresh` |
|---|---|---|---|---|---|---|
| 0.025 | 37.52 | 36.28 | 38.16 | 35.60 | 38.18 | 38.88 |
| 0.05 | 40.10 | 40.78 | 39.73 | 37.83 | 40.40 | 40.16 |
| 0.1 | 43.10 | 43.07 | 43.70 | 40.80 | 44.44 | 44.18 |
| 0.2 | 45.85 | 46.08 | 47.89 | 44.80 | 48.49 | 48.52 |
| 0.3 | 48.62 | 48.79 | 50.45 | 48.12 | 51.44 | 52.01 |
| 0.4 | 50.48 | 52.02 | 51.29 | 51.85 | 53.92 | 54.40 |
| 0.5 | 50.42 | 52.05 | 51.29 | 54.36 | 54.69 | 54.58 |
| 0.6 | 49.76 | 52.48 | 52.47 | 55.65 | 54.95 | 54.80 |
| 0.7 | 47.37 | 49.65 | 50.03 | 55.14 | 52.31 | 52.43 |
| 0.8 | 43.87 | 46.63 | 45.25 | 51.99 | 49.96 | 49.68 |
| 0.9 | 32.68 | 39.33 | 36.83 | 43.02 | 45.46 | 44.89 |

p_tr = 0.15 is omitted here (ERM / L2 / T2 / `perfect` / `independent_bank` / `fresh` =
44.57 / 46.10 / 46.30 / 43.15 / 46.16 / 46.84). For `independent_bank`, T1 is best at every p_tr except
0.05 (L2 leads by 0.38) and 0.15 (T2 leads by 0.14). For `fresh`, T1 is best at every p_tr except 0.05,
where L2 leads by 0.62.

### Two-moons certificates

Each folder certifies with that construction's 6 scales only, so K = 6. Each certificate is valid on
its own if its construction was fixed in advance. Choosing the best of the three afterwards needs the
K = 18 certificate in `bounds/` above.

| case | T1 fixed Q_W | `perfect` | `independent_bank` | `fresh` | T2 |
|---|---|---|---|---|---|
| fading matched | 0.406 | 0.379 | **0.375** | 0.381 | 0.361 |
| fading mismatch | 0.691 | 0.665 | **0.659** | 0.665 | 0.646 |
| erasure matched | 0.419 | 0.382 | 0.378 | **0.378** | 0.359 |
| erasure mismatch | 0.564 | 0.523 | 0.521 | **0.520** | 0.503 |

The three constructions give certificates within 0.006 of each other, and all are tighter than the
fixed Q_W. The construction matters for accuracy, not for the bound.

## Reproduce

```bash
bash -l fading_unit_power/submit_data_dependent.sh RUN_DIR                 # all four studies
for c in perfect independent_bank fresh; do                               # per-construction folders
    CONSTRUCTIONS=$c bash -l fading_unit_power/submit_data_dependent.sh RUN_DIR
done
```

The published runs used `RUN_DIR=/data/gpfs/projects/punim2792/yangshuo/data-dependent-prior-20261126`
on Spartan:

* revision 3551a87 for the any-construction selection;
* revision 82b6a2f for the `only_perfect` table, MNIST and CIFAR-10 folders;
* revision af476c0 for the other `only_*` table, MNIST and CIFAR-10 folders;
* revision cc409ab for every `only_*/bounds` folder.

Checkpoints stay on Spartan under `RUN_DIR/<study>/runs/`. Their SHA-256 hashes are listed in each
`observations.csv`, `deployment_by_seed.csv` and `bound.json`.
