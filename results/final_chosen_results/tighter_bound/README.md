# Tighter two-moons bound decomposition (branch `tighter-bound-claude`)

This uses the same S, bank U, seeds, channels, width d = 8, receiver cap L = 0.075, Q_W = N(0, 0.24² I)
and confidence allocation as the archived figure. Only the posterior (training configuration) changes.
The exact certificates are Corollary 1 (perfect), Theorem 1 with the fresh Q_{W|S}, and Theorem 2.

Code: `fading_unit_power/tighter_bound.py` (training and certificates) and
`fading_unit_power/tighter_bound_frontier.py` (trade-off plot). Spartan run:
`/data/gpfs/projects/punim2792/yangshuo/tighter-bound-20261126/surrogate`. Posterior and Q_{W|S}
checkpoints (`*.pt`) stay there.

## What changed

1. **Posterior training.** Both the means and the per-weight standard deviations are trained. The
   objective is the bounded loss plus λ × (the exact complexity terms of the certificate), with
   λ ∈ {1, 0.3, 0.1, 0.03, 0}. The candidate grid also covers 3 initial standard deviations, 2
   learning rates and 5 checkpoints. There is no mean clip.
2. **Surrogate receiver cap during training.** Training may use 8× or 32× the cap, but every
   certificate uses L = 0.075. The projection scales w2 and b2 together, so accuracy is unchanged.
   Since the certified architecture is the archived one, no extra union bound is needed.
3. **Fresh Q_{W|S}.** Q_{W|S} = N(μ_S, (c σ_S)²) with c ∈ {1, 1.5, 2}, so K = 3. Here (μ_S, σ_S) is a
   stage-1 posterior trained on S with fresh P_tr draws, never U. Its hyperparameters are chosen by an
   S-only criterion (fresh-channel empirical risk + sample term). Stage 2 starts from it.
4. **Corollary 1 for fading.** The archived value used the second inequality with Q = P_iid, which
   gives 0.596. The first inequality with Jensen gives K·sqrt(E D²) = 0.138, and the second inequality
   with Q = P_ch gives 0.139. All three are deterministic, so the smallest is used.
5. **Monte Carlo.** The empirical risk uses 2,000,000 i.i.d. draws, so the Hoeffding radius drops from
   0.0129 to 0.0013. The population loss uses 10,000 test points × 100 draws under the deployment law.

The bounds hold simultaneously for every posterior, so any selection rule is valid. Each rule picks
with one Monte Carlo seed and certifies with an independent one:

* `tightest`: the smallest bound.
* `balanced`: the smallest bound within 3 points of the best validation accuracy.
* `accurate`: the smallest bound within 1 point of the best validation accuracy.

## Results (bound / population bounded loss / test accuracy %)

The bounds include the Monte Carlo radius. The `only_fresh` population loss is about 0.147 in every
case.

| case | only_fresh | tightest | balanced | accurate |
|---|---|---|---|---|
| fading/perfect (C1) | 0.918 / — / 56.1 | 0.367 / 0.1485 / 50.1 | 0.438 / 0.1468 / 56.2 | 0.515 / 0.1462 / 57.8 |
| fading/t1_matched | 0.381 / — / 55.7 | 0.287 / 0.1485 / 50.1 | 0.357 / 0.1476 / 54.0 | 0.485 / 0.1474 / 57.7 |
| fading/t2_matched | 0.361 / — / 56.1 | 0.260 / 0.1485 / 50.1 | 0.381 / 0.1472 / 56.5 | 0.538 / 0.1472 / 58.5 |
| fading/t1_mismatch | 0.665 / — / 55.6 | 0.573 / 0.1485 / 50.1 | 0.634 / 0.1478 / 53.2 | 0.772 / 0.1474 / 57.8 |
| fading/t2_mismatch | 0.645 / — / 56.0 | 0.545 / 0.1485 / 50.1 | 0.665 / 0.1467 / 56.8 | 0.825 / 0.1472 / 58.5 |
| erasure/perfect (C1) | 0.370 / — / 74.5 | 0.274 / 0.1484 / 50.0 | 0.382 / 0.1460 / 74.9 | 0.432 / 0.1461 / 77.6 |
| erasure/t1_matched | 0.378 / — / 72.3 | 0.287 / 0.1484 / 50.0 | 0.427 / 0.1463 / 80.5 | 0.481 / 0.1459 / 83.2 |
| erasure/t2_matched | 0.359 / — / 72.0 | 0.260 / 0.1484 / 50.0 | 0.448 / 0.1460 / 82.1 | 0.586 / 0.1458 / 84.4 |
| erasure/t1_mismatch | 0.520 / — / 72.3 | 0.429 / 0.1484 / 50.0 | 0.555 / 0.1464 / 79.4 | 0.711 / 0.1459 / 84.4 |
| erasure/t2_mismatch | 0.503 / — / 72.7 | 0.402 / 0.1484 / 50.0 | 0.590 / 0.1460 / 82.7 | 0.728 / 0.1459 / 84.4 |

**Final choice: `tightest`.** The paper figure is
`bounds/bound_decomposition_d8_t1_t2_both_channels_tightest.*`, redrawn identically by
`../plot_bound_decomposition_d8.py` as `../bound_decomposition_d8_t1_t2_both_channels.*`, with y-axis limits of 0.65 for panel
(a), fading, and 0.5 for panel (b), erasure. Its bar labels omit the Monte Carlo radius (0.0013),
like the archived figure.

Figures: `bounds/bound_decomposition_d8_t1_t2_both_channels_<rule>.*` and
`bound_accuracy_tradeoff.*`. In the trade-off plot, the archived points use test accuracy and the
fronts use validation accuracy.

## Reading

* **With L = 0.075 the population loss barely moves.** It stays between 0.146 and 0.1485 (chance is
  0.1483) because the logits are tiny. Learning therefore buys almost nothing in bounded loss and
  always costs KL, so the bound-optimal posterior is essentially the prior: `tightest` sits at chance
  accuracy.
* **The tightest bounds approach their floors.** The gap to the population loss is 0.11 (T2 matched),
  0.14 (T1 matched), 0.13/0.22 (C1 erasure/fading) and 0.25–0.42 (mismatch). With n = 800 and
  m = 2000 these are close to the floors: about 0.082 for the sample term, about 0.055 for the channel
  term and about 0.110 for the joint term at KL = 0. Mismatch adds the fixed T1/T2 shift of 0.285
  (fading) or 0.142 (erasure).
* **At the archived accuracy, the archived posteriors already lie on the bound–accuracy front.** The
  exception is C1 fading, where the Corollary 1 inequality choice cuts the bound from 0.918 to 0.438 at
  the same 56% accuracy. Elsewhere, the gain at matched accuracy is about the Monte Carlo radius
  (0.011).
* **The surrogate cap makes erasure much more accurate (80–84% vs 72%) at a looser bound.**
