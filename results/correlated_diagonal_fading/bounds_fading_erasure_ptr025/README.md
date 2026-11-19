# Two-moons fading and erasure bounds at the 0.25 training erasure probability

Each erasure predictor deploys at a 0.50 feature-erasure probability. The matched learner trains at 0.50 and the mismatched learner trains at 0.25. Fading uses a deployment correlation of 0.35 and a mismatch training correlation of 0.15. The training set contains 800 examples and the channel bank contains 2000 draws.

## Uncapped upper bounds on bounded deployment loss

| Channel | Width | Perfect (C1) | Matched T1 | Matched T2 | Mismatched T1 | Mismatched T2 |
|---|---:|---:|---:|---:|---:|---:|
| Erasure | 16 | 0.4293 | 0.4796 | 0.4028 | 1.5606 | 1.4792 |
| Erasure | 8 | 0.3704 | 0.4185 | 0.3588 | 1.1844 | 1.1204 |
| Fading | 16 | 1.5413 | 0.4808 | 0.4033 | 0.8980 | 0.8221 |
| Fading | 8 | 0.9176 | 0.4058 | 0.3608 | 0.6912 | 0.6455 |

Perfect-trained erasure uses Corollary 1(i) and perfect-trained fading uses Corollary 1(ii). Each T1 and T2 certificate uses a posterior fitted with the corresponding theorem objective. The 20 cases use 0.00125 failure probability per theorem event and per empirical Monte Carlo estimate for a simultaneous 95% confidence budget. The Monte Carlo estimates use 20,000 observations. Held-out accuracy and loss use 20 Gibbs draws over 10,000 test examples.

The erasure mismatch adds 1.072720 to each width-16 bound and 0.758528 to each width-8 bound. All four mismatch bounds exceed the unit loss ceiling. The six erasure perfect and matched bounds and nine of the ten fading bounds are below one.

[Erasure figure](erasure_bound_decomposition.pdf) · [Erasure data](erasure_bound_decomposition.csv) · [Two-channel figure](fading_erasure_bound_decomposition.pdf) · [Two-channel data](fading_erasure_bound_decomposition.csv)

## Reproduction

On Spartan, run `bash journal/submit_correlated_bounds.sh /absolute/shared/output 0.15 both 0.25` with `DATASET` set. Training array 35405634 fitted four new erasure posteriors. Audit 35405753 verified the saved weights, training banks, empirical losses and bound terms, and confirmed byte-identical results for the 16 reused controls. The raw reports and checkpoints are in `/data/gpfs/projects/punim2792/yangshuo/erasure_fading_bound_ptr025_20261119`.
