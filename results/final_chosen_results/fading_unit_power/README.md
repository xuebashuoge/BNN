# Selected fading studies with unit transmit power

These results use one unit of transmit power per real channel symbol. The [campaign instructions](../../../fading_unit_power/README.md) describe the normalization and the Spartan launcher.

| Output | Reproduction records |
|---|---|
| [MNIST generalization and deployment accuracy vs. $m$](mnist_analog_finite_channel_generalization.pdf) | [54 observations](data/mnist_bank/observations.csv) at 18 bank sizes, including measured transmit power and checkpoint SHA-256. |
| [Two-moons bound decomposition](bound_decomposition_d8_t1_t2_both_channels.pdf) | [Five fading certificates](data/bounds/fading) at width 8 and $m=2000$. Panel (b) uses the archived erasure certificates. |

In the MNIST sweep, the mean bounded-loss gap decreases from 0.40108 at $m=2$ to 0.01240 at $m=25600$. Mean deployment accuracy increases from 36.81% to 73.94% across those endpoints.

The complete fading bounds, including the empirical Monte Carlo correction, are 1.05897 for C1(ii), 0.70045 and 0.63717 for mismatched T1 and T2, and 0.41550 and 0.35322 for matched T1 and T2. C1(ii) is vacuous at the normalized feature radius. The figure stacks the five named components and shows the held-out Gibbs loss as a diamond. The full reports include the 0.01293 Monte Carlo correction.

The fixed studies ran on Spartan at Git revision `987c1d3` under arrays `35553567` and `35553568`. Job `35553569` rendered these figures. Full records and checkpoints are in `/data/gpfs/projects/punim2792/yangshuo/unit-power-fading-20261123-r2/fixed`. Local checkpoint copies are under `../models/fading_unit_power`. Git includes the five small [bound checkpoints](../models/fading_unit_power/bounds).
