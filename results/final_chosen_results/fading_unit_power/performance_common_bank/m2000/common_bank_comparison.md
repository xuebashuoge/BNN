# Common channel-bank size m = 2000

Fading uses matched diagonal fading with correlation 0.35 and unit transmit power. Each row reports 128 held-out deployment draws for the validation-selected seed. Erasure uses matched independent erasure at probability 0.50 under the original transmit-power protocol. Each erasure row reports a paired training-seed mean with 20 deployment draws per fit.

Accuracy is in percent. The gap subtracts the stronger test baseline from the validation-selected proposed method for fading or the stronger proposed mean for erasure.

| Dataset | Channel | Training seeds | ERM | L2 | T1 | T2 | Proposed | Gap (pp) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Two moons | Fading | 1 | 71.519 | 71.294 | 71.272 | 71.272 | T2 | -0.247 |
| Two moons | Erasure | 5 | 86.866 | 86.867 | 87.102 | 87.150 | T2 | +0.283 |
| MNIST | Fading | 1 | 78.243 | 78.455 | 78.622 | 78.655 | T2 | +0.200 |
| MNIST | Erasure | 5 | 96.239 | 96.239 | 96.289 | 96.229 | T1 | +0.051 |
| Fashion-MNIST | Fading | 1 | 70.752 | 70.925 | 71.003 | 71.004 | T2 | +0.078 |
| Fashion-MNIST | Erasure | 9 | 86.297 | 86.229 | 86.273 | 86.319 | T2 | +0.022 |
| CIFAR-10 | Fading | 1 | 54.677 | 55.706 | 57.394 | 55.138 | T1 | +1.688 |
| CIFAR-10 | Erasure | 5 | 65.227 | 65.101 | 65.226 | 65.336 | T2 | +0.109 |
