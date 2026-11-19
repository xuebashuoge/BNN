### m = 1000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | **65.54** ± 0.25 | 65.03 ± 0.61 | 65.51 ± 0.88 | 65.51 ± 0.88 | 56.96 ± 1.18 |
|  | Erasure | 86.98 ± 0.21 | 86.87 ± 0.24 | 87.00 ± 0.35 | **87.04** ± 0.32 | 82.68 ± 0.47 |
| MNIST | Fading | **77.63** ± 0.44 | 76.58 ± 0.52 | 77.56 ± 0.59 | 77.52 ± 0.35 | 53.87 ± 0.79 |
|  | Erasure | 96.18 ± 0.07 | **96.21** ± 0.06 | 96.19 ± 0.06 | 96.09 ± 0.04 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 67.28 ± 0.36 | 67.63 ± 0.21 | **67.80** ± 0.24 | 67.52 ± 0.17 | 40.82 ± 0.40 |
|  | Erasure | **86.30** ± 0.15 | 86.20 ± 0.13 | 86.30 ± 0.16 | 86.20 ± 0.16 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 51.89 ± 2.66 | 53.33 ± 0.63 | 51.64 ± 2.87 | **53.79** ± 1.62 | 18.72 ± 0.61 |
|  | Erasure | 64.80 ± 1.09 | **64.94** ± 0.78 | 63.47 ± 1.95 | 64.47 ± 0.48 | 37.07 ± 1.94 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 9/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.
