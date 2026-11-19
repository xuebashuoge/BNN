### m = 500

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.48 ± 0.63 | 65.55 ± 0.43 | 65.55 ± 0.41 | **65.57** ± 0.43 | 56.96 ± 1.18 |
|  | Erasure | 86.91 ± 0.39 | 86.87 ± 0.26 | 87.00 ± 0.39 | **87.03** ± 0.43 | 82.68 ± 0.47 |
| MNIST | Fading | 76.09 ± 0.41 | 75.60 ± 0.50 | **76.30** ± 0.53 | 76.15 ± 0.42 | 53.87 ± 0.79 |
|  | Erasure | 95.98 ± 0.08 | 95.97 ± 0.12 | 95.94 ± 0.12 | **95.98** ± 0.08 | 87.41 ± 0.52 |
| Fashion-MNIST | Fading | 66.14 ± 0.15 | 66.13 ± 0.21 | **66.23** ± 0.18 | 66.08 ± 0.30 | 40.82 ± 0.40 |
|  | Erasure | 86.21 ± 0.12 | 86.23 ± 0.13 | 86.24 ± 0.17 | **86.25** ± 0.16 | 78.27 ± 0.60 |
| CIFAR-10 | Fading | 49.35 ± 2.44 | 52.47 ± 0.94 | 50.53 ± 1.54 | **52.63** ± 3.14 | 18.72 ± 0.61 |
|  | Erasure | 64.33 ± 1.04 | 64.25 ± 1.34 | **64.59** ± 0.78 | 63.33 ± 1.60 | 32.30 ± 9.70 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 9/9, Fashion-MNIST 9/8, CIFAR-10 5/8.
Bold marks the highest unrounded mean in each displayed row.
Illustrative rounds were chosen using observed rankings.
