### m = 500

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | **65.85** ± 0.13 | 65.47 ± 0.55 | 65.79 ± 0.20 | 65.79 ± 0.20 | 57.34 ± 1.42 |
|  | Erasure | 86.91 ± 0.39 | 86.87 ± 0.26 | 87.00 ± 0.39 | **87.03** ± 0.43 | 82.68 ± 0.47 |
| MNIST | Fading | **76.68** ± 0.52 | 75.76 ± 0.27 | 76.48 ± 0.44 | 76.34 ± 0.36 | 45.38 ± 0.38 |
|  | Erasure | **95.45** ± 0.10 | 94.67 ± 0.15 | 95.08 ± 0.04 | 95.19 ± 0.09 | 87.11 ± 0.37 |
| Fashion-MNIST | Fading | 66.14 ± 0.15 | 66.13 ± 0.21 | **66.23** ± 0.18 | 66.08 ± 0.30 | 40.82 ± 0.40 |
|  | Erasure | **85.85** ± 0.24 | 85.63 ± 0.20 | 85.84 ± 0.24 | 85.67 ± 0.24 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | 51.01 ± 3.10 | 43.43 ± 13.14 | 50.55 ± 3.06 | **52.36** ± 2.99 | 16.39 ± 0.52 |
|  | Erasure | 63.85 ± 0.69 | **64.77** ± 0.76 | 64.72 ± 0.60 | 64.43 ± 0.48 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/5, Fashion-MNIST 9/5, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
