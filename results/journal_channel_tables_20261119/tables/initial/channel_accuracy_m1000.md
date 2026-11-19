### m = 1000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.60 ± 0.51 | 65.82 ± 0.81 | **65.84** ± 0.81 | 65.84 ± 0.81 | 57.34 ± 1.42 |
|  | Erasure | 86.98 ± 0.21 | 86.87 ± 0.24 | 87.00 ± 0.35 | **87.04** ± 0.32 | 82.68 ± 0.47 |
| MNIST | Fading | 77.76 ± 0.32 | 76.28 ± 0.17 | **77.82** ± 0.37 | 77.71 ± 0.11 | 45.38 ± 0.38 |
|  | Erasure | **95.52** ± 0.23 | 94.80 ± 0.16 | 95.22 ± 0.19 | 95.11 ± 0.17 | 87.11 ± 0.37 |
| Fashion-MNIST | Fading | 67.28 ± 0.36 | 67.63 ± 0.21 | **67.80** ± 0.24 | 67.52 ± 0.17 | 40.82 ± 0.40 |
|  | Erasure | 85.80 ± 0.23 | 85.72 ± 0.11 | **85.88** ± 0.16 | 85.81 ± 0.25 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | 52.15 ± 3.00 | 52.16 ± 1.99 | 52.74 ± 1.29 | **53.10** ± 2.50 | 16.39 ± 0.52 |
|  | Erasure | 64.11 ± 1.01 | **65.17** ± 0.79 | 64.21 ± 0.85 | 64.54 ± 0.43 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/5, Fashion-MNIST 9/5, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
