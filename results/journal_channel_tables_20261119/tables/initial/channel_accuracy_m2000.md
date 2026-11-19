### m = 2000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.66 ± 0.14 | **65.98** ± 0.30 | 65.67 ± 0.58 | 65.67 ± 0.58 | 57.42 ± 0.30 |
|  | Erasure | 86.87 ± 0.20 | 86.87 ± 0.29 | 87.10 ± 0.36 | **87.15** ± 0.33 | 82.68 ± 0.47 |
| MNIST | Fading | **78.55** ± 0.05 | 76.97 ± 0.23 | 78.26 ± 0.34 | 78.33 ± 0.30 | 45.38 ± 0.38 |
|  | Erasure | **95.71** ± 0.06 | 94.94 ± 0.21 | 95.12 ± 0.04 | 95.11 ± 0.19 | 87.11 ± 0.37 |
| Fashion-MNIST | Fading | 68.29 ± 0.12 | 68.49 ± 0.07 | 68.53 ± 0.48 | **68.58** ± 0.19 | 41.49 ± 0.87 |
|  | Erasure | **85.93** ± 0.15 | 85.63 ± 0.27 | 85.82 ± 0.20 | 85.68 ± 0.13 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | 50.10 ± 1.72 | 37.94 ± 24.20 | **50.55** ± 1.71 | 49.99 ± 2.46 | 16.62 ± 0.86 |
|  | Erasure | 64.10 ± 0.65 | **64.61** ± 0.47 | 63.89 ± 1.06 | 64.21 ± 0.93 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Fading uses 3 paired training seeds and erasure uses 5.
Bold marks the highest unrounded mean in each displayed row.
