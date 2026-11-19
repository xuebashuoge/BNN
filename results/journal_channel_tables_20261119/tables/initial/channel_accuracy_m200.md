### m = 200

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | **65.13** ± 0.38 | 64.42 ± 0.87 | 64.93 ± 0.39 | 64.93 ± 0.39 | 57.34 ± 1.42 |
|  | Erasure | 86.95 ± 0.44 | 86.92 ± 0.23 | 86.92 ± 0.61 | **86.95** ± 0.65 | 82.68 ± 0.47 |
| MNIST | Fading | 73.64 ± 0.71 | 72.52 ± 0.77 | 73.67 ± 0.36 | **73.78** ± 0.60 | 45.38 ± 0.38 |
|  | Erasure | **95.41** ± 0.12 | 94.63 ± 0.20 | 94.89 ± 0.21 | 94.90 ± 0.22 | 87.11 ± 0.37 |
| Fashion-MNIST | Fading | 61.30 ± 0.68 | 61.14 ± 0.93 | 61.47 ± 0.80 | **61.54** ± 0.84 | 40.82 ± 0.40 |
|  | Erasure | 85.58 ± 0.36 | 85.47 ± 0.24 | **85.65** ± 0.13 | 85.50 ± 0.21 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | **46.66** ± 3.64 | 44.73 ± 4.61 | 43.83 ± 7.44 | 40.41 ± 17.16 | 16.39 ± 0.52 |
|  | Erasure | 62.77 ± 0.77 | **63.93** ± 0.41 | 63.23 ± 1.42 | 63.07 ± 1.21 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/5, Fashion-MNIST 9/5, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
