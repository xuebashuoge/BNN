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

### m = 100

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | Fading | 43.93 ± 1.84 | 45.55 ± 0.78 | 42.26 ± 1.48 | **46.88** ± 3.18 | 16.34 ± 1.30 |
|  | Erasure | 60.94 ± 1.30 | 62.08 ± 2.19 | 62.83 ± 2.04 | **62.89** ± 1.34 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Fading uses 3 paired training seeds and erasure uses 5.
Bold marks the highest unrounded mean in each displayed row.

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
