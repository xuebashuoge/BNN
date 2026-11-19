### m = 200

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.25 ± 0.60 | 65.29 ± 0.70 | **65.46** ± 0.73 | 65.30 ± 0.67 | 56.96 ± 1.18 |
|  | Erasure | 86.95 ± 0.44 | 86.92 ± 0.23 | 86.92 ± 0.61 | **86.95** ± 0.65 | 82.68 ± 0.47 |
| MNIST | Fading | 73.66 ± 0.47 | 72.65 ± 0.63 | 73.75 ± 0.56 | **73.83** ± 0.59 | 53.87 ± 0.79 |
|  | Erasure | 95.92 ± 0.05 | **95.98** ± 0.07 | 95.94 ± 0.07 | 95.96 ± 0.08 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 61.30 ± 0.68 | 61.14 ± 0.93 | 61.47 ± 0.80 | **61.54** ± 0.84 | 40.82 ± 0.40 |
|  | Erasure | **86.09** ± 0.13 | 85.99 ± 0.20 | 85.96 ± 0.17 | 86.08 ± 0.12 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 34.24 ± 22.28 | 41.89 ± 10.42 | **44.53** ± 4.47 | 34.31 ± 22.30 | 18.72 ± 0.61 |
|  | Erasure | 64.37 ± 0.83 | 64.36 ± 0.62 | **64.86** ± 0.56 | 63.60 ± 1.25 | 37.07 ± 1.94 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 9/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.

### m = 500

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.48 ± 0.63 | 65.55 ± 0.43 | 65.55 ± 0.41 | **65.57** ± 0.43 | 56.96 ± 1.18 |
|  | Erasure | 86.91 ± 0.39 | 86.87 ± 0.26 | 87.00 ± 0.39 | **87.03** ± 0.43 | 82.68 ± 0.47 |
| MNIST | Fading | 76.09 ± 0.41 | 75.60 ± 0.50 | **76.30** ± 0.53 | 76.15 ± 0.42 | 53.87 ± 0.79 |
|  | Erasure | **96.15** ± 0.07 | 96.15 ± 0.07 | 96.15 ± 0.09 | 96.11 ± 0.09 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 66.14 ± 0.15 | 66.13 ± 0.21 | **66.23** ± 0.18 | 66.08 ± 0.30 | 40.82 ± 0.40 |
|  | Erasure | **86.18** ± 0.14 | 86.14 ± 0.15 | 86.17 ± 0.13 | 86.13 ± 0.21 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 49.35 ± 2.44 | 52.47 ± 0.94 | 50.53 ± 1.54 | **52.63** ± 3.14 | 18.72 ± 0.61 |
|  | Erasure | **64.36** ± 0.53 | 61.13 ± 9.38 | 53.94 ± 24.57 | 63.57 ± 3.14 | 37.07 ± 1.94 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 9/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.

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
| Two-moon | Fading | 65.64 ± 0.35 | **65.71** ± 0.40 | 65.61 ± 0.40 | 65.61 ± 0.40 | 57.09 ± 0.64 |
|  | Erasure | 86.87 ± 0.20 | 86.87 ± 0.29 | 87.10 ± 0.36 | **87.15** ± 0.33 | 82.68 ± 0.47 |
| MNIST | Fading | **78.43** ± 0.39 | 77.44 ± 0.47 | 78.39 ± 0.59 | 78.38 ± 0.57 | 47.99 ± 0.72 |
|  | Erasure | 96.24 ± 0.04 | 96.24 ± 0.05 | **96.29** ± 0.03 | 96.23 ± 0.07 | 87.59 ± 0.58 |
| Fashion-MNIST | Fading | 68.29 ± 0.12 | 68.49 ± 0.07 | 68.53 ± 0.48 | **68.58** ± 0.19 | 41.49 ± 0.87 |
|  | Erasure | 86.30 ± 0.13 | 86.23 ± 0.12 | 86.27 ± 0.14 | **86.32** ± 0.20 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 50.10 ± 1.72 | 37.94 ± 24.20 | **50.55** ± 1.71 | 49.99 ± 2.46 | 16.62 ± 0.86 |
|  | Erasure | 65.23 ± 0.95 | 65.10 ± 1.13 | 65.23 ± 0.70 | **65.34** ± 0.53 | 37.07 ± 1.94 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 9/5, Fashion-MNIST 3/9, CIFAR-10 3/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.
