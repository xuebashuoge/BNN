### m = 200

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 73.57 ± 0.94 | 73.46 ± 1.11 | 73.61 ± 1.17 | **73.61** ± 1.17 | 63.44 ± 1.23 |
|  | Erasure | 86.95 ± 0.44 | 86.92 ± 0.23 | 86.92 ± 0.61 | **86.95** ± 0.65 | 82.68 ± 0.47 |
| MNIST | Fading | 78.46 ± 1.00 | 77.79 ± 1.31 | **78.64** ± 0.57 | 78.39 ± 1.01 | 57.41 ± 0.44 |
|  | Erasure | 95.92 ± 0.05 | **95.98** ± 0.07 | 95.94 ± 0.07 | 95.96 ± 0.08 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 68.56 ± 0.28 | 68.51 ± 0.26 | **68.83** ± 0.18 | 68.82 ± 0.18 | 50.76 ± 0.43 |
|  | Erasure | **86.09** ± 0.13 | 85.99 ± 0.20 | 85.96 ± 0.17 | 86.08 ± 0.12 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 44.29 ± 19.20 | **53.28** ± 3.06 | 51.37 ± 3.31 | 51.43 ± 5.07 | 18.19 ± 0.81 |
|  | Erasure | 64.37 ± 0.83 | 64.36 ± 0.62 | **64.86** ± 0.56 | 63.60 ± 1.25 | 37.07 ± 1.94 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.

### m = 500

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 74.46 ± 0.97 | 74.38 ± 1.05 | 74.48 ± 1.13 | **74.48** ± 1.13 | 63.44 ± 1.23 |
|  | Erasure | 86.91 ± 0.39 | 86.87 ± 0.26 | 87.00 ± 0.39 | **87.03** ± 0.43 | 82.68 ± 0.47 |
| MNIST | Fading | 81.50 ± 0.34 | 80.92 ± 0.45 | **81.60** ± 0.28 | 81.33 ± 0.51 | 57.41 ± 0.44 |
|  | Erasure | **96.15** ± 0.07 | 96.15 ± 0.07 | 96.15 ± 0.09 | 96.11 ± 0.09 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 72.04 ± 0.35 | 72.29 ± 0.28 | 72.28 ± 0.37 | **72.52** ± 0.29 | 50.76 ± 0.43 |
|  | Erasure | **86.18** ± 0.14 | 86.14 ± 0.15 | 86.17 ± 0.13 | 86.13 ± 0.21 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 46.18 ± 20.33 | 54.07 ± 5.43 | 54.52 ± 3.18 | **55.56** ± 4.61 | 18.19 ± 0.81 |
|  | Erasure | **64.36** ± 0.53 | 61.13 ± 9.38 | 53.94 ± 24.57 | 63.57 ± 3.14 | 37.07 ± 1.94 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.

### m = 1000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | **74.34** ± 0.99 | 73.98 ± 1.54 | 74.34 ± 0.99 | 74.34 ± 0.99 | 63.44 ± 1.23 |
|  | Erasure | 86.98 ± 0.21 | 86.87 ± 0.24 | 87.00 ± 0.35 | **87.04** ± 0.32 | 82.68 ± 0.47 |
| MNIST | Fading | 82.82 ± 1.07 | 82.24 ± 0.33 | **83.07** ± 0.99 | 82.81 ± 0.46 | 57.41 ± 0.44 |
|  | Erasure | 96.18 ± 0.07 | **96.21** ± 0.06 | 96.19 ± 0.06 | 96.09 ± 0.04 | 87.25 ± 0.46 |
| Fashion-MNIST | Fading | 73.54 ± 0.26 | 73.85 ± 0.27 | 73.86 ± 0.29 | **73.99** ± 0.32 | 50.76 ± 0.43 |
|  | Erasure | **86.30** ± 0.15 | 86.20 ± 0.13 | 86.30 ± 0.16 | 86.20 ± 0.16 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 57.08 ± 1.17 | 56.75 ± 1.25 | **57.15** ± 1.13 | 57.02 ± 1.22 | 18.19 ± 0.81 |
|  | Erasure | 64.80 ± 1.09 | **64.94** ± 0.78 | 63.47 ± 1.95 | 64.47 ± 0.48 | 37.07 ± 1.94 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/9, Fashion-MNIST 9/9, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.

### m = 100

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | Fading | 49.79 ± 0.82 | 49.38 ± 1.19 | 48.63 ± 0.87 | **52.23** ± 2.36 | 17.85 ± 1.98 |
|  | Erasure | 60.94 ± 1.30 | 62.08 ± 2.19 | 62.83 ± 2.04 | **62.89** ± 1.34 | 34.41 ± 2.61 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Fading uses 3 paired training seeds and erasure uses 5.
Bold marks the highest unrounded mean in each displayed row.

### m = 2000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 74.09 ± 0.26 | 74.25 ± 0.46 | 74.26 ± 0.46 | **74.26** ± 0.46 | 61.65 ± 1.01 |
|  | Erasure | 86.87 ± 0.20 | 86.87 ± 0.29 | 87.10 ± 0.36 | **87.15** ± 0.33 | 82.68 ± 0.47 |
| MNIST | Fading | 83.31 ± 0.63 | 83.24 ± 0.24 | **83.43** ± 0.19 | 83.42 ± 0.28 | 57.41 ± 0.44 |
|  | Erasure | 96.24 ± 0.04 | 96.24 ± 0.05 | **96.29** ± 0.03 | 96.23 ± 0.07 | 87.59 ± 0.58 |
| Fashion-MNIST | Fading | 73.87 ± 0.68 | 74.13 ± 0.78 | **74.35** ± 0.48 | 74.27 ± 0.50 | 50.36 ± 0.60 |
|  | Erasure | 86.30 ± 0.13 | 86.23 ± 0.12 | 86.27 ± 0.14 | **86.32** ± 0.20 | 77.81 ± 0.58 |
| CIFAR-10 | Fading | 54.23 ± 1.43 | 49.07 ± 13.87 | **55.75** ± 0.91 | 54.21 ± 2.87 | 17.29 ± 0.68 |
|  | Erasure | 65.23 ± 0.95 | 65.10 ± 1.13 | 65.23 ± 0.70 | **65.34** ± 0.53 | 37.07 ± 1.94 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/5, Fashion-MNIST 3/9, CIFAR-10 3/5.
Bold marks the highest unrounded mean in each displayed row.
The MNIST erasure comparison allowed 60 epochs for every method.
