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
