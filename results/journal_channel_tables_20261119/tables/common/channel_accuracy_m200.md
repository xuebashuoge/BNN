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
