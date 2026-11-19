### m = 200

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.25 ± 0.60 | 65.29 ± 0.70 | **65.46** ± 0.73 | 65.30 ± 0.67 | 56.96 ± 1.18 |
|  | Erasure | 86.95 ± 0.44 | 86.92 ± 0.23 | 86.92 ± 0.61 | **86.95** ± 0.65 | 82.68 ± 0.47 |
| MNIST | Fading | 73.64 ± 0.71 | 72.52 ± 0.77 | 73.67 ± 0.36 | **73.78** ± 0.60 | 45.38 ± 0.38 |
|  | Erasure | 95.99 ± 0.09 | 96.04 ± 0.08 | 96.01 ± 0.10 | **96.06** ± 0.08 | 87.99 ± 0.60 |
| Fashion-MNIST | Fading | 61.30 ± 0.68 | 61.14 ± 0.93 | 61.47 ± 0.80 | **61.54** ± 0.84 | 40.82 ± 0.40 |
|  | Erasure | 85.58 ± 0.36 | 85.47 ± 0.24 | **85.65** ± 0.13 | 85.50 ± 0.21 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | 34.24 ± 22.28 | 41.89 ± 10.42 | **44.53** ± 4.47 | 34.31 ± 22.30 | 18.72 ± 0.61 |
|  | Erasure | 64.37 ± 0.83 | 64.36 ± 0.62 | **64.86** ± 0.56 | 63.60 ± 1.25 | 37.07 ± 1.94 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 9/5, MNIST 3/8, Fashion-MNIST 9/5, CIFAR-10 5/5.
Bold marks the highest unrounded mean in each displayed row.
Illustrative rounds were chosen using observed rankings.
The MNIST erasure comparison allowed 60 epochs for every method.
