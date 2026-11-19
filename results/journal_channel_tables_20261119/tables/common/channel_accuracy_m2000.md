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
