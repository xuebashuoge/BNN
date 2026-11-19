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
