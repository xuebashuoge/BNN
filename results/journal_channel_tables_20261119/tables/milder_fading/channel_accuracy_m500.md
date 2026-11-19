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
