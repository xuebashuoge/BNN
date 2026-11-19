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
