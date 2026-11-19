### m = 1000

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| Two-moon | Fading | 65.60 ± 0.51 | 65.82 ± 0.81 | **65.84** ± 0.81 | 65.84 ± 0.81 | 57.34 ± 1.42 |
|  | Erasure | 86.98 ± 0.21 | 86.87 ± 0.24 | 87.00 ± 0.35 | **87.04** ± 0.32 | 82.68 ± 0.47 |
| MNIST | Fading | 77.76 ± 0.32 | 76.28 ± 0.17 | **77.82** ± 0.37 | 77.71 ± 0.11 | 45.38 ± 0.38 |
|  | Erasure | 96.21 ± 0.04 | 96.21 ± 0.04 | **96.24** ± 0.05 | 96.02 ± 0.09 | 87.99 ± 0.60 |
| Fashion-MNIST | Fading | 67.28 ± 0.36 | 67.63 ± 0.21 | **67.80** ± 0.24 | 67.52 ± 0.17 | 40.82 ± 0.40 |
|  | Erasure | 85.80 ± 0.23 | 85.72 ± 0.11 | **85.88** ± 0.16 | 85.81 ± 0.25 | 77.74 ± 0.82 |
| CIFAR-10 | Fading | 51.89 ± 2.66 | 53.33 ± 0.63 | 51.64 ± 2.87 | **53.79** ± 1.62 | 18.72 ± 0.61 |
|  | Erasure | 64.56 ± 0.56 | 64.20 ± 0.78 | **64.91** ± 1.25 | 64.04 ± 1.20 | 38.92 ± 3.66 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Paired training-seed counts (fading/erasure) are Two-moon 3/5, MNIST 3/8, Fashion-MNIST 9/5, CIFAR-10 5/3.
Bold marks the highest unrounded mean in each displayed row.
Erasure seed triples selected on validation for CIFAR-10.
The CIFAR-10 seed triple in this table was selected separately for this bank size.
The MNIST erasure comparison allowed 60 epochs for every method.
