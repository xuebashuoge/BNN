### m = 100

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | Fading | 43.93 ± 1.84 | 45.55 ± 0.78 | 42.26 ± 1.48 | **46.88** ± 3.18 | 16.34 ± 1.30 |
|  | Erasure | 60.94 ± 1.30 | 62.08 ± 2.19 | 62.83 ± 2.04 | **62.89** ± 1.34 | 34.41 ± 2.61 |

Fading $\rho_c=0.85$, $\sigma_N=2$. Erasure $p=0.5$.
Fading uses 3 paired training seeds and erasure uses 5.
Bold marks the highest unrounded mean in each displayed row.
