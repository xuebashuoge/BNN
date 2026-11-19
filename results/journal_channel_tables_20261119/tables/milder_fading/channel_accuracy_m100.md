### m = 100

| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | Fading | 49.79 ± 0.82 | 49.38 ± 1.19 | 48.63 ± 0.87 | **52.23** ± 2.36 | 17.85 ± 1.98 |
|  | Erasure | 60.94 ± 1.30 | 62.08 ± 2.19 | 62.83 ± 2.04 | **62.89** ± 1.34 | 34.41 ± 2.61 |

Fading $\rho_c=0.35$, $\sigma_N=1$. Erasure $p=0.5$.
Fading uses 3 paired training seeds and erasure uses 5.
Bold marks the highest unrounded mean in each displayed row.
