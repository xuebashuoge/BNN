# Correlated Gaussian fading with AWGN

The channel transmits a feature vector $f$ as $Y=Hf+N$, where

```math
H=0.5\,\mathbf 1_{d_r}\mathbf 1_{d_t}^{\top}+R_r^{1/2} G R_t^{1/2},
\qquad G_{ij}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad N\sim\mathcal N(0,I_{d_r}).
```

The noise is independent of $H$. With unit diagonals in $R_r$ and $R_t$,
$\operatorname{Cov}(H_{ij},H_{k\ell})=(R_r)_{ik}(R_t)_{j\ell}$ and
each gain is $\mathcal N(0.5,1)$. The experiments use AR(1) correlations
$(R_s)_{ij}=\rho_s^{|i-j|}$ with $\rho_r=\rho_t=0.25$ at deployment. The
mismatch changes $\rho_r$ to $0.20$ during training. Erasure uses independent
coordinate outages with probability $0.50$ at deployment and $0.45$ for
mismatched training.

Matched and mismatched training reuse a fixed bank of complete gain matrices
and noise vectors. The deployment evaluator samples the exact conditional law

```math
Y\mid f\sim\mathcal N\left(0.5(\mathbf 1^{\top}f)\mathbf 1,\,
    (f^{\top}R_t f)R_r+I\right).
```

This reduces fresh-channel evaluation cost. Paired methods share standardized
received-space draws and training-bank seeds. Under this law without receiver
channel-state information, the distribution depends on the transmitter
through $\mathbf 1^{\top}f$ and $f^{\top}R_t f$.

## Corollary 1

For the bounded two-moons network, fix a feature norm cap $R$ and receiver
matrix Frobenius norm cap $L$. With smooth bounded log loss and parameter
$\varepsilon_\ell=0.01$, the channel Lipschitz constant is
$K=\sqrt{2}L/\log(1/\varepsilon_\ell)$ for the metric
$R\|H-H'\|_F+\|N-N'\|_2$. The projections apply to every posterior
draw. The Gaussian prior and posterior are defined on the raw weights.

For perfect training with fading, choose the data-independent reference
$Q_C=P_{\mathrm{ch}}$ in the second inequality. The KL term is zero and
independence splits the log moment into gain and AWGN terms. In the correlation
eigenbases, $H-I$ has independent coordinates with means given by the rotated
matrix $0.5\mathbf 1\mathbf 1^\top-I$ and variances given by the pairwise
products of the receive and transmit eigenvalues. For independent
$X_i\sim\mathcal N(\mu_i,v_i)$ and $0<a<1/\max_i v_i$, Young's inequality
and the Gaussian quadratic moment give

```math
\log\mathbb E e^{t\|X\|_2}
\le \frac{t^2}{2a}
  +\frac12\sum_i\left[-\log(1-av_i)+\frac{a\mu_i^2}{1-av_i}\right].
```

Use $t=KR$ for the gain and $t=K$ for AWGN. The implementation chooses $a$
from the distribution parameters and reports the analytic sum. For perfect
training with erasures, the first inequality uses the finite binomial
expectation of the mask distance. The paper models use the complete-channel
discrete metric with $K=1$, which is valid for their bounded loss and yields
a unit perfect-fading shift. The compact experiment uses $R=1$, $L=0.14$,
and width eight to obtain a nontrivial Frobenius-metric guarantee.

The decomposition stacks empirical Gibbs loss, its one-sided Monte Carlo
correction, sample and channel complexities, and channel shift. The primary
cases use the four paper architectures. The compact two-moons cases use the
same $m=2000$ and an isotropic prior with standard deviation $0.2$.
Matched and mismatch use the separate bound in Theorem 1. Their shift is zero
for matched training and uses the analytic channel KL for mismatch.

## Spartan execution

Run from an isolated checkout with `torch28` available. The launcher uses
`DATASET` and checks the submitted Git revision. Manifests pin task
configurations and seeds by SHA-256. Independent array slots train one
configuration and keep individual reports. Dependent jobs select configurations
or plot once the required set completes.

```bash
export JOURNAL_GIT_SHA=$(git rev-parse HEAD)
export FULL_FADING_OUTPUT=/path/to/new/project/results
export FULL_FADING_PYTHON=/path/to/torch28/bin/python
bash journal/submit_full_fading.sh pilot
FULL_FADING_IMAGE_DEVICE=cpu bash journal/submit_full_fading.sh sweep
FULL_FADING_IMAGE_DEVICE=cpu bash journal/submit_full_fading.sh bounds
bash journal/submit_full_fading.sh bounds-compact
FULL_FADING_IMAGE_DEVICE=cpu FULL_FADING_COMPARISON_DATASETS=mnist,fashionmnist \
  bash journal/submit_full_fading.sh comparison
FULL_FADING_IMAGE_DEVICE=cuda FULL_FADING_COMPARISON_DATASETS=cifar10 \
  bash journal/submit_full_fading.sh comparison
```

The MNIST sweep holds the data size, training budget, and method-specific
optimizer settings fixed across the 15 bank sizes in the reference figure.
Its two panels report test Gibbs accuracy and the signed difference between
held-out deployment loss and empirical reused-bank loss. An optional
`sweep-extended` phase adds $m=6400,12800,25600$.

The accuracy study starts at $m=2000$ with ERM, L2, T1, and T2. The
perfect-training T1 and T2 objectives coincide and share one row. Each of the
seven labels has four hyperparameter candidates with two paired validation
seeds. Configuration selection uses mean validation accuracy and breaks ties
with cross-entropy. Each selected configuration is retrained on three fresh
paired seeds for final deployment evaluation. The report keeps all
confirmation rows and records the highest measured mean for each dataset.

The optional `bounds-compact` phase confirms a more predictive width-eight
configuration chosen with validation data. Its fixed projection caps and
Gaussian prior match the original compact design. It uses new two-moons
samples and random seeds with a separate simultaneous 95% bound family.
The pilot and frozen choices are recorded in
`configs/full_fading_compact_validation.json`.

The optional `comparison-refine` phase adds four validation candidates per
method to the original grid at the same bank size. It freezes the complete
combined validation grid and evaluates on a new three-seed training and
channel cohort. The official test examples are reused across comparison
rounds.

An optional CPU bootstrap sets flush-to-zero and denormals-are-zero before
PyTorch initializes its worker threads. For CPU campaigns that specify
`cpu_flush_denormal=True`, compile `journal/cpu_flush_bootstrap.c` as a shared
library on Spartan and export its path as `FULL_FADING_FTZ_LIBRARY` and its
SHA-256 as `FULL_FADING_FTZ_SHA256`. The Slurm script checks the fingerprint
before loading the library.
