# Two-moons bound decomposition

The correlated diagonal fading experiment gives non-vacuous bounds on smooth bounded deployment loss for perfect, matched and i.i.d. mismatched training at $d=8$ and $m=2000$. At $d=16$, the perfect and i.i.d. mismatch certificates exceed the maximum possible loss of one. Both widths use the same 800 i.i.d. training examples, 200 validation examples, Gaussian posterior training procedure and channel law.

| Width | Deployment law | Perfect training | Matched training | Mismatched training |
|---:|---|---:|---:|---:|
| 16 | Fading | 1.5400 | 0.4790 | 1.1805 |
| 16 | Erasure | 0.4280 | 0.4778 | 0.6816 |
| 8 | Fading | 0.9161 | 0.4036 | 0.8819 |
| 8 | Erasure | 0.3689 | 0.4164 | 0.5618 |

Entries are uncapped population-risk upper bounds from the Spartan CPU experiment. For fading at width eight, held-out Gibbs accuracies were 56.14%, 53.62% and 53.48%, respectively. The certificate-specific network projects each transmitted feature vector to the Euclidean ball of radius $B=1$ and each receiver matrix to the Frobenius ball of radius $L=0.075$. These fixed projections hold for every raw Gaussian weight draw in the two-layer $2\to d\to2$ classifier.

## Perfect training under correlated fading

The reference $Q_C=P_{\rm iid}$ has the same $\mathcal N(0.5,1)$ gain marginals and independent unit-variance AWGN as deployment. Deployment sets $\rho_c=0.35$. Write $D(C,c_0)=\sqrt{\|h-\mathbf1\|_2^2+\|N\|_2^2}$. Diagonality gives $\|H-I\|_F=\|h-\mathbf1\|_2$.

For the smooth bounded loss $\ell=-\log(\varepsilon+(1-\varepsilon)p_y)/\log(1/\varepsilon)$ with $\varepsilon=0.01$, the logit gradient has norm at most $\sqrt{2}/\log(1/\varepsilon)$. At fixed $f$, the linear map from channel perturbations to received-feature perturbations is $[\operatorname{diag}(f),I]$. Its operator norm is $\sqrt{1+\|f\|_\infty^2}\le\sqrt{1+B^2}$. Composing with the projected receiver gives a channel-Lipschitz constant $K=\sqrt{2}L\sqrt{1+B^2}/\log(1/\varepsilon)$ under the stated joint metric.

The first inequality in Corollary 1 uses $K\mathbb E_{P_{\rm ch}}D(C,c_0)$, the mean of a noncentral Gaussian norm with unequal principal variances. The second inequality gives the computable upper bound

$$
\log\mathbb E_{P_{\rm iid}}e^{K D(C,c_0)}
+D_{\rm KL}(P_{\rm ch}\,\|\,P_{\rm iid}).
$$

The AWGN law cancels from the divergence. The gain laws have equal means, $\operatorname{tr}R=d$ and $\det R=(1-\rho_c^2)^{d-1}$, so the divergence is $-(d-1)\log(1-\rho_c^2)/2$. Under the reference, $D^2$ has a noncentral chi-square law with $2d$ degrees of freedom and noncentrality $d/4$. Applying $KD\le K^2/(2a)+aD^2/2$ and integrating the quadratic exponential yields the elementary closed-form guarantee

$$
\log\mathbb E_{P_{\rm iid}}e^{KD}
\le \frac{K^2}{2a}-d\log(1-a)+\frac{ad}{8(1-a)},
\qquad a=\min\left(\tfrac12,\frac{K}{1.5\sqrt d}\right).
$$

At $d=8$ the log-moment upper bound is 0.13849 and the divergence is 0.45737. At $d=16$ these become 0.19573 and 0.98009. The divergence accounts for most of the change with width. The guarantee uses a closed-form upper bound on the log moment. The numerical quadrature in the regression test checks this calculation independently.

## Erasure and mismatched training

Erasure deployment drops coordinates independently with probability $p=0.50$. For perfect training, Corollary 1's first inequality gives a shift of

$$
\frac{\sqrt2 LB}{\log(1/\varepsilon)}
\sum_{j=1}^{d}\sqrt j \binom{d}{j}p^j(1-p)^{d-j}.
$$

The sum is the exact expected Frobenius distance between the erasure mask and the identity. The matched training law equals deployment in each family. Fading mismatch trains with $\rho_c=0$, preserving every coordinate marginal and noise distribution. Erasure mismatch trains with $p_{\rm tr}=0.45$. Both stochastic training regimes use the separated data and channel complexity terms and the bounded-loss shift $\sqrt{D_{\rm KL}(P_{\rm dep}\|P_{\rm tr})/2}$.

## Confidence and outputs

Each stacked bar adds empirical Gibbs loss, a one-sided empirical Monte Carlo correction, sample complexity, channel complexity and channel shift. Black diamonds mark held-out Gibbs loss over 10,000 independently generated test examples and 20 draws. Perfect training has zero channel complexity. Matched training has zero shift. The $[0,1]$ loss gives the non-vacuity threshold of one.

The confidence budget covers both declared widths, two channel families and three training regimes. Each of the twelve cases receives failure probability $0.05/24$ for the theorem and $0.05/24$ for its independent empirical Monte Carlo estimate. Each empirical estimate uses 20,000 independent triples of weight draw, training-example index and training-bank index. Its one-sided Hoeffding correction is 0.01242. The union bound gives simultaneous coverage of at least 95%, including the decision to run width eight if any width-sixteen bound exceeds or equals one.

Run the six width-sixteen cells, then apply the conditional fallback from the repository root in the torch28 environment.

~~~bash
out=results/correlated_diagonal_fading/bounds
for family in fading erasure; do
  for regime in perfect matched mismatch; do
    python -m journal.diagonal_fading_bounds run --width 16 --family "$family" \
      --regime "$regime" --output "$out"
  done
done
python -m journal.diagonal_fading_bounds fallback --output "$out"
~~~

Each width directory contains the stacked figures in PNG and PDF and the complete numerical decomposition in JSON. Each case retains its posterior, bank, training history and selected epoch.
