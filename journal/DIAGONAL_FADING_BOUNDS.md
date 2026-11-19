# Two-moons bound decomposition

The [combined fading and erasure report](../results/correlated_diagonal_fading/bounds_fading_erasure/README.md) gives
a single simultaneous 95% guarantee over 20 trained predictors. It adds separate T1 and T2 erasure learners under
matched and mismatched training. Perfect-trained erasure uses Corollary 1(i). Its ten erasure upper bounds are
below one. The following table and confidence allocation describe the earlier fading-only family.


Five independently trained posteriors per width compare the perfect-channel corollary, the separated
bound and the joint bound. All five are deployed at $\rho_{\rm test}=0.35$. The matched
training bank uses $\rho_{\rm tr}=0.35$ and the mismatched bank uses $\rho_{\rm tr}=0.15$.
Both stochastic laws have diagonal gains with marginal distribution $\mathcal N(0.5,1)$
and independent standard Gaussian receiver noise. Perfect training uses $(I,0)$.

| Training law | Objective and certificate | Width 16 | Width 8 |
|---|---|---:|---:|
| Perfect | Sample-only objective, C1(ii) | 1.5396 | 0.9156 |
| Matched | T1 learner, Theorem 1 | 0.4784 | 0.4028 |
| Matched | T2 learner, Theorem 2 | 0.4014 | 0.3586 |
| Correlated mismatch | T1 learner, Theorem 1 | 0.8955 | 0.6881 |
| Correlated mismatch | T2 learner, Theorem 2 | 0.8202 | 0.6433 |

The entries are uncapped population bounds for the smooth bounded deployment loss. The two-layer
$2\to d\to 2$ network projects each feature vector onto the Euclidean ball of radius $B=1$
and each receiver matrix onto the Frobenius ball of radius $L=0.075$. The prior has standard
deviation 0.24 and the posterior standard deviation is fixed at 0.1. All five fits at each
width use the same 800 i.i.d. training examples, 200 validation examples, initialization seed,
120-epoch budget and $m=2000$. T1 and T2 share the training bank and random seeds within each
stochastic regime. Separate optimizations learn distinct posterior means. T1 trains with
0.02 times its **sum** of data and channel complexities. T2 trains with 0.02 times its **joint**
complexity. The perfect-channel learner uses 0.02 times the data complexity. The certificates
use the full unweighted terms.

[Decomposition figures and numerical results](../results/correlated_diagonal_fading/bounds_rho015_test035/README.md)

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

## Stochastic training

Write $p=\rho_{\rm test}$ and $q=\rho_{\rm tr}$. The noise law is identical at training and
deployment. The stationary centered AR(1) gain process has conditional variance $1-\rho^2$.
Summing the conditional Gaussian divergences gives the deployment-to-training KL

$$
D_{\rm KL}(P_p\|P_q)=(d-1)
\left[\tfrac12\log\frac{1-q^2}{1-p^2}+\frac{q(q-p)}{1-q^2}\right].
$$

The bounded-loss shift is $S=\sqrt{D_{\rm KL}(P_p\|P_q)/2}$. It vanishes under matched
training. At $(p,q)=(0.35,0.15)$ it equals 0.417763 for width 16 and 0.285387 for width 8.
Both stochastic learners use the same deployment shift because it depends only on the two channel laws.

Let $D_W$ denote the KL of the particular learned weight posterior from its prior.
With bounded loss in $[0,1]$, both sub-Gaussian proxies are $1/4$. Write $\delta_T$ for the
theorem event's failure probability. T1 uses the two separated terms

$$
B_{\rm data}=\sqrt{\frac{D_W+\log(2\sqrt n/\delta_T)}{2(n-1)}},\qquad
B_{\rm channel}=\sqrt{\frac{D_W+\log(2\sqrt m/\delta_T)}{2(m-1)}}.
$$

For the T2-trained posterior, Theorem 2 instead uses the single term

$$
B_{\rm joint}=\sqrt{\left(\frac{1}{2(n-1)}+\frac{1}{2(m-1)}\right)
             \left(D_W+\log\frac{\sqrt{nm}}{\delta_T}\right)}.
$$

Each bound adds its own empirical Gibbs loss upper estimate and $S$. Perfect training uses only
the data term with $\log(\sqrt n/\delta_T)$, together with the analytic C1(ii) shift above.

## Confidence and reproduction

The joint failure budget is 0.05 over the five training/guarantee combinations and both widths.
Each of the ten fits gets $\delta_T=0.0025$ for its one assigned theorem/corollary and
0.0025 for its empirical Monte Carlo event. The T1 stochastic certificate splits its theorem
budget equally between the data and channel terms. Its empirical correction, also used for
the other fits, is $\sqrt{\log(1/0.0025)/(2\times20000)}=0.0122387$.
The 20,000 independent weight/example-index/bank-index observations estimate the empirical
Cartesian-product loss of the actual checkpoint. Upper bounds below one are non-vacuous.
Held-out loss markers use 10,000 examples and 20 paired Gibbs draws and are descriptive.

Run the ten fits on Spartan from a clean checkout with the prepared dataset root in `DATASET`.
The output directory holds per-case checkpoints, training histories, channel-bank fingerprints,
uncapped bound totals and the figure export.

```bash
bash journal/submit_correlated_bounds.sh /absolute/shared/results_directory 0.15
```

The [archived combined-channel decomposition](../results/correlated_diagonal_fading/final/bounds/decomposition.csv)
records the previous T1-only design, including i.i.d. mismatch and erasure. Those historical rows
have their original confidence budget.
