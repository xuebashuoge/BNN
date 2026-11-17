# C1: analytic tractability through a reference channel

The all-Bayesian accuracy campaign and its small-model T1/T2 certificates used bounded-loss TV/Pinsker shift terms. Those calculations do not demonstrate C1's second inequality. The separate centered-lognormal example discussed in chat `01a1d20d-146c-7822-a6df-fe7b3d068cbb` was preserved on `TIT_gpt` at `4ba8578`; it was not part of the new all-Bayesian campaign.

This diagnostic evaluates C1 on twelve current frozen Gaussian posteriors: ERM/L2/T1/T2, three paired seeds, using the perfect-trained outage-validation cohort of the width-8, n=512 experiment. No training, model selection or favorable-seed selection is performed. All twelve posteriors and all severities are retained.

C1 bounds the perfect-training gap by a channel-distance term plus a data term:

\[
R_{\rm dep}(q)\leq\widehat R_{S,c_0}(q)+B_{\rm data}(q)
+K\mathbb E_{P_{\rm ch}}d(C,c_0)
\leq\widehat R_{S,c_0}(q)+B_{\rm data}(q)
+\log\mathbb E_{Q_C}e^{K d(C,c_0)}+D(P_{\rm ch}\Vert Q_C).
\]

The second expression is a relaxation of the first. Its role here is analytical computability, not a different training objective or a claim that the first expectation is infinite.

The direct case uses hidden-feature erasure with independent erasure probability p and the discrete metric on complete masks. Since the normalized loss is in [0,1], K=1 uniformly over all weights and inputs. At hidden width d=8, the first channel term is exactly `1-(1-p)^8`.

The reference case moves the deployment link to the two output logits: the full trained NN is the transmitter and a fixed softmax is the receiver. With additive noise C=N, c0=0 and Euclidean distance, the common smooth loss is

\[
\ell_\epsilon=-\frac{\log[\epsilon+(1-\epsilon)p_y]}{\log(1/\epsilon)},\qquad \epsilon=.01.
\]

Its logit gradient is `a(p_y)*(p-e_y)/log(1/epsilon)`, with `0<=a<=1`, so the uniform channel-Lipschitz constant is

\[
K=\frac{\sqrt2}{\log100}.
\]

This constant is independent of NN weights, including unbounded Gaussian posteriors. It is not a Lipschitz certificate for the original hidden-feature Gaussian link. Perfect-channel training and the saved noiseless classifier remain unchanged.

The fully specified deployment law has independent centered-lognormal coordinates

\[
N_i=\frac{\sigma}{\sqrt{e^{\tau^2}-1}}\left(e^{\tau Z_i-\tau^2/2}-1\right),
\quad Z_i\sim\mathcal N(0,1),\quad i=1,2,\quad\tau=.1.
\]

Each coordinate has mean zero and variance sigma squared. The first expectation `K*E sqrt(N1^2+N2^2)` is a two-dimensional integral without a simple known closed form. This wording does not claim mathematical impossibility of an alternative special-function representation.

Choose the data-independent, moment-matched reference `Q_C=N(0,sigma^2 I2)`. For sigma>0, P is absolutely continuous with respect to Q, and entropy plus the exact second moment give

\[
J_\tau=D(P_{\rm ch}\Vert Q_C)
=\tau^2+\log\frac{e^{\tau^2}-1}{\tau^2}.
\]

Under Q, `||C||/sigma` is Rayleigh. With `t=K*sigma` and standard normal CDF Phi,

\[
\boxed{B_{\rm ref}(\sigma)=
\log\left[1+t\sqrt{2\pi}e^{t^2/2}\Phi(t)\right]+J_\tau.}
\]

The entire channel term, including the reference choice, is analytical: no deployment-distance integration, distance MC estimate or numerical reference optimization enters it. At sigma=.5 the log moment is 0.1976093712, the exact KL is 0.0150041667, and C1's second channel term is **0.2126135379**. At sigma=0 both laws are treated separately as the same Dirac law, with zero shift.

Independent distance MC is plotted as a diagnostic with its descriptive standard error. It supplies no certified first-inequality risk bound. Jensen also gives the separate analytical upper bound `K*sigma*sqrt(2)`; the example demonstrates C1's tractability, not uniqueness among all possible analytical bounds. At sigma=.5 C1's second term is smaller than this Jensen upper bound, 0.2171472410.

The channel term is closed form. The full reported population-risk guarantee additionally uses the actual checkpoint KL, a fresh empirical Gibbs-risk estimate, a bounded-loss MC correction and the data-complexity term. Its new family budget is delta=.05: .04 across twelve data events and .01 across twelve empirical MC events. Analytic channel terms apply simultaneously to every displayed severity; held-out test metrics are descriptive. This budget is separate from the earlier main/T1/T2 certificate families and does not assert an overall .05 union budget across all studies.

Run under the required environment, without training:

```bash
conda run -n torch28 python -m unittest journal.test_c1_reference journal.test_evaluate_c1_bayesian journal.test_plot_c1_bayesian -v
conda run -n torch28 python -m journal.evaluate_c1_bayesian \
  --checkpoint-root results/journal_bayesian_certificate/20261117/campaign \
  --output results/journal_c1_bayesian/20261117
conda run -n torch28 python -m journal.plot_c1_bayesian \
  results/journal_c1_bayesian/20261117/c1_evaluation.json \
  --output results/journal_c1_bayesian/20261117/figures
```

The evaluator records frozen source fingerprints and uses fresh empirical and test streams. The declared grids are erasure probabilities `0,.01,.05,.1,.25,.5` and lognormal standard deviations `0,.1,.25,.5,1,2`. Reports include all twelve individual posteriors, normalized empirical/test risks, accuracy, analytic components, uncapped/capped risk bounds and vacuity flags. Figures are exported as PNG, PDF and EPS without confidence shading.

Distribution definitions and reference moments can also be checked against the official [SciPy lognormal documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html) and [SciPy Rayleigh documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html). Tests integrate the density-ratio and Rayleigh expressions independently and check the loss-gradient bound.

Completed evaluation on 17 November 2026 using the twelve current frozen posteriors: 1000 fresh empirical draws, 100 deployment draws and a separate two-million-sample distance diagnostic. No optimizer was created and no checkpoint was modified. All22 focused tests passed, and all three actual figures were rendered as PNG/PDF/EPS without confidence shading.

At sigma=.5, the first-distance diagnostic is 0.1918912956 (descriptive MC standard error about0.0000718). The analytic second term is 0.2126135379 =0.1976093712+0.0150041667. It is independent of the distance estimate. The current corrected risk results below average all three fixed seeds for each algorithm; all twelve individual risk bounds at this severity are below one.

| Frozen perfect-trained method | Held-out bounded-risk estimate | C1 second-inequality risk RHS |
| --- | ---: | ---: |
| ERM | 0.077857 | 0.646119 |
| L2 | 0.077916 | 0.646121 |
| T1 | 0.077701 | 0.645531 |
| T2 | 0.077701 | 0.645531 |

The mean RHS across all twelve posteriors is 0.6458254620. These new-posterior values supersede the historical0.8801 number for this diagnostic; they do not overwrite that older experiment. Complete per-seed/severity data are in `results/journal_c1_bayesian/20261117/c1_evaluation.csv` and `.json`. `figures/c1_channel_terms` illustrates the tractability tradeoff; `c1_erasure_population_risk` and `c1_lognormal_population_risk` compare held-out risks with both C1 applications. Each figure has PNG, PDF and EPS copies.
