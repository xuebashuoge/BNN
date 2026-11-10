# Main journal comparisons with ERM and L2

The main competing baselines are ERM and L2. The proposed learner is itself a PAC-Bayesian method, so the existing data-only PAC penalty is an internal ablation rather than an independent algorithm family. Its saved results remain available in the full campaign report and the all-method figures.

## Objectives and their relationship

Let `q` be the learned Gaussian posterior, `Q` the fixed prior, and
`D=KL(q||Q)`. Write `R_hat(q)` for the same bounded empirical risk on the
reused data-by-channel bank. Define

\[
a=\frac{2\sigma^2}{n-1},\qquad b=\frac{2\sigma_{\rm tr}^2}{m-1},
\quad c_n=\log\frac{\sqrt n}{\epsilon_n},\quad
c_m=\log\frac{\sqrt m}{\epsilon_m},\quad
c_{nm}=\log\frac{\sqrt{nm}}{\delta}.
\]

The implemented data-only ablation and proposed objectives are

\[
\mathcal J_{\rm data\text{-}PAC}(q)
=\widehat R(q)+\gamma\sqrt{a(D+c_n)},
\]
\[
\mathcal J_{\rm T1}(q)
=\widehat R(q)+\eta_{\rm data}\sqrt{a(D+c_n)}
+\eta_{\rm ch}\sqrt{b(D+c_m)},
\]
\[
\mathcal J_{\rm T2}(q)
=\widehat R(q)+\eta\sqrt{a+b}\sqrt{D+c_{nm}}.
\]

T2 has exactly the same single-square-root KL form as a data-PAC objective,
with a changed coefficient and confidence logarithm. T1 combines two shifted
square roots. It is exactly a coefficient change when `c_n=c_m`; when `D`
dominates both logarithms, it is approximately

\[
\mathcal J_{\rm T1}(q)\approx\widehat R(q)
+\left(\eta_{\rm data}\sqrt a+\eta_{\rm ch}\sqrt b\right)\sqrt D.
\]

More precisely, the T1 gradient matches the data-PAC gradient with local
coefficient

\[
\gamma_{\rm eff}(D)=\eta_{\rm data}
+\eta_{\rm ch}\sqrt{b/a}\sqrt{(D+c_n)/(D+c_m)}.
\]

This is a gradient-matching coefficient at the current KL value, rather than
an objective coefficient whose derivative should also be taken. It is nearly
constant in the image experiments. These statements concern the particular
square-root-KL baseline implemented here, rather than all PAC-Bayesian
learning objectives.

The theory remains different: the data-complexity term alone does not bound
the entire gap from a reused finite `S x U` empirical risk to deployment risk.
The new results distinguish data size `n`, channel-bank size `m`, and channel
distribution shift. They justify the channel-dependent scaling and the
corresponding full risk guarantee, even when optimization remains in the same
regularizer family.

For deterministic ERM and L2, the corresponding main objectives are

\[
\mathcal J_{\rm ERM}(w)=\widehat R(w),\qquad
\mathcal J_{\rm L2}(w)=\widehat R(w)+\lambda_2\|w\|_2^2.
\]

The main comparisons use the same bounded loss and channel/data conditions
within each experiment. Raw-CE ERM and stochastic ERM remain optional
diagnostic controls. The prior-to-posterior KL also regularizes posterior
variance, so it is not identical to deterministic squared-weight L2.

## Existing matched-channel results

These are means over every saved seed of the validation-selected
configurations, using the common ten-draw test evaluation. No favorable-seed
selection is used in this table.

| Setting | Seeds | ERM | L2 | Proposed T1 | Proposed T2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST, 55,000 training examples, `m=100` | 3 | 74.5980% | 70.1150% | **76.6307%** | 76.4347% |
| FashionMNIST, 10,000 training examples, `m=100` | 2 | 61.5685% | 61.4905% | **63.2370%** | 63.2085% |
| MNIST, 10,000 training examples, `m=50` | 2 | 66.0355% | 64.3690% | 66.8340% | **66.9570%** |

Both proposed methods exceed both ERM and L2 in every saved seed of these
three comparisons. For full-data MNIST, the mean gains over ERM are 2.0327
percentage points for T1 and 1.8367 points for T2. These are the existing
searched configurations, not a claim that the baselines are globally
optimal; the L2 search was coarse and its settings were transferred to the
full-data experiment.

## Perfect training and presentation

Proposed Perfect uses identity-channel empirical loss plus the data PAC-Bayes
penalty, with zero channel complexity. Therefore the saved `PAC-Perfect`
checkpoint is precisely this special case and is displayed as
`Proposed-Perfect`; it is not duplicated as a separate competing method.
Testing still uses the deployment channel.

The existing campaign has ERM Perfect and Proposed Perfect references, but
does not contain L2 Perfect under this new protocol. Main figures include the
available references and record the missing cell; legacy L2 Perfect models
are not silently substituted. No new training is needed to produce the
current main comparisons.

`plot_campaign.py --comparison main` is now the default for accuracy and
decomposition figures. It shows ERM/L2 and the proposed variants. The
`--comparison all` option retains the complete ablation view. Mechanism and
mismatch figures retain their labelled diagnostic controls because those
curves explain the finite-channel phenomenon rather than rank independent
algorithm families.

Main PNG/PDF pairs are under `results/journal_campaign/figures_main/`.
Original all-method figures remain under `figures_final/`; all training,
test and Monte Carlo results remain unchanged. The full internal findings
and caveats remain in `CAMPAIGN_RESULTS.md`.
