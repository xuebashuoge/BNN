# Results inventory and presentation plan

The main numerical story should connect deployment performance against ERM/L2, generalization from finite channel observations, and the three bound contributions. Use all-seed means for the primary tables. Keep selected favorable seeds and data-only PAC comparisons supplementary.

There are 473 unique search trials, not 473 independent replications of one setting. Main figures/tables use ten inference draws and two or three initialization/channel-bank seeds conditional on a fixed dataset. The five-draw screen and 100-draw selected-checkpoint checks stay separate.

## Recommended main text

| Item | Presentation | Supported claim |
| --- | --- | --- |
| Experimental settings | Table I | Exact data/validation/test sizes, width, bank size, epochs and repetitions for the two image and two toy protocols. |
| Image deployment performance | Table II | Six available method/training-law rows, with accuracy and normalized bounded loss for full MNIST and FashionMNIST. Mean plus sample SD over every listed seed. |
| Finite-bank mechanism | Figure 1, accuracy and channel-gap panels | Fix data, width, deployment law and 600 optimizer updates; vary `m=5,20,100,500,2000`. Include perfect-trained references. |
| Training-law sensitivity | Figure 2, accuracy and CE panels | Fix deployment erasure probability 0.5 and vary training probability, including perfect at zero. Matching is best here, not a universal theorem about accuracy. |
| Bound decomposition | Figure 3, theoretical and signed empirical panels | Compare perfect, matched and milder mismatched training at the same `n=800,d=16,m=100`. Stack unweighted T1 terms and mark T2 at each same posterior. These illustrative worst-case gap bounds are vacuous. |
| Corrected numerical bound | Table III | At one width-8 posterior, `n=512,m=2000`, include empirical risk, MC correction, separate/joint terms and full risk bounds. Both theorems evaluate the same T1-trained posterior. |

This is three tables and three empirical figures in the main simulation section. Accuracy bars would duplicate Table II; the table reports uncertainty and theoretical loss more compactly. The corrected analytical T1/T2 comparison belongs near the theory example, outside Simulation.

## Main numbers

| Mean accuracy | MNIST, 55,000 training examples | FashionMNIST, 10,000 training examples |
| --- | ---: | ---: |
| ERM Perfect | 58.07 +/- 0.42% | 50.95 +/- 1.30% |
| ERM Matched | 74.60 +/- 1.12% | 61.57 +/- 0.09% |
| L2 Matched | 70.12 +/- 1.21% | 61.49 +/- 0.42% |
| Proposed Perfect | 57.94 +/- 0.60% | 52.44 +/- 0.29% |
| Proposed T1 Matched | **76.63 +/- 0.44%** | **63.24 +/- 0.21%** |
| Proposed T2 Matched | 76.43 +/- 0.76% | 63.21 +/- 0.17% |

Both proposed variants beat the evaluated ERM and L2 settings in every seed of these cohorts. T1 has the highest mean accuracy; T2 has slightly lower mean bounded loss. MNIST bounded-risk means are ERM 0.194443, L2 0.195033, T1 0.179480 and T2 0.179234. FashionMNIST means are 0.228679, 0.229478, 0.218845 and 0.218730. Do not claim minimum raw CE: MNIST L2 raw CE is about 1.1604, below T1 1.7027 despite lower accuracy.

For fixed matched erasure probability 0.5, T1's observed channel gap drops from 0.010314 at `m=5` to 0.001605 at `m=20`, then fluctuates near zero. Accuracy is 77.44%, 78.05%, 79.52%, 79.26% and 79.26%. This supports the finite-channel mechanism, not strict monotonic accuracy or a ranking of nearly overlapping PAC-family optimizers.

At deployment erasure probability 0.5, T1 accuracy is 83.126%, 84.418%, 84.903%, 84.839% and 84.266% for training probabilities 0.1, 0.3, 0.5, 0.7 and 0.9. Proposed Perfect gives 81.870%. Matched training also minimizes CE in this experiment.

The corrected risk calculation has empirical bounded loss 0.0909427, MC radius 0.1517427, T1 channel/data terms 0.2129374/0.4203686 and T2 joint term 0.4750782. Total risk bounds are 0.8759914 and 0.7177636. They include simultaneous confidence allocation over both theorems and 21 matched-erasure setups. They concern normalized Gibbs loss, not classification error or unbounded CE.

## Remaining usable evidence

| Result family | Placement | Interpretation |
| --- | --- | --- |
| Reduced MNIST, `n=10000,m=20,50`, 35 epochs | Appendix Table IV | Four matched methods, two seeds, ten draws. T2 leads at `m=50`; the same official test set is reused, so it is not independent confirmation. |
| Initial MNIST, `n=10000,m=100`, 25 epochs | Appendix Table V | Five-draw screen, reported separately rather than pooled with the main ten-draw tables. |
| Width-64 toy erasures, `m=5,20,50` | Appendix Table VI | Additional width/bank sensitivity. ERM wins at some bank sizes; retain those outcomes. L2 is unavailable in this cohort. |
| Data-size slice, `n=128,512,2048` | Appendix Table VII | Accuracy rises but updates also increase from 150 to 600 to 2400 and data realizations change. Signed data gaps are negative; no quantitative empirical `n`-rate is established. |
| Selected hyperparameters | Appendix Table VIII | Exact LR, posterior-noise initialization and active penalties for all main rows. The conditional search is not exhaustive baseline optimization. |
| Data-only PAC/stochastic ERM/raw-CE ERM | Appendix discussion and source CSV/JSON | Internal objective/predictor controls. T1/T2 largely rescale the same KL regularizer. Raw-CE diagnostics have a different bounded-loss definition. |
| Selected favorable 100-draw checks | Explicit exploratory appendix examples | Full MNIST seed 9001: T1 75.9712% versus PAC 75.7801%. Reduced MNIST `m=50`, seed 1003: T2 67.7729% versus PAC 67.4309%. Conditional MC intervals exclude selection and training-seed uncertainty. |
| Matched-strength FashionMNIST check | Appendix interpretation | T1 63.0982% versus PAC 63.0774% over two fixed checkpoints. The earlier larger gap shrank after missing LR/noise controls; treat the final result as a near tie. |
| Analytical separate/joint complexity curves | Theory section | Corrected `fig_bound_structure`, KL 2, variance proxies 1/4, failure budget 0.05, exact product-confidence equal and 99:1 allocations. Formula evaluation, not learning data. |
| Legacy conference tables/sweeps | Historical evidence, unless reevaluated | Earlier Theorem-3 or joint-ensemble objectives and inconsistent risk/trajectory logging cannot directly validate the current single-posterior theorem. New common-protocol tables replace them. |

The legacy four-scenario MNIST summaries still motivate severe impairment: their 100-seed grid favors the conference-style proposal by 1.5093 percentage points under combined fading/noise, while ERM wins in the other three scenarios. Corresponding MNIST checkpoints were not saved. The legacy two-seed mismatch prototype is superseded by the new three-seed bounded-loss experiment. Do not merge these different protocols into the main table.

## Manuscript consistency and submission priorities

- Main comparisons use the same bounded loss, preventing attribution of a loss change solely to regularization. Canonical raw-CE ERM remains supplementary.
- The campaign is exploratory: test outcomes informed later search rounds, although final configurations/checkpoints are validation-selected. Avoid claiming an untouched confirmatory test. A locked protocol with fresh seeds or a new held-out split would strengthen final claims.
- L2 Perfect is unavailable under the new protocol. Its absence is recorded, not filled by a legacy model. Broader decoupled LR/L2 searches are needed before calling that baseline exhaustively optimized.
- Perfect describes the training forward channel; validation still uses deployment channels. These references are not untouched channel-unaware pretrained models.
- Update the algorithm initialization: fan-in-scaled random means, zero biases and posterior std 0.003/0.01, not `q0=Q` with unit variance. The data-independent prior remains zero-mean unit Gaussian.
- Incorporate the smooth bounded-loss definition/Hoeffding justification and the TV shift specialization. A local weight-gradient norm is not a channel-Lipschitz certificate.
- The erasure experiment masks continuous features without quantization, coding, erasure flags or retransmission. Do not call it a complete bit-level BEC link.
- Feature power is unnormalized. Full MNIST means are about T1 23.06, ERM 15.88 and L2 5.29 per coordinate. State channel-law parameters; power-normalized experiments are needed for fixed-SNR or energy-efficiency claims, particularly for TCOM.
- Image models are two-layer MLPs. Toy width variation is not replication across CNN/transformer architectural families.
- Nominal decomposition bars and corrected certificates use different confidence budgets; their components must not be interchanged.
- Replace the old simulation section and its tables rather than appending this draft: the old tables reuse labels and mix risk conventions.

## Source and reproducibility files

The main manuscript fragment is `simulation_section.tex`; optional appendix content is `supplementary_results.tex`. `main.tex` builds the preview. The table generator writes seven table/CSV pairs and `data_manifest.json` with 93 hashed inputs. `../prepare_paper_figures.py` writes five vector PDF/PNG pairs and `assets/figure_data.json` with 41 hashed inputs. Original trial records and all-method results remain under `results/journal_campaign/` and `journal/CAMPAIGN_RESULTS.md`.

Dataset citations were checked against the [authors' MNIST article page](https://bottou.org/papers/lecun-98h) and [Fashion-MNIST paper](https://arxiv.org/abs/1708.07747). Both manual-IEEE and BibTeX entries are provided.
