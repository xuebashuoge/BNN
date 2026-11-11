# Results inventory and presentation plan

The main numerical story should connect deployment performance against ERM/L2, generalization from finite channel observations, and the three bound contributions. Use all-seed means for the primary tables. Keep selected favorable seeds and data-only PAC comparisons supplementary.

The original MNIST/FashionMNIST/toy campaigns contain 473 unique search trials, not 473 independent replications of one setting. The CIFAR extension adds a separate dataset/CNN family. Main figures/tables use ten inference draws and two or three initialization/channel-bank seeds conditional on a fixed dataset. The five-draw screen and 100-draw selected-checkpoint checks stay separate.

## Recommended main text

| Item | Presentation | Supported claim |
| --- | --- | --- |
| Experimental settings | Table I | Exact data/validation/test sizes, width, bank size, epochs and repetitions for the three image and two toy protocols. |
| Image deployment performance | Table II | Six available method/training-law rows, with accuracy and normalized bounded loss for full MNIST and FashionMNIST. Mean plus sample SD over every listed seed. |
| RGB/CNN extension | Separate CIFAR table | All seven matched/perfect rows, including L2 Perfect, with accuracy and normalized bounded loss. Keep the 10k screen separate from the 45k full-data cohort. |
| Finite-bank mechanism | Figure 1, accuracy and channel-gap panels | Fix data, width, deployment law and 600 optimizer updates; vary `m=5,20,100,500,2000`. Include perfect-trained references. |
| Training-law sensitivity | Figure 2, accuracy and CE panels | Fix deployment erasure probability 0.5 and vary training probability, including perfect at zero. Matching is best here, not a universal theorem about accuracy. |
| Bound decomposition | Figure 3, theoretical and signed empirical panels | Compare perfect, matched and milder mismatched training at the same `n=800,d=16,m=100`. Stack unweighted T1 terms and mark T2 at each same posterior. These illustrative worst-case gap bounds are vacuous. |
| Corrected numerical bound | Certificate table | At one width-8 posterior, `n=512,m=2000`, include empirical risk, MC correction, separate/joint terms and full risk bounds. Both theorems evaluate the same T1-trained posterior. |

This is four tables and three empirical figures in the main simulation section. Accuracy bars would duplicate the performance tables; tables report uncertainty and theoretical loss more compactly. CIFAR validation trajectories are an optional supplementary diagnostic. The corrected analytical T1/T2 comparison belongs near the theory example, outside Simulation.

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

The full CIFAR-10 comparison uses 45,000 training/5,000 validation examples, the compact 156,074-weight CNN, `m=100`, 100 epochs, and all three new seeds 5101--5103. Its separate main table reports:

| Method | Accuracy, mean +/- sample SD (%) | Mean bounded risk |
| --- | ---: | ---: |
| ERM Matched | 46.1617 +/- 1.3318 | 0.420510 |
| L2 Matched | 53.6700 +/- 0.7274 | 0.352637 |
| **T1 Matched** | **55.0300 +/- 0.2945** | **0.339546** |
| T2 Matched | 53.2637 +/- 0.6037 | 0.354092 |
| ERM Perfect | 34.4410 +/- 0.9738 | 0.443803 |
| L2 Perfect | 34.4290 +/- 0.9162 | 0.439990 |
| Proposed Perfect | 34.0197 +/- 0.8873 | 0.450062 |

T1 exceeds L2 in all three seeds by 0.976/1.996/1.108 percentage points, averaging 1.3600 points; its gain over ERM is 8.8683 points. T2 is slightly below L2 in mean accuracy and bounded-risk performance. Proposed Perfect is also below the two deterministic perfect-trained references in mean accuracy. Preserve these outcomes rather than claiming that every proposed variant wins.

The 34-run CIFAR screen uses 10k training examples, 30 epochs and two other seeds. Its mean accuracies are ERM 39.3800%, L2 40.0315%, T1 44.2885%, T2 43.8585%; ERM has the lowest bounded risk in that stage. Keep the screen in its own supplementary table. Both stages share the validation prefix and seeded permutation, and the training subsets are nested. Selected L2 full-data checkpoints occur at epochs 99/99/100 and T1 at 99/92/93, so the full-data results are a fixed-budget comparison rather than evidence of converged/global optimum accuracy.

CIFAR adds 55 completed comparison trials (34 screen plus 21 full) and 21 resource probes (18 initial plus 3 sustained numerical-policy probes). Cancelled partial attempts are retained in the job ledger and excluded from scientific summaries. All actual training used Spartan CPUs. The CPU subnormal-arithmetic fix is applied uniformly within the full-data cohort; its default remains off for older runs. The completed replacement reference is byte-identical to its staged publication checkpoint. `cifar_data_manifest.json` hashes all 74 CIFAR source reports/histories; the main and screen statistics are independently verified against all 350 individual MC replicates across the two stages.

For fixed matched erasure probability 0.5, T1's observed channel gap drops from 0.010314 at `m=5` to 0.001605 at `m=20`, then fluctuates near zero. Accuracy is 77.44%, 78.05%, 79.52%, 79.26% and 79.26%. This supports the finite-channel mechanism, not strict monotonic accuracy or a ranking of nearly overlapping PAC-family optimizers.

At deployment erasure probability 0.5, T1 accuracy is 83.126%, 84.418%, 84.903%, 84.839% and 84.266% for training probabilities 0.1, 0.3, 0.5, 0.7 and 0.9. Proposed Perfect gives 81.870%. Matched training also minimizes CE in this experiment.

The corrected risk calculation has empirical bounded loss 0.0909427, MC radius 0.1517427, T1 channel/data terms 0.2129374/0.4203686 and T2 joint term 0.4750782. Total risk bounds are 0.8759914 and 0.7177636. They include simultaneous confidence allocation over both theorems and 21 matched-erasure setups. They concern normalized Gibbs loss, not classification error or unbounded CE.

## Remaining usable evidence

| Result family | Placement | Interpretation |
| --- | --- | --- |
| Reduced MNIST, `n=10000,m=20,50`, 35 epochs | Appendix Table V | Four matched methods, two seeds, ten draws. T2 leads at `m=50`; the same official test set is reused, so it is not independent confirmation. |
| Initial MNIST, `n=10000,m=100`, 25 epochs | Appendix Table VI | Five-draw screen, reported separately rather than pooled with the main ten-draw tables. |
| Width-64 toy erasures, `m=5,20,50` | Appendix Table VII | Additional width/bank sensitivity. ERM wins at some bank sizes; retain those outcomes. L2 is unavailable in this cohort. |
| Data-size slice, `n=128,512,2048` | Appendix Table VIII | Accuracy rises but updates also increase from 150 to 600 to 2400 and data realizations change. Signed data gaps are negative; no quantitative empirical `n`-rate is established. |
| Selected grayscale hyperparameters | Appendix Table IX | Exact LR, posterior-noise initialization and active penalties for the grayscale rows. The conditional search is not exhaustive baseline optimization. |
| CIFAR 10k screen and selected settings | Appendix Table X and prose | All seven methods, two seeds and ten draws; keep separate from the 45k/100-epoch cohort. The optional full-data validation plot checks budget/selection behaviour. |
| Data-only PAC/stochastic ERM/raw-CE ERM | Appendix discussion and source CSV/JSON | Internal objective/predictor controls. T1/T2 largely rescale the same KL regularizer. Raw-CE diagnostics have a different bounded-loss definition. |
| Selected favorable 100-draw checks | Explicit exploratory appendix examples | Full MNIST seed 9001: T1 75.9712% versus PAC 75.7801%. Reduced MNIST `m=50`, seed 1003: T2 67.7729% versus PAC 67.4309%. Conditional MC intervals exclude selection and training-seed uncertainty. |
| Matched-strength FashionMNIST check | Appendix interpretation | T1 63.0982% versus PAC 63.0774% over two fixed checkpoints. The earlier larger gap shrank after missing LR/noise controls; treat the final result as a near tie. |
| Analytical separate/joint complexity curves | Theory section | Corrected `fig_bound_structure`, KL 2, variance proxies 1/4, failure budget 0.05, exact product-confidence equal and 99:1 allocations. Formula evaluation, not learning data. |
| Legacy conference tables/sweeps | Historical evidence, unless reevaluated | Earlier Theorem-3 or joint-ensemble objectives and inconsistent risk/trajectory logging cannot directly validate the current single-posterior theorem. New common-protocol tables replace them. |

The legacy four-scenario MNIST summaries still motivate severe impairment: their 100-seed grid favors the conference-style proposal by 1.5093 percentage points under combined fading/noise, while ERM wins in the other three scenarios. Corresponding MNIST checkpoints were not saved. The legacy two-seed mismatch prototype is superseded by the new three-seed bounded-loss experiment. Do not merge these different protocols into the main table.

## Manuscript consistency and submission priorities

- Main comparisons use the same bounded loss, preventing attribution of a loss change solely to regularization. Canonical raw-CE ERM remains supplementary.
- The campaign is exploratory: test outcomes informed later search rounds, although final configurations/checkpoints are validation-selected. Avoid claiming an untouched confirmatory test. A locked protocol with fresh seeds or a new held-out split would strengthen final claims.
- L2 Perfect is unavailable in the older MNIST/FashionMNIST cohorts and is included in CIFAR. Its older missing cells are not filled using a different dataset or a legacy model. Broader decoupled LR/L2 searches are needed before calling that baseline exhaustively optimized.
- Perfect describes the training forward channel; validation still uses deployment channels. These references are not untouched channel-unaware pretrained models.
- Update the algorithm initialization: fan-in-scaled random means, zero biases and posterior std 0.003/0.01 for the grayscale searches and 0.001/0.003 for CIFAR, not `q0=Q` with unit variance. The data-independent prior remains zero-mean unit Gaussian.
- Incorporate the smooth bounded-loss definition/Hoeffding justification and the TV shift specialization. A local weight-gradient norm is not a channel-Lipschitz certificate.
- The erasure experiment masks continuous features without quantization, coding, erasure flags or retransmission. Do not call it a complete bit-level BEC link.
- Feature power is unnormalized. Full MNIST means are about T1 23.06, ERM 15.88 and L2 5.29 per coordinate. State channel-law parameters; power-normalized experiments are needed for fixed-SNR or energy-efficiency claims, particularly for TCOM.
- Full CIFAR mean feature powers are T1 22.73, L2 12.59, ERM 6.71 and T2 19.40 per coordinate. Its accuracy improvement also does not establish a controlled-power advantage.
- MNIST/FashionMNIST use two-layer MLPs; CIFAR uses the shared compact CNN with 156,074 weights and biases and 128 transmitted features. Dataset, architecture, preprocessing, training count and budget all change, so CIFAR supports an extension to RGB/CNN inputs rather than an isolated architecture comparison. It does not establish transformer-scale performance.
- Nominal decomposition bars and corrected certificates use different confidence budgets; their components must not be interchanged.
- Replace the old simulation section and its tables rather than appending this draft: the old tables reuse labels and mix risk conventions.

## Source and reproducibility files

The main manuscript fragment is `simulation_section.tex`; optional appendix content is `supplementary_results.tex`. `main.tex` builds the preview. The original table generator writes seven table/CSV pairs and `data_manifest.json` with 93 hashed inputs. `../prepare_paper_figures.py` writes five vector PDF/PNG pairs and `assets/figure_data.json` with 41 hashed inputs. `generate_cifar_assets.py` adds two table/CSV pairs, the optional CNN validation plot, and `cifar_data_manifest.json` with 74 hashed inputs. Original trial records and all-method results remain under `results/journal_campaign/`, `results/journal_cifar/`, and the corresponding campaign/status reports.

Dataset citations were checked against the [authors' MNIST article page](https://bottou.org/papers/lecun-98h), [Fashion-MNIST paper](https://arxiv.org/abs/1708.07747), and [official CIFAR dataset page](https://cave.cs.toronto.edu/kriz/cifar.html). Both manual-IEEE and BibTeX entries are provided.
