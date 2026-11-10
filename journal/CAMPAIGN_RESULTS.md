# Journal simulation results

The completed campaign contains 473 unique training trials searching for settings where a Theorem 1 or Theorem 2 learner has the highest wireless deployment test accuracy. Selected proposed models improve on the evaluated baselines, but broader comparisons include near ties with tuned sample-only PAC-Bayes and conditions where a baseline wins. Every trial is retained, and validation-based configuration selection, exploratory test/seed selection, and fresh posterior/channel Monte Carlo verification are distinguished. Results refer to split MLPs and finite reusable channel banks; they do not establish a universal accuracy ordering among learning algorithms.

## Completed screening and verification

The main analog deployment law has independent coordinate gains `N(0.5,1)` and additive noise `N(0,1)`. Training is matched unless explicitly labelled perfect. Feature power is measured but not normalized, so the noise parameter should not be described as a fixed physical SNR across learned models.

The following are actual 100-draw checkpoint-only inference results. Intervals concern Monte Carlo error conditional on the fixed trained models and test set; they exclude training-seed, test-data sampling and selection uncertainty.

| Setting | Proposed | PAC baseline | Difference in percentage points | Scope |
| --- | ---: | ---: | ---: | --- |
| MNIST, 10,000 training examples, width 128, `m=100`, seed 1002 | T1 70.8332% | 70.5801% | +0.2531 | Initial PAC grid; exploratory selected training seed; MC interval [0.2024,0.3038] |
| MNIST, 10,000 training examples, width 128, `m=50`, seeds 1003 and 1004 | T2 66.9905% | 66.8195% | +0.1710 | Mean of two fixed checkpoints; one seed improves, the other ties |
| MNIST, 55,000 training examples, width 128, `m=100`, seed 9001 | T1 75.9712% | 75.7801% | +0.1911 | Test-selected full-data configuration/seed; all three PAC strengths rechecked; conditional MC interval [0.1080,0.2742] |
| FashionMNIST, 10,000 training examples, width 128, `m=100`, seeds 1001 and 1002 | T1 63.0982% | 63.0774% | +0.0208 | Expanded PAC control with matched leading KL strength; conditional MC interval [0.0088,0.0328]; near tie |

For the `m=50` MNIST case, seed 1003 gives T2 67.7729% versus PAC 67.4309%, a conditional MC margin of 0.3420 percentage point with approximate paired interval [0.2852,0.3988]. Seed 1004 gives identical sampled accuracies in all 100 draws; this does not prove exact predictor equivalence. Averaging these two fixed checkpoints gives a conditional MC margin of 0.1710 percentage point with interval [0.1426,0.1994]. The separate descriptive training-seed interval is wide, [-2.0018,2.3438] percentage points, because only two training seeds are available. PAC and T2 have matched leading KL strengths. The improving seed selects different epochs (PAC 28, T2 24), while the tied seed selects epoch 31 for both and has nearly identical parameters.

For FashionMNIST before the additional controls, seed 1002 gives T1 63.1628% versus PAC 62.0757%, while seed 1001 is a near tie with stochastic ERM (63.0336% versus 63.0944%). Thus the initial claim that T1 wins against every baseline in both seeds does not survive the larger Monte Carlo evaluation. These outcomes remain visible when presenting selected favorable cases.

The configurations used in these selected comparisons are:

| Setting | Learning rate | Initial posterior std | Training epochs | Channel/data coefficients |
| --- | ---: | ---: | ---: | --- |
| MNIST T1, `m=100` | 0.003 | 0.01 | 25 | `eta_channel=0.01`, `eta_data=0.1` |
| MNIST T2, `m=50` | 0.003 | 0.01 | 35 | joint coefficient `0.005985352343443053` |
| MNIST T1, full-data selected seed | 0.003 | 0.01 | 100 | `eta_channel=0.003`, `eta_data=0.03` |
| FashionMNIST T1 | 0.001 | 0.01 | 25 | `eta_channel=0.003`, `eta_data=0.03` |

All use batch size 256, fixed Gaussian prior standard deviation 1, smooth bounded loss with epsilon 0.01, and validation-selected checkpoints. Each configuration and all seeds remain in the versioned manifests. The initial posterior mean uses independent random layer initialization with zero biases; it is not initialized at the zero-mean unit-variance prior. The manuscript pseudocode should match this initialization.

## Full-data comparisons

The full-data extension uses 55,000 training and 5,000 validation examples, width 128, 100 epochs, and initialization/channel-bank seeds 9001–9003. Directly transferring settings from the 10,000-example screen gave T1 74.490% and T2 73.280%, versus PAC 76.637%. The final stage adds two weaker configurations for each proposed objective and two additional PAC coefficients, reusing all 27 existing checkpoints. Every one of the resulting 45 full-data checkpoints is evaluated, including 12 new checkpoints not selected by validation. No additional training is needed for this exploratory test-grid check.

| Method | Validation-selected configuration, mean test accuracy | Best test-selected configuration, mean test accuracy |
| --- | ---: | ---: |
| PAC matched | 76.6370% | **76.7513%** |
| T1 matched | 76.6307% | 76.6307% |
| T2 matched | 76.4347% | 76.4347% |
| Stochastic ERM matched | 74.9907% | 74.9907% |
| Bounded-loss ERM matched | 74.5980% | 74.5980% |
| Raw-CE ERM matched | 74.3007% | 74.3007% |
| L2 matched | 70.1150% | 70.1150% |
| ERM perfect | 58.0730% | 58.0730% |
| PAC perfect | 57.9363% | 57.9363% |

The validation-selected T1 and PAC configurations are effectively tied: T1 is lower by 0.0063 percentage point. Allowing the explicitly requested test-based hyperparameter search gives PAC coefficient 0.1 the highest three-seed mean, 0.1207 percentage point above T1. The full-data highest-mean goal is therefore not met by the proposed learner within this tested grid. T1 nevertheless improves on practical stochastic ERM and deterministic ERM by 1.6400 and 2.0327 percentage points respectively.

The best full-data T1 coefficients are `eta_channel=0.003, eta_data=0.03`; T2 uses joint coefficient `0.0015`; the validation-selected PAC coefficient is `0.06`, while test selection prefers `0.1`. All matched stochastic models use learning rate 0.003 and initial posterior standard deviation 0.01. The complete 45-checkpoint outcomes and test-selection rules are saved in `results/journal_campaign/full_refine/exploratory_full_grid.json` and `.csv`. Test-selected configurations and training seeds are exploratory and do not inherit independent test-selection guarantees.

The test-based search also compares each proposed configuration against the strongest evaluated baseline at the same training seed. Its largest positive margin occurs at seed 9001: T1 gives 76.189% versus PAC coefficient 0.1 at 76.100% in the original 10-draw estimates. A fresh 100-draw paired stream at evaluation seed 898001 reevaluates this T1 checkpoint and all three PAC coefficients, 0.03, 0.06 and 0.1. Their accuracies are T1 75.9712%, PAC 75.7801%, 75.7456% and 75.7537% respectively. T1 exceeds the strongest fresh-MC PAC control, coefficient 0.03, by 0.1911 percentage point. Its approximate pointwise MC interval is [0.1080,0.2742], and the interval adjusted simultaneously for all three PAC comparisons is [0.0891,0.2931].

This verifies the selected fixed-checkpoint improvement under the requested test/seed search, not highest mean across training seeds. The best proposed margins at seeds 9002 and 9003 are negative, -0.3650 and -0.2470 percentage point, on the complete 10-draw grid. Other practical baselines retain their lower 10-draw estimates. The fresh intervals account for posterior/channel Monte Carlo uncertainty within the selected setting, excluding training-seed, test-data and whole-campaign selection uncertainty. Details are in `exploratory_full_high_mc_summary.json`.

## Expanded FashionMNIST controls

The initial FashionMNIST grid coupled learning rate and posterior noise. Its selected T1 configuration used learning rate 0.001 and posterior standard deviation 0.01, a combination absent from the initial PAC and stochastic-ERM controls. The final stage preserves all 44 original trials and adds eight PAC trials matching the four T1 candidates' learning rate, posterior noise and leading KL strength, plus four missing stochastic-ERM combinations. Configuration selection is rerun using aggregate validation accuracy across both seeds.

| Method | Initial-grid mean test accuracy, 10 draws | Expanded-control mean test accuracy, 10 draws |
| --- | ---: | ---: |
| T1 matched | 63.2370% | **63.2370%** |
| T2 matched | 63.2085% | 63.2085% |
| PAC matched | 62.4195% | 63.1685% |
| Stochastic ERM matched | 62.4355% | 63.0105% |

The expanded PAC control reduces the apparent T1 advantage from 0.8175 to 0.0685 percentage point in the 10-draw comparison. T2 is only 0.0400 percentage point above PAC. The selected PAC coefficient is `0.06014962686336267`, with learning rate 0.001 and initial posterior standard deviation 0.01; it matches the leading KL strength of T1's coefficients `0.003,0.03`. The final stochastic-ERM control uses the same learning rate and posterior noise.

Fresh 100-draw paired inference against the expanded PAC control gives T1 63.0982% and PAC 63.0774%, a fixed-two-checkpoint mean difference of only 0.0208 percentage point with approximate conditional MC interval [0.0088,0.0328]. Seed 1001 ties in all 100 sampled accuracy replicates, while seed 1002 favors T1 by 0.0416 percentage point with interval [0.0176,0.0656]. The tied seed has nearly identical parameters and the same selected epoch. These are practical near ties, not evidence of a substantially different regularization mechanism. T1's earlier 0.62255-percentage-point 100-draw advantage over the old PAC grid is superseded by this matched-control comparison. The final report is `results/journal_campaign/fashion_controls/fashion_controls_high_mc_summary.json`.

## Interpretation of the objectives

The mathematical reason for these controls is substantive. Writing the separate penalties as `eta_ch sqrt[b(D+c_m)] + eta_data sqrt[a(D+c_n)]`, their gradient matches a sample-only PAC gradient with the local coefficient

\[
\gamma(D)=\eta_{\rm data}+\eta_{\rm ch}\sqrt{b/a}\sqrt{(D+c_n)/(D+c_m)}.
\]

For the large MNIST KL values, the last ratio is close to one. T2 is also a single square-root KL regularizer with adjusted scaling and confidence. Accuracy gains over ERM can therefore coexist with near-equivalence to a suitably tuned PAC baseline. A defensible algorithmic interpretation is theory-guided dependence on channel-bank size, rather than a wholly different regularizer family.

## Loss assumptions and numerical bounds

The campaign loss is

\[
\ell_\epsilon=-\log[\epsilon+(1-\epsilon)p_y]/\log(1/\epsilon),\qquad\epsilon=0.01.
\]

It lies uniformly in `[0,1]`, so the sub-Gaussian variance proxies are `1/4`. Perfect-channel training has zero channel variance. Ordinary cross entropy and accuracy remain separate metrics. The raw-CE ERM reference is retained and excluded from smooth-loss bound comparisons. The manuscript-ready derivation is in `bounded_loss.tex`.

One chosen width-8 IID-toy posterior at `n=512,m=2000` was evaluated using a new 100-draw empirical-risk stream, independent of checkpoint selection. The confidence budget allocates 0.04 across both theorems and the seven matched-erasure `(n,m)` settings with three channel-bank seeds each (21 declared setups), and 0.01 to the empirical Monte Carlo upper bound. With data-independent prior fixed, posterior/hyperparameter selection within each setup is covered by the uniform posterior event.

| Quantity | Value |
| --- | ---: |
| Empirical normalized loss estimate | 0.0909427 |
| One-sided MC radius | 0.1517427 |
| T1 channel complexity | 0.2129374 |
| T1 data complexity | 0.4203686 |
| T2 joint complexity | 0.4750782 |
| Matched-channel shift | 0 |
| T1 population-risk upper bound | 0.8759914 |
| T2 population-risk upper bound | 0.7177636 |

Both bounds are nonvacuous on the normalized-loss scale. This is conditional on the supplied theorems and their assumptions, whose proofs are not verified by the numerical calculation. It is not a direct error-rate or raw-CE certificate. The saved sampler source, split and channel-bank checks pass. The report is `results/journal_campaign/certificate_m2000/certificate.json`.

For bounded loss, channel shift can also be bounded by total variation without a channel-Lipschitz assumption. Exact independent-erasure TV supports finite perfect-training shift bars even when channel KL is infinite. This is an added bounded-loss specialization, not a claim that a weight-gradient norm supplies the draft's Lipschitz constant.

## Simulation sequence findings

- The finite-channel phenomenon is visible: T1's normalized channel gap falls from about 0.010314 at `m=5` to 0.001605 at `m=20` and near zero by `m=100`. The methods largely overlap here; this supports the mechanism rather than a strict accuracy ordering.
- Under fixed deployment erasure probability 0.5, T1 test accuracy is 83.126%, 84.418%, 84.903%, 84.839% and 84.266% for training probabilities 0.1, 0.3, 0.5, 0.7 and 0.9. The matched point also minimizes CE. Perfect-trained PAC gives 81.870%, so the matched proposed learner gains roughly three percentage points over that reference.
- Accuracy rises with data size, but the observed data gap is negative in these particular three-seed runs. These results do not establish a quantitative empirical `n`-rate; independent data resampling would be needed for that stronger claim.
- All 168 full-risk reports pass the signed four-risk telescoping check. All 7,392 checked bounded-loss means and replicates lie in `[0,1]`. Deterministic baselines have no fabricated finite Gaussian-prior KL.

The final 17 figures and matching PDF exports are in `results/journal_campaign/figures_final/`, incorporating the completed full-data and expanded FashionMNIST controls. The bound bars evaluate unweighted T1 terms at each frozen posterior, including models trained with T2; hollow-square markers show the corresponding T2 bound. Raw weighted training penalties and changing minibatch posteriors are not used as certificates.

## Reproducibility and scope

Training ran on Spartan using `torch28`; local work consists of metadata analysis, inference and figure generation. The training checkout stayed at `69fa1d1` through the dependent job chain. Later commits contain manifests, analysis tools and manuscript notes. Job IDs, resource requests and completion states are recorded in `jobs_20261110.json`.

| Manifest | Entries | Newly trained unique trial IDs |
| --- | ---: | ---: |
| Initial screen | 104 | 104 |
| Follow-up including full-data, refinement, mechanism and mismatch | 295 | 295 |
| FashionMNIST screen | 44 | 44 |
| Full-data refinement | 45 | 18 |
| FashionMNIST expanded controls | 56 | 12 |
| Total | 544 | **473** |

The 71 repeated manifest entries reuse 27 original full-data and 44 original FashionMNIST checkpoints, with no conflicting duplicate configurations. There are 141 unique MNIST, 276 unique moons and 56 unique FashionMNIST trial IDs. Resource pilots are separate from these 473 comparison trials. All training and campaign evaluations completed without reported failures; all jobs used CPUs. The final 20 mathematical and protocol checks pass.

All saved trial outcomes and selection rules remain available. Seed-selected and test-selected findings are exploratory. A positive conditional Monte Carlo interval does not establish a population advantage across training seeds. The current benchmarks use reusable finite channel banks; they should not be described as comparisons against algorithms given unlimited fresh simulated training channels.
