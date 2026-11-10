# Journal simulation audit and experiment plan

The current main journal comparison uses ERM and L2 against Proposed T1/T2,
with the data-only PAC penalty retained as an optional ablation. The objective
equivalence and the existing main results are in `MAIN_COMPARISONS.md`.
Main figures use `plot_campaign.py --comparison main` (the default);
`--comparison all` preserves the full diagnostic view.

The journal story should connect finite reusable channel observations, channel generalization, deployment mismatch, and a complexity-controlled learner. Additional datasets help establish scope, but the first priority is aligning the implemented objective, evaluated predictor, and reported risks. The existing sweeps are useful pilot evidence; their saved bounds are not ready to serve as numerical certificates of the new Theorem 1.

This audit uses the draft supplied on 10 November 2026 and repository revision `c918333`. Original scripts and results are preserved. New implementation is isolated in `journal/` on branch `TIT_gpt`.

## Answers to the three checks

| Question | Finding |
| --- | --- |
| Do existing experiments optimize current Theorem 1? | `test12`–`test14` and MNIST `test17` use the marginalized conference-style structure: estimated channel mutual information plus mixture-to-prior KL, corresponding to current Theorem 3. `test15`–`test16` compute separate terms but train using the joint complexity, corresponding to current Theorem 2. Exact pasted table provenance still needs its run configuration, checkpoint and log; a table alone cannot establish the generating code revision. |
| Is Proposed Perfect correct? | Its default heuristic with `alpha=0` uses noiseless empirical CE plus a weighted data PAC-Bayes term, which has the intended training behavior. Its `objective="bound"` path additionally optimizes an invalid channel-Lipschitz proxy. Perfect-channel logged bounds remain problematic even with `alpha=0`. |
| Are existing code and results useful? | Yes, for choosing a compact validation grid, observing failure regimes, recovering mismatch curves, and inference-only reevaluation where checkpoints survive. They should retain their original theorem/objective labels. MNIST scenario sweeps retained JSON summaries but no `weights.pth` checkpoints. |

Code evidence: `test12.py:771`, `test13.py:796`, `test14.py:778`, `test15.py:797`, `test15.py:855`, `test16.py:452`, `test16.py:462`, and `test17.py:1109`.

For perfect training, with a fixed data-independent prior and `c0=(I,0)`, the intended objective is

\[
\mathcal J_{\rm perfect}(q_S)=\mathbb E_{W\sim q_S}\frac1n\sum_i\ell(W,c_0,Z_i)
+\eta_2\sqrt{\frac{2\sigma^2}{n-1}\left[D(q_S\Vert Q_W)+\log\frac{\sqrt n}{\delta}\right]}.
\]

Training uses the identity channel; testing uses the deployment distribution. The channel bank is deterministic, so its channel complexity vanishes. Deployment mismatch remains in the risk guarantee. Under the draft's uniform Lipschitz assumption, it is bounded by `K E[d(C',c0)]`. A fixed universal `K` makes this shift independent of the learned posterior, so it is excluded from optimization. The legacy norm of the gradient with respect to **weights** is not a Lipschitz constant with respect to the **channel** (`test17.py:1087`). Its channel-distance code also counts `width²` gain coordinates although the implemented channel is diagonal (`test17.py:219`, `test17.py:375`).

## What the recovered MNIST results show

Each scenario below contains all 100 paired seeds at saved settings `lr=0.0003`, `beta=0.1`, `gamma=1.0`, width 256, 150 epochs. Intervals describe paired training-seed variation, conditional on the saved dataset, fixed channel-bank protocol and test evaluations. They are pointwise descriptive intervals, not post-selection guarantees. ERM is deterministic while Proposed is a ten-component stochastic ensemble, so these comparisons also change predictor class and evaluation randomness.

| Deployment channel | ERM accuracy | Proposed accuracy | Proposed minus ERM in percentage points | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Near ideal | 98.2007% | 97.7161% | -0.4846 | [-0.5014, -0.4677] |
| Low SNR | 97.6658% | 97.1831% | -0.4827 | [-0.5021, -0.4632] |
| Severe fading | 86.9589% | 86.4149% | -0.5440 | [-0.6059, -0.4820] |
| Severe fading and low SNR | 80.5356% | 82.0449% | +1.5093 | [+1.4202, +1.5983] |

The last scenario favors Proposed in all 100 pairs; the first two favor ERM in all pairs. These results support a benefit under combined severe impairment, not a general claim that Proposed always improves accuracy. Searching for favorable reporting seeds would conceal this structure. Select hyperparameters on validation data/channels and report every predetermined seed, including runs that perform poorly.

`results/journal_audit/mnist_fixed_parameters.png` and `.pdf` show these comparisons. `legacy_mismatch.png` and `.pdf` recover both saved `test14` curves, with fixed deployment outage probability 0.5 and training outage probability from 0.1 to 0.9. That experiment already demonstrates the proposed mismatch idea; it needs the journal objective, controls and independent bank seeds for confirmatory use.

The metadata audit reads 71,412 run records from 353 files, keeping archived sources separate. It detects 841 overlapping fingerprints, five ambiguous method/configuration/seed groups and 22,370 unpaired proposed records across the older sweeps. These are provenance diagnostics, not additional independent runs. Reproduce the audit with:

```bash
conda run -n torch28 python -m journal.analyze_legacy \
  --root results/journal_legacy_snapshot --output results/journal_audit
```

The snapshot contains JSON and JSONL copied from Spartan, without modifying remote results. Its source hashes are in `results/journal_audit/source_inventory.json`. All generated results remain under ignored `results/`; analysis code and this report are versioned.

## Why the existing bound columns need revision

- **The predictor differs across risks.** Legacy empirical risk uses posterior means (`test17.py:673`), while deployment risk averages posterior samples (`test17.py:1249`). The draft's empirical and population risks both use the same posterior expectation.
- **The posterior changes during logging.** A final epoch's average of minibatch penalties is not the bound of the final saved posterior (`test17.py:1160`). Evaluate one frozen checkpoint and its saved bank.
- **Deterministic baselines have no finite Gaussian-prior KL as Dirac posteriors.** Omit their theorem-bound column, or explicitly construct and evaluate a randomized posterior. A zero complexity placeholder is not a certificate.
- **Raw CE does not automatically admit the hardcoded uniform sub-Gaussian constants.** Use the same bounded loss and units in empirical risk and certificate calculations. The new evaluator reports `min(CE,clip_ce)/clip_ce` in `[0,1]`, with valid variance proxies `1/4`; it still reports raw CE separately.
- **Channel-mixture estimates are not exact conditional mutual information.** Independently initialized components confound optimization variability with sensitivity to the channel bank. The estimate is bounded by `log(K)` and can saturate when high-dimensional components separate.
- **Legacy seed changes leave the bank fixed.** Bank RNG is 1337 (`test17.py:375`). New confirmatory runs should independently vary initialization and bank seeds, with identical paired seeds across methods.
- **Caches omit material settings.** Legacy hashes omit actual channel distributions and other constants (`test17.py:780`). New checkpoints include all channel settings, split and bank hashes, prior, loss definition, seeds and Git revision.

One pasted row already fails both natural interpretations of the bound column: S2 Proposed Perfect has population CE 2.3628, empirical CE 0.2040, and bound 1.9414. It exceeds the alleged population bound; its gap 2.1588 also exceeds 1.9414. Risk/MC conventions, assumptions and provenance must be checked before retaining that row. The supplied draft does not include proofs of the main theorems, so this audit does not certify their constants or derivations.

For bounded-loss evaluation and explicit Monte Carlo error accounting, see [Pérez-Ortiz et al., Tighter Risk Certificates for Neural Networks](https://jmlr.org/papers/v22/20-879.html). The current runner labels its outputs as diagnostics; a complete certificate additionally needs the relevant shift bound and allocated Monte Carlo confidence, and should report whether it is nonvacuous. Normalized bounded-risk bounds above 1 remain vacuous even if mathematically valid.

Local checkpoint-only inference of the two-epoch MNIST pilot passed using the same training/validation split hashes. Its final KL is approximately 798,516, giving channel and sample terms 63.51 and 2.69 for normalized bounded loss. Those defaults therefore give a vacuous bound, although the evaluator correctly separates the observed risks. Three posterior draws in this execution check also give a deliberately loose Monte Carlo radius; they are not sufficient for a scientific certificate. Use small toy networks such as `2→8→2` or `2→16→2`, modest KL and larger `m,n` to investigate nonvacuous behavior. Report an unscaled `eta=1` bound-minimization experiment separately from tuned-surrogate performance experiments. A smaller empirical variance does not justify lowering the uniform `1/4` proxy. Tightening MNIST certificates may need a genuinely independent prior-data split or a low-dimensional stochastic head; a prior learned on the same certification data would violate the current fixed-prior premise.

## Recommended experiment order

1. **Show the finite-channel mechanism on two moons.** Hold data, architecture, deployment law, training steps and prior fixed. Use matched laws and `m={5,10,20,50,100,200,500}`. Measure risk on training data with reused `U`, the same data with fresh training-law channels, and held-out data with fresh channels. Plot the observed channel gap against `m`, with uncertainty from independent bank/init seeds. Keep optimizer updates and per-step batch/channel sampling budget fixed when comparing different `m`.
2. **Separate data size and bank size.** Use two slices, varying `n` at fixed `m` and `m` at fixed `n`, or a small `n×m` grid. This directly tests why the Cartesian `nm` evaluations are not `nm` independent observations.
3. **Run objective controls with equal validation budgets.** Compare deterministic ERM/L2 for practical accuracy and stochastic ERM, sample-only PAC-Bayes, separate Theorem 1, and joint Theorem 2 using the same stochastic architecture and prior. Include channel-only ablation by setting `eta_data=0`. Both separate penalties depend on the same KL; at large KL they mainly change the coefficient of `sqrt(KL)`. A tuned sample-only control is essential to distinguish a channel-aware scaling rule from generic regularization strength.
4. **Use fixed deployment with varying training laws.** For outage deployment `p=0.5`, first use perfect and `p_train={0.1,0.3,0.5,0.7,0.9}`. Plot accuracy and raw CE against training probability; mark matched training. Include stochastic ERM and sample-only PAC-Bayes. Select optimizer/regularization settings on validation, not on the displayed test curve. Matching removes the shift penalty but need not maximize test accuracy.
5. **Add the decomposition figure.** Stack unweighted shift, channel-complexity and sample-complexity terms and overlay the observed gap using consistent bounded-loss units. For a population-risk bound, also stack empirical risk. Show empirical signed gaps in a separate panel, since they can be negative and are not the positive theorem terms. Include matched and mismatched cases. Perfect-to-noisy bars require a justified Lipschitz analysis or a separately valid bounded-loss shift inequality.
6. **Establish scale after the mechanism is stable.** MNIST plus FashionMNIST and MLP widths 64/256 are inexpensive first checks supported by the new runner. Add one compact split CNN and CIFAR-10 for a stronger TCOM evaluation if needed. The theory's abstract split does not restrict networks to two layers; the implementation must consistently define the transmitted representation. A TIT submission benefits more from controlled rate/decomposition/assumption experiments than a large model catalogue; a TCOM submission benefits from an additional practical architecture, realistic channel normalization and communication-cost comparisons.

For physical comparisons, state feature-power normalization, channel uses, split location, CSI/equalization assumptions and the meaning of SNR. Keep these consistent across methods; raw noise standard deviation alone does not fix SNR when learned feature power changes.

Use distinct tuning seeds and confirmatory seeds. A practical first confirmatory set is initialization seeds 1001–1005 paired with bank seeds 2001–2005, fixed data split seed 20261110, and independent evaluation seed streams. Report mean, standard deviation and paired confidence interval. Increase evaluation draws until their uncertainty is small relative to the claimed gain. Keep the prior fixed across hyperparameter selection, or account for selection among priors in formal certificates.

For one frozen posterior `q`, the observed decomposition is

\[
R_{\rm dep}-R_{\rm emp}
=(R_{\rm dep}-R_{\rm heldout,fresh,tr})
+(R_{\rm heldout,fresh,tr}-R_{S,\rm fresh,tr})
+(R_{S,\rm fresh,tr}-R_{\rm emp}).
\]

The new inference path computes all four risks using common posterior draws. These finite-test estimates decompose exactly algebraically; their Monte Carlo errors still need reporting.

## New runner and Spartan workflow

`theorem1.py` implements one diagonal-Gaussian posterior, one finite independently seeded bank, exact Gaussian KL, separate and joint terms, and the perfect-channel specialization. `run_experiment.py` supports moons, MNIST and FashionMNIST with a split MLP. New moons experiments sample independent Bernoulli labels, uniform arc angles and Gaussian noise, matching the theoretical IID-data model; `legacy_grid` preserves earlier sklearn checkpoints. The runner separates training from checkpoint-only inference. Training opens only training/validation data and selects the checkpoint by validation CE. In each evaluation replicate one sampled weight vector stays fixed across the entire data loader.

Training defaults to `training_loss="raw_ce"` plus tunable complexity penalties. This is a bound-inspired surrogate, not direct optimization of a certified raw-CE risk bound. For a proof-oriented experiment, set `training_loss="bounded_ce"`, `method="proposed"` and both `eta_channel=eta_data=1`; then the empirical term and penalties share the normalized bounded-loss units and match the trainable part of Theorem 1. The evaluation reports bounded-loss complexity diagnostics and refuses to invent a perfect-to-noisy Lipschitz certificate. Theorem 1 confidence uses conservative `delta/2` budgets; perfect training uses the full `delta` for data complexity.

```bash
# Locally: inspect and validate only. No sweep training.
conda run -n torch28 python -m unittest journal.test_theorem1 -v
conda run -n torch28 python -m journal.run_experiment --mode validate
conda run -n torch28 python -m journal.run_task \
  --manifest journal/configs/pilot.jsonl --validate-only

# Server: first fetch/pull the journal branch.
ssh unimelb
bash -l
cd ~/Git/bnn
git fetch origin TIT_gpt
git switch TIT_gpt
git pull --ff-only origin TIT_gpt
export DATASET=/home/yangshuo/Datasets
bash journal/submit.sh pilot journal/configs/pilot.jsonl
```

The pilot contains paired stochastic ERM/Proposed trials: two moons at width 64 and MNIST at width 256 with 55,000 training and 5,000 validation examples. Each runs only two epochs; it measures execution, memory and per-epoch cost rather than scientific performance. The job requests two CPUs, 8 GB and 20 minutes in `interactive`, running four trials sequentially. No GPU is needed for this check. Do not resubmit a completed pilot into the same output directories.

The launcher loads `Anaconda3/2024.02-1`, initializes bash conda, deactivates and activates `torch28`. It supplies the discovered `CONDA_ENVS_PATH` if absent because noninteractive bash does not read `.zshrc`. The dataset location still comes exclusively from `DATASET`.

The CPU pilot completed successfully as job **34666753** at revision `1dcd8e7`. The four sequential trials finished in about 47 seconds including process/environment overhead. Width-256 MNIST took 2.18 s/epoch for stochastic ERM and 2.29 s/epoch for Proposed, with 55,000 training and 5,000 validation examples. Training-process peak RSS was 593 MiB or less. These two-epoch runs establish execution and sizing, not scientific performance. At the measured rate, 150 epochs project to approximately 5.5–5.8 minutes before startup and additional final inference. Allow for variation across nodes and more evaluation draws.

For a manifest of independent trials, `bash journal/submit.sh cpu MANIFEST.jsonl` uses one configuration per array task and a default concurrency cap of eight. The measured MLP resource envelope is two CPUs, 4 GB and 20 minutes per task. Override `JOURNAL_CPUS`, `JOURNAL_MEM`, and `JOURNAL_TIME` for larger measured configurations. Estimate walltime from dataset loading plus epochs times measured epoch cost, validation overhead and separately budgeted final inference; include a modest margin for observed variation. Use measured peak RSS with headroom rather than the maximum available memory. GPU acceleration is unnecessary for the current MLP mechanism experiments.

Legacy MNIST jobs 30243181–30243184 ran for 9h33m–10h09m with 64 CPUs, requesting 1 TB but reaching only about 60 GB maximum RSS for the whole 64-worker job. Those ensemble-sweep measurements do not directly size the new single-posterior runner. No new 1-TB or 100-seed sweep is warranted before the pilot.

The 10 November queue check used `spartan-weather`: public A100 had 1,052 pending jobs and 91/100 cards in use; FEIT A100 had 164 pending jobs and 73/84 cards in use. These are a time-specific snapshot, not a wait-time guarantee. Live partition configuration allows `qos=feit` on `feit-gpu-a100` but not on public `gpu-a100`; the optional launcher therefore uses the FEIT partition. GPU submission is gated on a completed matching CPU pilot. A single A100 and two CPUs is the optional starting point; size host memory/time from the matching configuration and measure GPU utilization before batching configurations. No GPU job was submitted.

After selecting configurations using validation results, run checkpoint-only evaluation, locally or in a CPU array:

```bash
conda run -n torch28 python -m journal.run_experiment --mode evaluate \
  --checkpoint results/journal/CHOSEN_RUN --split test \
  --output results/journal/CHOSEN_RUN_test
```

Checkpoints contain `weights.pt`, `metadata.json` and `history.json`. Inference outputs `evaluation.json`. An optional evaluation JSON can override only deployment channel and evaluation settings. Missing checkpoints cause an error; inference never falls back to training.

## Draft consistency edits

Describe six table methods rather than four; align the stated `eta1,eta2,epsilon1,epsilon2` with the actual objective; identify whether `Bound` means a gap or population-risk bound; use unique table labels and six-row `multirow` groups. Replace claims that a few numerical rows validate a theorem with a statement that experiments illustrate its implications under the stated assumptions. Equal supports do not imply finite KL in general, and neither KL-based nor Wasserstein shift bounds uniformly dominate the other.
