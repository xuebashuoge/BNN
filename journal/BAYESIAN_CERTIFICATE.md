# Separate small-model all-Gaussian bound diagnostic

This is a separate numerical bound/decomposition experiment. It does not replace the main four-dataset accuracy comparison and does not search for a winning seed or hyperparameter. Every method uses the same diagonal Gaussian posterior over the **42 weights and biases** of a `2→8→2` MLP.

The earlier journal mechanism calculation used a selected stochastic toy posterior at width 8, `n=512,m=2000` and reported nonvacuous normalized-loss bounds; see `CAMPAIGN_RESULTS.md`. That result applies to its declared stochastic predictor and confidence allocation. It does not establish a corresponding finite Gaussian-prior bound for the historical deterministic ERM/L2 predictors. This new experiment revisits that scale with Gaussian ERM, Gaussian L2, T1 and T2 under a common architecture. Initial posterior standard deviation `.1` is an explicit setting of this separate proof-scale experiment, rather than a change to the main campaign.

The fixed matrix contains **48 checkpoints**: four algorithms × Gaussian analog/outage deployment × matched/perfect training × initialization seeds `76101,76102,76103`. Every seed is retained. Channel-bank seeds are initialization seed +10000; evaluation seeds are +20000. Dataset seed is 20261117 with the IID moons sampler and noise `.3`. There is one configuration per method/regime and no hyperparameter selection. Checkpoint selection within 150 epochs uses held-out validation accuracy and CE as the tiebreak, as in the main runner.

| Setting | Value |
| --- | --- |
| Training/validation observations | 512 / 512 |
| MLP width / finite training bank | 8 / 2000 |
| Epochs / batch size | 150 / 128 |
| Learning rate / scale learning-rate factor | .003 / .1 |
| Initial posterior std / fixed prior std | .1 / 1 |
| Smooth bounded loss epsilon | .01 |
| L2 coefficient | `1e-4`, multiplying `E_q[||W||²]` |
| T1 coefficients | channel `.01`, data `.03` |
| T2 coefficient | joint `.03` |
| Validation / final full-risk draws | 5 / 20 |

Both channel laws match the main campaign: Gaussian analog gain mean `.5`, gain std `1`, noise std `1`; outage probability `.5`. Matched training uses the deployment law, while perfect training uses the identity channel and is evaluated under the same noisy deployment law. T1 and T2 specialize to the same data penalty under perfect training, so their paired perfect references should coincide.

Generate the manifest and check it without opening datasets or starting training:

```bash
conda run -n torch28 python -m journal.build_bayesian_certificate
conda run -n torch28 python -m unittest journal.test_bayesian_certificate -v
conda run -n torch28 python -m journal.run_task \
  --manifest journal/configs/bayesian_certificate.jsonl --validate-only
```

The manifest and adjacent `.plan.json` record the fixed scope and SHA256. Entries carry `scientific_setup="small_model_bound_diagnostic"` and `main_accuracy_table=false`. Their standard eight method/regime labels allow ordinary campaign evaluation; their separate group names and output root keep the diagnostic identifiable.

Training belongs on Spartan in a separate committed source snapshot, with `torch28` active and the matching source/manifest hashes recorded. Use a short matching CPU resource pilot first. A proposed eight-CPU allocation can use four independent workers and two threads each; size memory/walltime from that pilot. The completed 48-case run uses:

```bash
python -m journal.campaign \
  --manifest journal/configs/bayesian_certificate.jsonl \
  --output-root results/journal_bayesian_certificate/20261117 \
  --workers 4 --threads 2 --evaluate-top 1 \
  --eval-draws 20 --full-risks --resume
```

Since every label has exactly one declared configuration and all three seeds, `--evaluate-top 1` retains all 48 trials. The 20-draw full evaluation measures empirical, fresh-channel and held-out Gibbs risks and their decomposition. The dedicated certificate evaluator additionally uses a fresh bank-only empirical Monte Carlo stream after checkpoint selection, with its own declared simultaneous confidence allocation. Its Monte Carlo radius and both unweighted theorem penalties enter the numerical certificates; tuned objective coefficients do not rescale the reported theorem bounds.

A finite KL makes bound calculation possible for every method. **Nonvacuity and proposed-method superiority are outcomes, not requirements imposed on the data.** Perfect-to-noisy channel shift, posterior complexity and simultaneous finite-Monte-Carlo confidence can make some or all bounds vacuous. In particular, a perfect-to-continuous-noisy total-variation shift can equal one. Report uncapped theorem right-hand sides, capped risk bounds, individual seeds and nonvacuity flags. These statements concern normalized smooth Gibbs loss, not unbounded cross entropy or classification error, and remain conditional on the supplied theorems and assumptions.

Export this diagnostic's accuracy, bound and decomposition figures to PNG, PDF and EPS without confidence shading. Individual runs, bound components and Monte Carlo information remain in the numerical exports. Results are pending until this separate campaign and its frozen-checkpoint certificate evaluation complete; no new training or test evaluation is performed by the builder or its unit tests.

The complete coordinator is `python -m journal.run_bayesian_certificate`. The Slurm wrapper `journal/bayesian_certificate.slurm` requests eight CPUs, 4 GiB and 15 minutes, checks the matching one-epoch pilot before full training, then evaluates 1,000 fresh bank-only draws for each checkpoint using four independent two-thread processes. Export `DATASET`, `JOURNAL_GIT_SHA=$(git rev-parse HEAD)` and `JOURNAL_MANIFEST_SHA256` for the committed certificate manifest before submission. Its `status.json` records progress and failures. The certificate family uses failure budget `.025`; the final main-family bound report separately uses `.025`, so their union failure budget is at most `.05`.
