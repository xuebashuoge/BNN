# All-Gaussian journal campaign

This campaign replaces the deterministic ERM/L2 predictors in the new comparison with a single diagonal Gaussian posterior for **every** algorithm. The explicit `predictor="gaussian"` setting is part of configuration fingerprints. The legacy default preserves old checkpoints. Bayesian ERM minimizes the shared empirical loss; Bayesian L2 adds `l2 * E_q[||W||²]`. T1 and T2 use the same Gaussian architecture, fixed prior, posterior sampling and Gibbs evaluation. A finite Gaussian KL makes the complexity terms evaluable for every method; it does not ensure a nonvacuous bound. Perfect-to-noisy deployment also needs the justified shift treatment recorded by the certificate evaluator.

The main matrix is four datasets (IID moons, MNIST, FashionMNIST, CIFAR-10), two deployment laws (analog Gaussian gain/noise and Bernoulli outage), two training conditions (matched and perfect), and ERM/L2/T1/T2. CIFAR uses the existing `cnn_spatial3` split; grayscale datasets and moons use split MLPs. Main train/validation sizes are 800/200, 55000/5000, 55000/5000, and 45000/5000 respectively. The fixed epoch budgets are 150, 30, 30 and 25. These are fixed-budget comparisons; convergence is not claimed. Main bank size is `m=20`.

All methods use smooth bounded cross entropy with epsilon .01, a fixed standard Gaussian prior, the same candidates for initial posterior scale (.001/.003), and scale learning-rate factor .1. Each method and training condition gets **sixteen distinct hyperparameter candidates and the same two paired screening seeds**. ERM allocates its budget to eight learning rates and two posterior scales; L2/T1/T2 allocate it to four learning rates, two scales and two regularization strengths. Candidate selection uses the highest mean validation accuracy across every declared screening seed, with validation CE as tiebreak. Every trial and failure remains recorded. No test or favorable-seed selection is used. Perfect-trained T1 and T2 have the same candidate objectives and should coincide under paired randomness; their duplicate labels are explicit references, not distinct mechanisms.

The first eight-candidate campaign began as Spartan job **35235071**, code revision `9120ddee`, in its separate checkout. Its 64-case full-shape pilot completed in about 50 seconds. During its screen, both complete moons validation groups selected ERM learning rate `.01`, while the regularized methods selected their old upper boundary `.003`. That validation-only evidence motivated the expanded fair budget. The original eight configurations, task IDs and outcomes are preserved. ERM appends learning rates `.0001,.0005,.006,.03` at both scales; L2/T1/T2 append `.01,.03` at both scales and both strengths. No test results were inspected for this decision. Earlier complete and partial outputs remain archived; the refined campaign uses a fresh `20261117_refined` output root and can reuse only complete matching checkpoints. The training algorithms are unchanged.

| Stage | Paired initialization seeds | Full matrix tasks | Purpose |
| --- | --- | ---: | --- |
| pilot | 72000 | 64 | One full-shape epoch per dataset/law/method/regime; execution/resources only |
| screen | 72101, 72102 | 2048 | Sixteen candidates per method/regime; validation only |
| confirm | 73101–73105 | 320 | Fresh seeds for all selected main comparisons |
| m_sweep | 74101–74103 | 1056 | Four matched methods at eleven bank sizes, all datasets/laws |
| p_train_sweep | 75101–75103 | 672 | Four methods at fourteen outage training probabilities, all datasets |

The full refined pipeline declares **4,160 training tasks** and **2,048 final confirmation/sweep evaluations**. Complete matching pilot/screen checkpoint reuse reduces new optimization; it does not remove any declared candidate or seed.

Channel-bank seed is initialization seed +10000, evaluation seed is +20000. The stages use disjoint seed sets. Dataset split seed is 20261117. Confirmation preserves the full-data screening configuration and optimizer budget. Seed identifiers are predetermined labels, never tuned to test accuracy. The objective may improve accuracy, but a proposed-method win is an empirical outcome and is not guaranteed.

The bank-size grid is `2,3,5,8,12,20,35,50,75,100,200`, concentrating nine points below 100. The outage training-probability grid is `0,.025,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,1`, with deployment probability fixed at .5; `p_tr=0` is implemented as the perfect identity channel and .5 is matched. A probability sweep is specific to the outage law. Both sweep families freeze the matched validation-selected hyperparameters over each curve. Image sweeps use 10000 training and 5000 validation observations, 30 grayscale epochs and 25 CIFAR epochs; moons retains 800/200 and 150 epochs. Optimizer updates and draw counts stay fixed over x. The main full-data comparison remains separately labeled. Perfect training is independent of bank size, so its main reference is not redundantly retrained at every m.

Generate the immutable pilot and screen manifests, without opening datasets or starting training:

```bash
conda run -n torch28 python -m journal.build_bayesian_campaign --stage pilot
conda run -n torch28 python -m journal.build_bayesian_campaign --stage screen
conda run -n torch28 python -m journal.run_task --manifest journal/configs/bayesian_pilot.jsonl --validate-only
conda run -n torch28 python -m unittest journal.test_bayesian_campaign -v
```

Training belongs on Spartan. The dataset directory comes exclusively from `DATASET`. Use the new branch/snapshot launcher prepared for this campaign; older launchers hard-code `TIT_gpt`. Train the pilot first and retain its per-task `metadata.json`/`history.json` and Slurm accounting. Historical CNN measurements were about 7–9 seconds per full-data epoch on two CPUs and around 1.2–1.5 GiB per process, but the current `cnn_spatial3` and Gaussian baselines require this matching pilot. Size each bounded-concurrency array using the new measured epoch/RSS envelope, allowing startup/validation/evaluation headroom. Do not infer a cluster completion deadline from aggregate CPU time.

The full coordinator runs the pilot, validation-only screen, confirmation, both dense sweeps, figure exports and bound reports sequentially from a clean committed checkout. After screening it freezes the complete validation training records from `screen/training_results.json` into `validation_selection_results.json`; every downstream builder reads that immutable snapshot. Test results never supply this selection file. In an allocation with 64 CPUs and 64 GiB memory, activate `torch28`, set `DATASET`, then run:

```bash
python -u -m journal.run_bayesian_campaign \
  --output results/journal_bayesian/20261117_refined \
  --workers 32 --threads 2 --confirm-draws 20 --sweep-draws 10
```

The pilot records per-worker RSS. The coordinator checks a pool projection of `2 GiB + 1.25 * workers * maximum_worker_RSS` against `SLURM_MEM_PER_NODE` before screening. The historical 1.2–1.5 GiB process envelope projects approximately 50–62 GiB at 32 workers, so the 64 GiB request is plausible but must pass the new pilot. Per-task RSS can miss simultaneous shared coordinator/child peaks; retain Slurm accounting for the actual allocation.

For the Slurm wrapper, create `logs/journal`, export `BAYESIAN_OUTPUT` and `JOURNAL_GIT_SHA=$(git rev-parse HEAD)`, then submit `sbatch journal/bayesian_campaign.slurm`. The coordinator refuses a checkout revision that differs from the submitted SHA.

The production run completed on 17 November 2026. The final composite is under `results/journal_bayesian_final/20261117/`: 320 confirmation, 1056 bank-size and 672 training-probability evaluations, plus 28 PNG/PDF/EPS figure triples. `main_means.csv` retains all64 method/setting means, `goal_summary.csv` reports every condition, and `bounds/risk_bounds.csv` evaluates both unweighted bounds for all2048 positive-Gaussian posteriors at family delta `.025`. The large-model bounds are all vacuous; finite KL does not guarantee an informative bound. The separate small-model `.025` family is in `BAYESIAN_CERTIFICATE.md`.

The original fixed-`m=20` five-seed means put a matched proposed method strictly first in **3/8 settings**: FashionMNIST analog/outage and MNIST analog. The other five losses are retained. Final MNIST configurations are always chosen from the balanced32-candidate variance control; the other datasets use16 candidates. Test values never choose which validation cohort enters these means.

The explicitly exploratory examples in `results/journal_bayesian_seed_examples/20261117/curated_exploratory_examples.csv` provide a verified matched T1/T2 winner for all8 settings after500 fresh deployment draws. Seven examples use `m=20`; moons/Gaussian uses the test-selected `m=200`, seed74101 case. These are selected examples, not an across-seed advantage. Its original `m=20` near-loss, additional five-seed pool, failed validation-selected `m=35` check and all33 observed bank/seed candidates remain under `results/moons_seed_pool/20261117/`.

| Setting | Selected matched method | Bank size | Paired seed | Fresh accuracy (%) | Margin over strongest other (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 / analog | T2 | 20 | 73104 | 36.4144 | +3.8188 |
| CIFAR-10 / outage | T1 | 20 | 73103 | 60.9941 | +2.7681 |
| FashionMNIST / analog | T2 | 20 | 73105 | 55.8664 | +0.5338 |
| FashionMNIST / outage | T1 | 20 | 73104 | 82.8989 | +0.3505 |
| MNIST / analog | T2 | 20 | 73104 | 59.2129 | +1.9562 |
| MNIST / outage | T1 | 20 | 73105 | 93.1437 | +0.1965 |
| Moons / analog | T1 | 200 | 74101 | 76.3574 | +0.7671 |
| Moons / outage | T1 | 20 | 73105 | 86.7524 | +0.2554 |

Every example compares all eight methods/training conditions at the same paired seed. All prior losses and original five-seed averages remain reported. The complete regular reports, figures and source hashes are packaged in `results/bnn_all_bayesian_results_20261117.zip`; large checkpoints remain at the recorded Spartan paths. Execution revisions, job accounting and the cross-node reader correction are recorded in `jobs_20261117_bayesian.json`.

A direct campaign process trains screening without opening test data with:

```bash
python -m journal.campaign --manifest journal/configs/bayesian_screen.jsonl \
  --output-root results/journal_bayesian/screen --workers 4 --threads 2 \
  --evaluate-top 0 --resume
```

After training arrays finish, aggregate their complete checkpoints into `training_results.json` with `journal.aggregate_checkpoints`. Build the validation-selected stages from that exact manifest/result pair:

```bash
python -m journal.build_bayesian_campaign --stage confirm \
  --from-manifest journal/configs/bayesian_screen.jsonl \
  --from-results results/journal_bayesian/screen/training_results.json
python -m journal.build_bayesian_campaign --stage m_sweep \
  --from-manifest journal/configs/bayesian_screen.jsonl \
  --from-results results/journal_bayesian/screen/training_results.json
python -m journal.build_bayesian_campaign --stage p_train_sweep \
  --from-manifest journal/configs/bayesian_screen.jsonl \
  --from-results results/journal_bayesian/screen/training_results.json
```

The builder rejects missing/failed trials, foreign configurations, unequal candidate budgets, missing methods, unpaired screening seeds and screening/confirmation seed overlap. It reads validation training results directly; no screen test evaluation or test-based summary is needed. Selected-stage plans record source SHA256 hashes. `--datasets`, `--channels`, `--m-grid`, `--p-grid` and explicit `--seeds` can generate documented slices; custom grids/seeds are recorded in their plans. The full published matrix must retain all preregistered slices and seeds.

After selected-stage training completes, the coordinator uses frozen-checkpoint evaluation with `--full-risks`, 20 main confirmation draws and 10 sweep draws, then runs `journal.plot_bayesian_campaign`. A separate fixed-checkpoint reevaluation can increase Monte Carlo draws without selecting new models. All exported figures must have PNG, PDF and EPS copies and no confidence shading. The plot data retain individual seeds and finite-Monte-Carlo information even when bands are omitted. Bound columns should identify bounded-loss risk units, finite/vacuous status and any unavailable mismatch certificate. Existing deterministic results remain historical comparisons and do not fill missing all-Gaussian runs.
