# Prespecified final Bayesian result assembly

The final main cohort uses **all MNIST cases selected by the separate 32-candidate validation control** and **all non-MNIST cases selected by the original fair16 main screen**. This source rule is fixed before the assembler opens any test tables. It always uses the MNIST32 control, even if original-scale candidates win its validation screen or its final test scores are lower than the earlier MNIST cohort. There is no test-based fallback, favorable-seed filter, per-seed method switch or new training.

The prescribed source revisions are main `f602987` and MNIST scale control `7087207`. Both source pipeline status files must record completion. The helper verifies committed scientific/builder code equivalence, then replays main16 selection and MNIST32 selection directly from their frozen training/validation records. It checks captured original inputs, source fingerprints, the full paired 1024-record MNIST validation family, saved finalists and every final-stage manifest against that replay. No source test table is opened before those choices are validated.

The composite keeps the original stage dimensions:

| Stage | Three non-MNIST datasets from main16 | MNIST from the32-candidate control | Composite |
| --- | ---: | ---: | ---: |
| Confirmation | 240 | 80 | 320 |
| m sweep | 792 | 264 | 1056 |
| p_train sweep | 504 | 168 | 672 |
| Total | 1536 | 512 | 2048 |

Every confirmation condition has ERM/L2/T1/T2 under matched and perfect training and all five fixed seeds. Both sweeps retain all three fixed seeds and every declared grid point. The 512 old primary MNIST final cases remain at their source and appear explicitly in the exclusion audit; their test results never determine the replacement decision. An original-scale candidate may win the32-candidate validation screen and produce the same task ID in both cohorts; the audit identifies excluded and included physical source checkpoints separately by cohort and stage. Sources are not deleted or overwritten.

Run on the server after both pipelines finish. Model weights remain on the server; integrity and formal-bound evaluation need access to them. Supply the original absolute roots, since the source capture and selection protocols record those paths:

```bash
conda run -n torch28 python -m journal.assemble_bayesian_results \
  --main-root /home/yangshuo/Git/bnn_bayesian_refined_20261117/results/journal_bayesian/20261117_refined \
  --mnist-root /path/to/completed/mnist_scale_control_output \
  --output /path/to/fresh/composite_output \
  --run-bounds --run-plots
```

The output must be absent and separate from both source roots and every resolved source stage. Source main `p_train_sweep` may be the intentionally published directory symlink to its permanent external cache. The assembler validates that source's real checkpoint directories and creates **owned leaf directory aliases** in its own output. Existing aliases, files, directories and broken links are refused. It writes only its new manifests, canonical global rows, audit, tables and optional output artifacts. Per-checkpoint metadata, weights, histories, logs and full reports are never changed.

Validation requires exact training/test identities and configurations, complete histories, correct source revisions, finite actual Gaussian KL, saved tensor/bank integrity, full matched-risk/decomposition reports and the fixed20-draw confirmation /10-draw sweep streams. Saved metadata/weights/evaluation hashes are checked. Source bytes and intentional stage-directory pointers are rechecked before delivery. Two CPU threads are used for integrity/KL operations; no datasets are opened, no optimizer is created and no Monte Carlo inference is rerun.

The owned output contains:

- `manifests/{confirm,m_sweep,p_train_sweep}.jsonl`, with source-cohort metadata and unchanged NN configurations/task IDs.
- Each stage's canonical `training_results.json` and `test_results.json`, plus leaf aliases to the verified actual checkpoints.
- `evaluation_reports/{stage}/`: all2048 full numerical evaluation reports, copied byte-for-byte into regular owned files for downloading without weights.
- `assembly_audit.json`, with validation replay, source hashes, exact aliases and checkpoint revisions, plus excluded old MNIST task IDs.
- `main_means.csv`/`.json`: all64 condition/method means over all five seeds, including accuracy, raw CE, bounded risks and observed bounded-loss gap.
- `goal_summary.csv`/`.json`: all eight dataset/channel conditions, their actual mean margins and goal flags.
- `accuracy_outcome.json`: the latest ranking logic with tolerance `1e-12`, one matched T1/T2 candidate fixed across each condition's five seeds, and per-seed margins. Ties and losses stay visible.
- `assembly_status.json`, distinguishing assembly completion from optional pending bounds/figures.

With `--run-bounds`, `journal.bayesian_bounds` recomputes both unweighted theorem right-hand sides for all **2048 final posteriors in one family at delta=.025**. Together with the separate small-model family's delta=.025, the stated union failure budget is .05. Original main bound files are retained as superseded source artifacts; they are not additional members of the final composite confidence claim. Finite Gaussian KL does not promise nonvacuity, and a proposed-method accuracy win is an observed outcome rather than an imposed result.

With `--run-plots`, the standard plotter exports **28 figure triples**—PNG, PDF and EPS—with no confidence shading. Individual seeds and numerical information remain available in the tables and figure data. Flags may be omitted for assembly-only operation, then the same standard bounds/plot CLIs can consume the three composite manifests and stage roots later.

For an assembly-only run, the explicit later commands are:

```bash
ASSEMBLED_ROOT=/path/to/composite_output
conda run -n torch28 python -m journal.bayesian_bounds --delta .025 \
  --campaign "$ASSEMBLED_ROOT/manifests/confirm.jsonl" "$ASSEMBLED_ROOT/confirm" \
  --campaign "$ASSEMBLED_ROOT/manifests/m_sweep.jsonl" "$ASSEMBLED_ROOT/m_sweep" \
  --campaign "$ASSEMBLED_ROOT/manifests/p_train_sweep.jsonl" "$ASSEMBLED_ROOT/p_train_sweep" \
  --output "$ASSEMBLED_ROOT/bounds"
conda run -n torch28 python -m journal.plot_bayesian_campaign \
  --source "$ASSEMBLED_ROOT/confirm" "$ASSEMBLED_ROOT/manifests/confirm.jsonl" \
  --source "$ASSEMBLED_ROOT/m_sweep" "$ASSEMBLED_ROOT/manifests/m_sweep.jsonl" \
  --source "$ASSEMBLED_ROOT/p_train_sweep" "$ASSEMBLED_ROOT/manifests/p_train_sweep.jsonl" \
  --output "$ASSEMBLED_ROOT/figures"
```

Download global JSON/CSV files, manifests, `evaluation_reports/`, audit, bounds and figures for local review. The leaf aliases are server pointers; do not dereference the entire assembled tree during transfer, which would copy the source model weights. Keep the original server checkpoints and permanent published p-stage cache available for reproducibility. This document describes the prepared workflow and does not claim either source pipeline or final assembly has completed.
