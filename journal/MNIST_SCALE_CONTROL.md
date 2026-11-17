# MNIST fixed Gaussian scale validation control

This separate extension investigates whether optimizing posterior scale affects
the MNIST comparison. It adds sixteen candidates to **every** method/training
regime under both deployment laws. Each new candidate is the exact counterpart
of an existing sixteen-candidate screen configuration, with only
`rho_lr_scale=.1` changed to `rho_lr_scale=0`. ERM, L2, T1 and T2 keep the same
positive Gaussian posterior standard deviation, `.001` or `.003`. Their
posterior means still train. No neural network implementation changes are
needed; freezing positive variance retains a Gaussian Bayesian predictor and a
finite KL to the Gaussian prior.

The original 512 MNIST tasks, configurations and IDs remain intact. The extra
512 tasks use the existing `bayes_screen` ID construction and paired screening
seeds `72101,72102`. Conditions remain `n=55000`, validation size `5000`, 30
epochs, width 128, `m=20`, data seed `20261117`, the same bounded loss, and the
same validation checkpoint rule. Every method/regime receives 32 candidates
after combining the original and added trials.

Selection requires all 1024 complete training/validation results, both laws,
all eight method/regime labels and both screening seeds for every candidate.
The selector ranks mean validation accuracy, then validation cross entropy,
then the deterministic variant ID. Original candidates remain eligible and
stay selected when their validation scores win. Test fields are excluded from
the canonical selection records, and explicit test-result paths are rejected
before file access. The implementation never scans experiment directories for
test results or opens a dataset.

Use a new control root and preserve the current campaign. First freeze the
original complete MNIST training records. If the main coordinator is still
updating its training-results file, use a stable read-only copy of that file as
the input; the freezer rejects sources that change while being read.

```bash
conda run -n torch28 python -m journal.build_bayesian_mnist_scale \
  --stage freeze \
  --original-manifest ORIGINAL_SCREEN_MANIFEST \
  --original-results ORIGINAL_TRAINING_RESULTS_COPY \
  --output-dir CONTROL_ROOT/inputs

conda run -n torch28 python -m journal.build_bayesian_mnist_scale \
  --stage screen \
  --original-manifest CONTROL_ROOT/inputs/original_manifest.jsonl \
  --output CONTROL_ROOT/manifests/scale_screen.jsonl
```

The freezer accepts the full original all-dataset manifest and training-results
list, extracting only its exact 512-task MNIST subset. Other datasets need not
have finished. It writes a canonical original MNIST manifest, a validation-only
training-results snapshot, and source hashes into a separate owned directory.
All generated snapshots and manifests are immutable on retries.

The separate `run_bayesian_mnist_scale` coordinator runs the whole control in a
clean committed checkout. It verifies that seven training/campaign source
blobs equal the running main revision, defaults to `f602987`, and verifies its
own builder/runner/Slurm files against the execution commit. It captures the
original main manifest and training records into owned immutable files once;
resume validates those historical files and never refreshes selection inputs
from the advancing main campaign.

The allocation is 32 CPUs, sixteen workers, two threads per worker, and 16 GiB.
Before the added screen, sixteen full-shape one-epoch pilots cover every
method/law/regime with positive fixed Gaussian variance and seed `76200`.
The measured memory gate requires `2048 + 1.25*16*max_worker_RSS_MB` to fit the
allocated host memory. Pilot and screening use `--evaluate-top 0`; neither
produces an official-test results file. Approximate screen/final training
times are 55/45 minutes, respectively; the Slurm limit is two hours and fifteen
minutes and actual runtime remains an observed outcome.

```bash
export JOURNAL_GIT_SHA="$(git rev-parse HEAD)"
export MNIST_SCALE_MAIN_ROOT=ORIGINAL_MAIN_OUTPUT_ROOT
export MNIST_SCALE_OUTPUT=results/journal_bayesian_mnist_scale/20261117
sbatch journal/mnist_scale_control.slurm
```

The coordinator validates every completed stage, stops on failures, enforces
manifest hashes and committed source identity throughout execution, and
resumes complete matching stages without rerunning training or evaluation. It
runs the selected 80-case confirmation with twenty full-risk Gibbs draws, then
the 264-case `m` and 168-case `p_train` sweeps with ten full-risk draws each.
All outputs, status and logs stay under the separate control root. Plotting and
formal bounds are deferred to the final composite campaign. It never edits the
running main controller, status, manifests or results.

For manual operation, the same builder stages and campaign commands remain
available:

```bash
conda run -n torch28 python -m journal.campaign \
  --manifest CONTROL_ROOT/manifests/scale_screen.jsonl \
  --output-root CONTROL_ROOT/screen \
  --workers 16 --threads 2 --evaluate-top 0 --resume

conda run -n torch28 python -m journal.build_bayesian_mnist_scale \
  --stage select \
  --original-manifest CONTROL_ROOT/inputs/original_manifest.jsonl \
  --original-results CONTROL_ROOT/inputs/original_training_results.json \
  --extra-manifest CONTROL_ROOT/manifests/scale_screen.jsonl \
  --extra-results CONTROL_ROOT/screen/training_results.json \
  --output-dir CONTROL_ROOT/selected
```

The selected output contains the complete combined manifest, all 1024 canonical
validation records, sixteen selected method/regime/law variants, a hashed
selection protocol, and three final manifests:

| Manifest | Tasks | Fixed fresh seeds |
| --- | ---: | --- |
| `confirm.jsonl` | 80 | `73101–73105` |
| `m_sweep.jsonl` | 264 | `74101–74103` |
| `p_train_sweep.jsonl` | 168 | `75101–75103` |

These reuse the existing declared confirmation and sweep seed sets; none is
selected for favorable accuracy. The existing `selected_entries` builder also
retains the same dense channel grids and fixed optimizer settings along each
curve. The main comparison uses full MNIST training data; the mechanism sweeps
retain their separately declared `n=10000` budget.

Finalize this validation selection before reading official-test outcomes. The
eventual main result combines these 32-candidate validation-selected MNIST
configurations with the unchanged sixteen-candidate choices for the other
datasets. Preserve and label the original MNIST cohort as the preceding
comparison. The fixed-scale extension does not guarantee any method wins on
the official test set.

Verification without training or dataset access:

```bash
conda run -n torch28 python -m unittest \
  journal.test_bayesian_mnist_scale journal.test_run_bayesian_mnist_scale -v
```
