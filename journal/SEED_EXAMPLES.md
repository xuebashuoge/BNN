# Exploratory paired-seed examples

This optional analysis uses the user's explicit seed-search authorization to
identify a favorable **existing paired seed** for each of the eight final
dataset/channel conditions. It supplements the complete five-seed main result;
it never replaces that result or changes its mean rankings. Every output is
labelled **test-selected exploratory paired-seed example**. A selected example
is not fresh-seed confirmation or evidence of general improvement.

The protocol is declared in `verify_bayesian_seed_examples.py` before opening
the final assembled test scores:

1. Require the completed final assembled confirmation cohort: four datasets,
   two deployment laws, eight matched/perfect method labels, and all five
   prescribed paired seeds `73101–73105`—320 checkpoints in total.
2. For each setting and each existing seed, compare its best matched T1/T2
   accuracy with the strongest of the other seven methods using the saved
   twenty-draw scores. Select the seed with the largest strictly positive
   margin, using seed number as a deterministic tiebreak. The selected seed is
   shared by **all eight methods**. A tie is not a strict win. If none wins,
   report no example for that setting; do not train another seed.
3. Freeze the selected seeds, every initial five-seed score, all 64 main mean
   rows and the original mean rankings, all 320 checkpoint identities, and
   source fingerprints before running any new Monte Carlo evaluation.
4. Evaluate all eight selected checkpoints using exactly **500** fresh
   deployment-only Gibbs draws, CPU execution with two threads, and
   `eval_seed = training_eval_seed + 9000019`. These weight/channel streams are
   checked against the validation and original twenty-draw streams. No
   optimizer is created and no checkpoint is changed.
5. Report every fresh score and all saved draw observations. Retain the
   initial chosen T1/T2 candidate and report its fresh margin, as well as the
   fresh descriptive ranking. Losses, ties and winner changes remain visible.
   A measured loss never triggers another Monte Carlo attempt or seed search.

Exactly 500 draws and the fixed seed offset are constants rather than tuning
options. A failed execution can resume the same frozen request and stream;
valid cached results are reused exactly, including losses. Changed requests,
corrupt results or changed source fingerprints are rejected. A preexisting MC
result without its prior frozen request is not adopted. No stopping rule based
on the observed winner is used.

Every resume rebuilds the complete selection protocol from the attested source
cohort and compares all 320 task identities/configurations, selected cases,
fresh-seed formulas, original means/rankings and fingerprints. Editing a cached
selected seed, candidate, task ID or stream is rejected before any new MC
request. The source assembly policy, 2048-case audit, confirmation manifest
hash, metadata configurations/revisions and byte-exact exported reports are
checked against the physical source map.

The JSON report includes the mean, sample SD and descriptive standard error of
the fresh accuracy/loss replicates, all 500 accuracy draws per method, and
paired accuracy-difference SD/SE for the fixed initial candidate and the fresh
best matched candidate against every other method. These summarize Monte Carlo
variation conditional on fixed checkpoints and the fixed test set. They omit
test-selection and training-seed uncertainty and are not confirmatory intervals
or population-risk certificates. Full per-draw accuracy, bounded-loss, raw-CE
and feature-power observations remain in each saved `evaluation.json`.

Run only after the final assembler reports completion. The source root keeps
its original manifests, model aliases, numerical reports, five-seed tables and
rankings unchanged. Before creating any output directory or lock, the driver
checks that the output is separate from the assembled root, original
main/control roots and every physical source stage—including the permanently
published external `p_train` cache. Task IDs must be safe ASCII path components;
writable cache paths must remain inside the owned output and cannot contain
symbolic links, including broken links or aliases of request/temporary/log
files.

```bash
# Optional: freeze/review selection before launching fresh MC.
conda run -n torch28 python -m journal.verify_bayesian_seed_examples \
  --source-root FINAL_ASSEMBLED_ROOT \
  --output results/journal_bayesian_seed_examples/20261117 \
  --select-only

# In a clean committed server checkout, after assembly completes:
export JOURNAL_GIT_SHA="$(git rev-parse HEAD)"
export SEED_EXAMPLES_SOURCE_ROOT=FINAL_ASSEMBLED_ROOT
export SEED_EXAMPLES_OUTPUT=results/journal_bayesian_seed_examples/20261117
sbatch journal/seed_examples.slurm
```

The launcher requests 32 CPUs, sixteen concurrent two-thread workers, 32 GiB,
and 45 minutes. The driver verifies seven NN/evaluation/campaign source blobs
against main revision `f602987`, checks its own committed driver, and checks the
submitted execution revision. It uses the existing checkpoint-only
`run_experiment --mode evaluate --risk-scope deployment` implementation.
Formal main-cohort bounds and all five-seed main figures stay separate from
this exploratory analysis. No additional plots or confidence shading are
produced by this helper; the saved scores and draw observations support a
clearly labelled exploratory display later if needed.

Outputs include `selection_protocol.json` frozen before new MC,
`verification_status.json`, one owned request/override/log/evaluation directory
per selected checkpoint, and `exploratory_seed_examples.json`. A setting with
no initial winning seed remains in the report with an explicit not-run status.
This document describes the prepared analysis; no final assembled scores have
been inspected and no verification job has been launched during preparation.

```bash
conda run -n torch28 python -m unittest journal.test_verify_bayesian_seed_examples -v
bash -n journal/seed_examples.slurm
```
