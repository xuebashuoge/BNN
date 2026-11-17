# Journal simulation section package

This package contains the main simulation draft, supplementary results, reproducible tables/figures and an IEEE-style compiled preview. It uses the completed experiments; asset regeneration performs no training or checkpoint inference.

Read `RESULTS_AND_PRESENTATION.md` for the evidence inventory, recommended main/appendix placement and manuscript consistency notes. ERM and L2 are the main competing baselines; data-only PAC remains an internal ablation.

## Build

From `/Users/yangshuo/Git/BNN`:

```bash
conda run -n torch28 python journal/paper/generate_tables.py
conda run -n torch28 python -m journal.prepare_paper_figures
# After BOTH CIFAR-10 screening and full-data confirmation are complete:
conda run -n torch28 python journal/paper/generate_cifar_assets.py
conda run -n torch28 python journal/paper/generate_risk_bound_table.py
cd journal/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The CIFAR wrapper validates both saved stages before writing publication files.
It generates the main `tables/cifar_comparison.tex` and supplementary
`tables/appendix_cifar_screen.tex`, their CSVs, an optional confirmation-validation
plot and `cifar_data_manifest.json` with repo-relative hashed source paths.
It performs reporting only; an incomplete or missing confirmation is an error.

The consolidated `tables/algorithm_risk_bounds` table replaces the two older
accuracy/risk tables in the main fragment. It retains their saved test metrics
and adds frozen-checkpoint training-bank risk and both unweighted T1/T2 risk
bounds. `generate_risk_bound_table.py` reads JSON only. Its stable 51-entry
`algorithm_risk_bounds_queue.jsonl` identifies the selected checkpoints;
`algorithm_risk_bounds_pending.jsonl` identifies any missing empirical reports.
The underlying inference is a separate operation:

```bash
conda run -n torch28 python -m journal.evaluate_empirical_bank \
  --queue journal/paper/tables/algorithm_risk_bounds_queue.jsonl \
  --index 0 --draws 10 --threads 2
```

Repeat the zero-based index for the declared queue, using the same complete
queue for its confidence allocation. `journal/empirical_queue.slurm` runs the
whole inference queue on eight CPUs with four two-thread workers. It never
trains a model. Reports go to each checkpoint's `empirical_bank/` directory.
The data location is supplied exclusively by `DATASET`.

`c1_evaluation.tex` compares two deployment laws using the same three saved
perfect-trained toy posteriors. Hidden-feature erasure uses C1's first
inequality with the discrete metric. Fully specified centered-lognormal noise
on output logits has a Euclidean-distance expectation without a simple known
closed form. Its second-inequality evaluation uses a moment-matched Gaussian
reference, giving closed-form expressions for both the exponential moment and
the exact channel KL. A separate numerical distance estimate is diagnostic;
it is not used in the analytic guarantee. The output loss has a uniform,
weight-independent Lipschitz constant. The changed deployment
split is explicit; this is not a certificate for the earlier hidden-feature
Gaussian channel. Regenerating this separate figure/table performs frozen
checkpoint inference, not training:

```bash
conda run -n torch28 python -m journal.evaluate_c1_reference
```

Its source hashes, loss/metric definitions, confidence allocation, and per-seed
records are in `assets/c1_evaluation.json`.

The reviewed final preview is `output/pdf/journal_simulation_draft.pdf`. Disposable logs and rendered QA pages are under ignored `build/`. The preview's section counter and theorem-label bridges are only for standalone display; do not copy those bridges into the full manuscript.

## Integrate

The main fragment requires `amsmath`, `amssymb`, `graphicx`, `booktabs`, `array` and `multirow`. It provides non-overriding defaults for `\E` and `\TV`. Define `\simroot` in the preamble, then put the `\input` command at the old Simulation section position:

```latex
\newcommand{\simroot}{journal/paper}
\input{\simroot/simulation_section}
```

The fragment has its own heading and `sec:simulation` label. All table/image paths use `\simroot`. Main theorem references use the supplied manuscript's labels `thm:separate_bound_US` and `thm:joint_bound_US`.

Optional material inside the existing appendix structure:

```latex
\input{\simroot/supplementary_results}
```

For the draft's manual bibliography, insert `bibliography_entries.tex` before `\end{thebibliography}`. Alternatively merge `references.bib` with the BibTeX database; do not duplicate citation keys.

`optional_bound_figures.tex` contains a certificate stack duplicating the numerical-certificate table and is omitted from the compact preview. `theory_example_replacement.tex` supplies the corrected analytical comparison for the theory section; use it only when replacing the earlier example and its existing `fig:bound_comparison` label.

## Result policy

Main entries use all listed seeds and the common ten-draw evaluation. Five-draw screens and selected 100-draw cases remain separate. Normalized bounded risk is not raw CE or classification error. Perfect-training models still use deployment-channel validation. L2 Perfect is available for CIFAR and absent from the older MNIST/FashionMNIST cohorts. The L2 search is not exhaustive, and feature power is unnormalized. These limitations are preserved rather than filled with invented results.

The image-table risk and bound columns all use the same normalized smooth
loss. Accuracy is reported separately. All image-model stochastic bounds are
vacuous; their uncapped values remain visible, and the universal bound is one.
Deterministic ERM/L2 models have infinite Dirac-to-Gaussian KL, so these theorem
bound cells are unavailable. The table does not replace their empirical risks
with zero or add Gaussian weight noise to change their evaluated predictors.
