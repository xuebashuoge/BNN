# Journal simulation section package

This package contains the main simulation draft, supplementary results, reproducible tables/figures and an IEEE-style compiled preview. It uses the completed experiments; asset regeneration performs no training or checkpoint inference.

Read `RESULTS_AND_PRESENTATION.md` for the evidence inventory, recommended main/appendix placement and manuscript consistency notes. ERM and L2 are the main competing baselines; data-only PAC remains an internal ablation.

## Build

From `/Users/yangshuo/Git/BNN`:

```bash
conda run -n torch28 python journal/paper/generate_tables.py
conda run -n torch28 python -m journal.prepare_paper_figures
cd journal/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

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

`optional_bound_figures.tex` contains a certificate stack duplicating Table III and is omitted from the compact preview. `theory_example_replacement.tex` supplies the corrected analytical comparison for the theory section; use it only when replacing the earlier example and its existing `fig:bound_comparison` label.

## Result policy

Main entries use all listed seeds and the common ten-draw evaluation. Five-draw screens and selected 100-draw cases remain separate. Normalized bounded risk is not raw CE or classification error. Perfect-training models still use deployment-channel validation. L2 Perfect is unavailable, the L2 search is not exhaustive, and feature power is unnormalized. These limitations are preserved rather than filled with invented results.
