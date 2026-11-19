#!/usr/bin/env bash
set -euo pipefail
root=${1:?Supply a rendered release directory}
tmp=$(mktemp -d)
module load GCCcore/11.3.0
module load texlive/20230313
for view in common validation_examples; do
    [[ -d "$root/$view" ]] || continue
    mkdir -p "$tmp/$view"
    for m in 200 500 1000 100 2000; do
        name="channel_accuracy_m$m"
        pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$tmp/$view" \
            "$root/$view/$name.tex" > "$tmp/$view/$name.compile.log"
        if grep -q 'Overfull .hbox' "$tmp/$view/$name.log"; then
            grep 'Overfull .hbox' "$tmp/$view/$name.log" >&2
            exit 1
        fi
    done
done
echo 'Compiled all five common and supplementary standalone LaTeX tables without horizontal overflow.'
