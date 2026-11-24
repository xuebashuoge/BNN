# Four-method unit-power fading comparison

T1, T2, ERM, and L2 used the archived Gaussian MLP on Two moons, MNIST, and Fashion-MNIST, and the archived split Gaussian CNN on CIFAR-10. The sweep used matched diagonal fading at unit power. It searched learning rates, regularization strengths, learning-rate decay, epoch budgets, channel banks of 16, 128, or 2000, and one to four training Monte Carlo draws per update. The initial search and refinement used 480 and 288 validation fits, respectively.

Each method's configuration and the shared bank size were selected on validation. The primary comparison refit the four selected configurations on five fresh paired training seeds and measured held-out accuracy with 64 deployment draws per fit. The exploratory comparison selected one paired training seed and one fit per method on validation, then used 128 held-out deployment draws. The exploratory gaps describe the selected fits.

| Dataset | Bank size | Five fresh seeds (pp) | Validation-selected seed (pp) |
|---|---:|---:|---:|
| Two moons | 128 | T1 −0.125 | T1 +0.258 |
| MNIST | 2000 | T2 −0.461 | T2 −0.028 |
| Fashion-MNIST | 2000 | T1 −0.044 | T2 +0.078 |
| CIFAR-10 | 128 | T1 −0.363 | T1 +1.035 |

Each gap subtracts the stronger of ERM and L2 on the test set from the proposed method selected on validation. The [primary table](performance_table.tex) shows the mean and sample standard deviation for all four methods. The [primary summary](performance_summary.json) contains both baseline comparisons and paired bootstrap intervals. The [exploratory summary](seed_sweep/seed_sweep_summary.json) identifies the selected seeds and checkpoint fingerprints.

Spartan jobs `35608074` and `35608075` assembled the final primary and exploratory summaries. The [primary ledger](jobs.tsv) and [exploratory ledger](seed_sweep/jobs.tsv) record the search, refinement, and confirmation stages, including the MNIST retry. The portable trial bundles contain the held-out evaluation records and checkpoint metadata. The 96 verified checkpoint files are in the local models directory at `../../models/fading_unit_power/performance_sweep` and in the Spartan campaign records.

From the repository root, rebuild both reports from the saved evaluations.

```bash
study="$PWD/results/final_chosen_results/fading_unit_power/performance_sweep"
replay=$(mktemp -d)
for phase in primary seed_sweep; do
    bundle="$study"
    if [[ "$phase" == seed_sweep ]]; then bundle="$study/seed_sweep"; fi
    mkdir "$replay/$phase"
    (cd "$bundle" && shasum -a 256 -c selected_trials.sha256)
    tar -xzf "$bundle/selected_trials.tar.gz" -C "$replay/$phase"
    cp -R "$bundle/manifests" "$replay/$phase/manifests"
done
conda run -n torch28 python -m fading_unit_power.performance_sweep summarize \
    --output "$replay/primary" --recorded-checkpoints "$study/selected_checkpoints.jsonl"
conda run -n torch28 python -m fading_unit_power.performance_seed_sweep summarize \
    --output "$replay/seed_sweep"
```
