# Four-method unit-power fading comparison

T1, T2, ERM, and L2 used the archived Gaussian MLP on Two moons, MNIST, and Fashion-MNIST, and the archived split Gaussian CNN on CIFAR-10. The sweep used matched diagonal fading at unit power. It searched learning rates, regularization strengths, learning-rate decay, epoch budgets, channel banks of 16, 128, or 2000, and one to four training Monte Carlo draws per update. The initial search and refinement used 480 and 288 validation fits, respectively.

## Selected training seeds

For each dataset, the selected fit set comes from the campaign with the larger proposed-to-baseline validation accuracy gap. Both campaigns used the same validation and test splits. The table reports held-out accuracy (%) from 128 deployment draws for the selected shared training seed and the gap to the stronger test baseline.

| Dataset | Campaign | Bank size | Training seed | ERM | L2 | T1 | T2 | Selected gap (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Two moons | Initial | 128 | 24112413 | 69.440 | 69.544 | **69.801** | 69.703 | T1 +0.258 |
| MNIST | Regularization | 2000 | 24112433 | 78.243 | 78.455 | 78.622 | **78.655** | T2 +0.200 |
| Fashion-MNIST | Initial | 2000 | 24112413 | 70.752 | 70.925 | 71.003 | **71.004** | T2 +0.078 |
| CIFAR-10 | Initial | 128 | 24112413 | 48.680 | 49.428 | **50.462** | 49.572 | T1 +1.035 |

The selected gaps describe one set of four trained checkpoints per dataset. The fresh-seed tables report fit-to-fit variation across five paired seeds.

## Initial sweep

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

## Regularization follow-up

Each dataset's follow-up kept the selected channel bank, architectures, and training schedules. Each method tried six regularization or posterior-scale profiles on three paired validation seeds. T1 and T2 included profiles with their initial mean-weight shrinkage calibrated against L2. The selected profiles were refit on five new paired seeds. The follow-up seed comparison selected a shared training seed and four checkpoints on validation before measuring accuracy on the same held-out test split.

| Dataset | Five fresh seeds (pp) | Validation-selected seed (pp) |
|---|---:|---:|
| [Two moons](regularization/moons/performance_summary-moons.json) | T1 −0.006 | T1 +0.107 |
| [MNIST](regularization/mnist/performance_summary-mnist.json) | T2 −0.133 | T2 +0.200 |
| [Fashion-MNIST](regularization/fashionmnist/performance_summary-fashionmnist.json) | T1 −0.254 | T1 −0.029 |
| [CIFAR-10](regularization/cifar10/performance_summary-cifar10.json) | T2 −1.929 | T2 +0.012 |

The stronger baseline led on every dataset across five fresh paired seeds in both campaigns. The validation-selected seed results at the top of this report show the four checkpoint-level wins.

The per-dataset archives under [regularization](regularization) include 72 validation fits, 20 fresh-seed test evaluations, four selected-seed test evaluations, and checkpoint fingerprints. The local model copies are under `../../models/fading_unit_power/performance_sweep/regularization`.

Rebuild a follow-up dataset's selection and test summaries from the archive. The commands use the `study` and `replay` variables above.

```bash
dataset=mnist
followup="$study/regularization/$dataset"
for phase in paired seed_sweep; do
    bundle="$followup"
    if [[ "$phase" == seed_sweep ]]; then bundle="$followup/seed_sweep"; fi
    mkdir "$replay/$dataset-$phase"
    (cd "$bundle" && shasum -a 256 -c trials.sha256)
    tar -xzf "$bundle/trials.tar.gz" -C "$replay/$dataset-$phase"
    cp -R "$bundle/manifests" "$replay/$dataset-$phase/manifests"
done
conda run -n torch28 python -m fading_unit_power.performance_regularization_sweep select \
    --dataset "$dataset" --output "$replay/$dataset-paired"
conda run -n torch28 python -m fading_unit_power.performance_sweep summarize \
    --dataset "$dataset" --output "$replay/$dataset-paired" \
    --recorded-checkpoints "$followup/selected_checkpoints-$dataset.jsonl"
conda run -n torch28 python -m fading_unit_power.performance_seed_sweep summarize \
    --dataset "$dataset" --output "$replay/$dataset-seed_sweep"
```
