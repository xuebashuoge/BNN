# Selected fading studies with unit transmit power

These results use one unit of transmit power per real channel symbol. The [campaign instructions](../../../fading_unit_power/README.md) describe the normalization and the Spartan launcher.

| Output | Reproduction records |
|---|---|
| [MNIST generalization and deployment accuracy vs. $m$](mnist_analog_finite_channel_generalization.pdf) | [54 observations](data/mnist_bank/observations.csv) at 18 bank sizes, including measured transmit power and checkpoint SHA-256. |
| [Deployment accuracy at $m=2000$](performance/performance_table.tex) | [Paired results](performance/deployment_by_seed.csv), [method gaps](performance/performance_summary.json), [selected trial reports](performance/selected_trials.tar.gz) and [validation manifests](performance/provenance). The table's erasure rows come from the original archive. |
| [Four-method tuning sweep](performance_sweep/README.md) | [Fresh-seed comparison](performance_sweep/performance_table.tex), [paired results](performance_sweep/deployment_by_seed.csv) and [validation-selected seed comparison](performance_sweep/seed_sweep/seed_sweep_summary.json). |

In the MNIST sweep, the mean bounded-loss gap decreases from 0.40108 at $m=2$ to 0.01240 at $m=25600$. Mean deployment accuracy increases from 36.81% to 73.94% across those endpoints.

The [two-moons bound decomposition](../bound_decomposition_d8_t1_t2_both_channels.pdf) uses the original feature norm cap $\lVert z\rVert_2 \leq 1$. Its [reports](../data/bounds) and [plotting script](../plot_bound_decomposition_d8.py) are in the parent results directory.

The MNIST sweep ran on Spartan at Git revision `987c1d3` under array `35553567`. Job `35553569` rendered the figure. Full MNIST records and checkpoints are in `/data/gpfs/projects/punim2792/yangshuo/unit-power-fading-20261123-r2/fixed/mnist`. Local checkpoint copies are under `../models/fading_unit_power/mnist`.

The deployment comparison used 300 validation fits across the initial and refined searches. Each dataset's five validation-selected configurations were evaluated on three fresh paired training seeds. L2 had the highest mean held-out deployment accuracy on all four fading datasets. The mean gap from the better of T1 and T2 to L2 was −0.039 percentage points on Two moons, −0.282 on MNIST, −0.044 on Fashion-MNIST and −0.447 on CIFAR-10.

Spartan job `35555080` assembled the table and checked the 60 fading confirmation checkpoints. The full selected records and weights are in `/data/gpfs/projects/punim2792/yangshuo/unit-power-fading-20261123-published`. The report archive contains the configurations, checkpoint metadata, and test evaluations. Each normalized row's `archive_key` identifies its directory in the reports archive and under `../models/fading_unit_power/performance`.

From the repository root, rebuild the table and paired summaries from the archived evaluations.

```bash
selected=results/final_chosen_results/fading_unit_power/performance
replay=$(mktemp -d)
tar -xzf "$selected/selected_trials.tar.gz" -C "$replay"
mkdir "$replay/manifests"
for dataset in moons mnist fashionmnist cifar10; do
    cp "$selected/provenance/$dataset/refined/confirmation.jsonl" \
        "$replay/manifests/$dataset-confirmation.jsonl"
done
conda run -n torch28 python -m fading_unit_power.performance summarize --output "$replay"
```
