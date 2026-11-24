# Deployment accuracy at a common channel-bank size

The [comparison](common_bank_comparison.md) uses $m=2000$ for Two moons, MNIST, Fashion-MNIST, and CIFAR-10 under both matched channels. The four-method accuracies and gaps are also available as [CSV](common_bank_comparison.csv) and [JSON](common_bank_comparison.json). The architectures match the archived Gaussian MLPs and split CIFAR-10 CNNs.

The unit-power fading rows use matched diagonal fading at correlation 0.35. Each row reports 128 held-out deployment draws for one seed selected on validation. The erasure rows use the original matched erasure probability of 0.50 and report the mean over five, five, nine, and five paired training seeds, respectively, with 20 held-out draws per fit. The gap is in percentage points against the stronger of ERM and L2.

| Fading dataset | Source at $m=2000$ |
|---|---|
| Two moons | Fixed-bank regularization on six paired validation seeds, followed by three training schedules per method on twelve fresh paired seeds. The reported seed comes from the run with the larger validation gap. |
| MNIST | [Verified regularization follow-up](../../performance_sweep/regularization/mnist/seed_sweep/seed_sweep_summary-mnist.json). |
| Fashion-MNIST | [Verified initial seed comparison](../../performance_sweep/seed_sweep/seed_sweep_summary.json). |
| CIFAR-10 | Existing initial-search checkpoints chosen from the $m=2000$ validation records, then evaluated using 128 held-out draws. |

Two moons fading is the only row where the stronger baseline leads. The schedule follow-up narrowed the held-out deficit from 0.456 to 0.247 percentage points. Across five fresh paired seeds in the follow-up, T2 trailed by 0.157 points. Both the [first fixed-bank seed audit](provenance/moons-regularization/seed_sweep_summary-moons.json) and the [fresh-seed schedule report](provenance/moons-schedule-fresh_seed_summary.json) are archived here.

The new Two moons and CIFAR-10 seed audits include the selection manifests, Slurm ledgers, and checksummed trial records under `fading/`. The corresponding checkpoint weights are in the local models directory at `../../../models/fading_unit_power/performance_common_bank/m2000`. The reused MNIST and Fashion-MNIST weights remain in the [original sweep's local models directory](../../../models/fading_unit_power/performance_sweep).

Rebuild the common-bank table from the repository root with the verified fading selections and original erasure archive.

```bash
study="$PWD/results/final_chosen_results/fading_unit_power/performance_common_bank/m2000"
replay=$(mktemp -d)
conda run -n torch28 python -m fading_unit_power.performance_common_bank_report \
    --bank-size 2000 --fading-root "$study/fading" --output "$replay"
cmp "$study/common_bank_comparison.json" "$replay/common_bank_comparison.json"
```
