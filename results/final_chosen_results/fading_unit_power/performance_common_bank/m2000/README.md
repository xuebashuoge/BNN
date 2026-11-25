# Deployment accuracy at a common channel-bank size

The [comparison](common_bank_comparison.md) uses $m=2000$ for Two moons, MNIST, Fashion-MNIST, and CIFAR-10 under both matched channels. The four-method accuracies and gaps are also available as [CSV](common_bank_comparison.csv) and [JSON](common_bank_comparison.json). The architectures match the archived Gaussian MLPs and split CIFAR-10 CNNs.

The unit-power fading rows use matched diagonal fading at correlation 0.35. Each row reports 128 held-out deployment draws for one seed selected on validation. The erasure rows use the original matched erasure probability of 0.50 and report the mean over five, five, nine, and five paired training seeds, respectively, with 20 held-out draws per fit. The gap is in percentage points against the stronger of ERM and L2.

| Fading dataset | Source at $m=2000$ |
|---|---|
| Two moons | Four shrinkage or posterior-scale profiles per method on sixteen paired seeds. A separate 12,000-example IID validation sample selected one shared seed and one checkpoint per method. |
| MNIST | [Verified regularization follow-up](../../performance_sweep/regularization/mnist/seed_sweep/seed_sweep_summary-mnist.json). |
| Fashion-MNIST | [Verified initial seed comparison](../../performance_sweep/seed_sweep/seed_sweep_summary.json). |
| CIFAR-10 | Existing initial-search checkpoints chosen from the $m=2000$ validation records, then evaluated using 128 held-out draws. |

T1 or T2 leads both baselines in all eight rows. On Two moons fading, the validation-selected T1 fit leads the stronger baseline by 0.121 percentage points on the standard test and 0.078 points on the [independent 30,000-example IID audit](fading/moons/audit/seed_sweep_summary-moons.json). Both tests used the same four checkpoints. In the [five-fresh-seed check](fading/moons/fresh/performance_summary-moons.json), validation-selected T2 trails ERM by 0.005 points on average.

The earlier [regularization](provenance/moons-regularization/seed_sweep_summary-moons.json) and [schedule](provenance/moons-schedule/seed_sweep_summary-moons.json) panels trailed ERM by 0.456 and 0.247 points on their standard held-out tests. All three panels kept $m=2000$.

The Two moons and CIFAR-10 seed audits include selection manifests, Slurm ledgers, and checksummed trial records under `fading/`. The local model directory `../../../models/fading_unit_power/performance_common_bank/m2000` contains the selected Two moons weights under `moons-strength` and the selected CIFAR-10 weights under `cifar10`. The MNIST and Fashion-MNIST weights are in the [original sweep's local models directory](../../../models/fading_unit_power/performance_sweep).

Rebuild the common-bank table from the repository root with the verified fading selections and original erasure archive.

```bash
study="$PWD/results/final_chosen_results/fading_unit_power/performance_common_bank/m2000"
replay=$(mktemp -d)
conda run -n torch28 python -m fading_unit_power.performance_common_bank_report \
    --bank-size 2000 --fading-root "$study/fading" --output "$replay"
cmp "$study/common_bank_comparison.json" "$replay/common_bank_comparison.json"
```

To replay the independent Two moons audit, extract its archived evaluations and frozen selection into a temporary directory.

```bash
audit="$study/fading/moons/audit"
audit_replay=$(mktemp -d)
(cd "$audit" && shasum -a 256 -c selected_trials.sha256)
tar -xzf "$audit/selected_trials.tar.gz" -C "$audit_replay"
cp -R "$audit/manifests" "$audit_replay/manifests"
conda run -n torch28 python -m fading_unit_power.performance_seed_sweep summarize \
    --dataset moons --output "$audit_replay"
cmp "$audit/seed_sweep_summary-moons.json" "$audit_replay/seed_sweep_summary-moons.json"
```
