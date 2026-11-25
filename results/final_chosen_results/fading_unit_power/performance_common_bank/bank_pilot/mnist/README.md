# MNIST unit-power fading bank-size validation pilot

The pilot compares $m=128,144,160,176,192,200$ with an $m=2000$ control. Each of the four methods uses its archived validation-winning profiles from $m=128$ and $m=2000$ on the same two fresh paired training seeds. The transferred T1 and T2 coefficients preserve each active penalty term's initial gradient on the mean weights. The architecture, data split, schedules, and deployment channel follow the source profiles.

All 112 fits used 24 deployment draws on validation. At each bank size, validation mean accuracy selects a profile for each method. The reported gap subtracts the stronger ERM or L2 mean from the stronger T1 or T2 mean. The [CSV](bank_pilot_mnist.csv) includes both paired seed gaps against the stronger baseline at that seed.

All smaller banks gave a larger MNIST validation deficit than the matched $m=2000$ control. The best smaller-bank mean gap was −2.141 percentage points at $m=200$, versus −0.400 at $m=2000$. The strongest proposed mean validation accuracy was 69.586% at $m=200$ and 77.468% at $m=2000$. As a validation screen with two profiles and two paired seeds, this experiment supplies evidence about the tested configurations. The [shared-bank held-out comparison](../../m2000/common_bank_comparison.md) uses $m=2000$ for both channels and all four datasets.

Spartan jobs `35678385` and `35678386` ran the validation fits and wrote the summary. The checksummed trial archive contains the original validation records. Replay both summaries from the repository root.

```bash
pilot="$PWD/results/final_chosen_results/fading_unit_power/performance_common_bank/bank_pilot/mnist"
replay=$(mktemp -d)
(cd "$pilot" && shasum -a 256 -c validation_trials.sha256)
tar -xzf "$pilot/validation_trials.tar.gz" -C "$replay"
cp -R "$pilot/manifests" "$replay/manifests"
conda run -n torch28 python -m fading_unit_power.performance_bank_pilot summarize \
    --output "$replay" --dataset mnist
cmp "$pilot/bank_pilot_mnist.json" "$replay/bank_pilot_mnist.json"
cmp "$pilot/bank_pilot_mnist.csv" "$replay/bank_pilot_mnist.csv"
```
