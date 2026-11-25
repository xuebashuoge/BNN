# MNIST fading margin search at m=200 and m=2000

The search compares T1 and T2 separately against ERM and L2. Each method received six profiles at each bank on the same three fresh paired training seeds. The profiles include the validation winner from the [earlier bank pilot](../../bank_pilot/mnist/README.md) and five common schedules with different learning rates, decay, posterior settings and regularization strengths. The T1 and T2 coefficients were calibrated at each actual bank size. The runs preserved the model architecture, data split, and unit-power fading channel.

Each of the 144 fits used 40 deployment draws on validation. Mean validation accuracy selects one profile for each method at each bank. The following gaps are in percentage points.

| $m$ | T1−ERM | T1−L2 | T2−ERM | T2−L2 |
|---:|---:|---:|---:|---:|
| 200 | −0.049 | −0.089 | −0.061 | −0.101 |
| 2000 | −0.040 | −0.108 | +0.134 | +0.066 |

The wider $m=200$ search brought the proposed-method deficit against the strongest baseline near parity. Its best validation-selected paired seed reached +0.039 points against both baselines with the stronger proposed method. At the same search budget, the $m=2000$ control reached +0.089 on its best paired seed and had higher absolute validation accuracy. The detailed [CSV](bank_margin_mnist.csv) and [JSON](bank_margin_mnist.json) include per-seed and mean-selected margins.

The [held-out pairwise CSV](m2000_heldout_pairwise.csv) subtracts each baseline from each proposed method in the [shared $m=2000$ held-out comparison](../../m2000/common_bank_comparison.json). CIFAR-10 fading T1 is the only method and cohort in that table with at least a +0.5-point lead against both baselines.

Spartan jobs `35681818` and `35681832` ran the validation fits and wrote the summary. Replay the validation analysis from the repository root.

```bash
search="$PWD/results/final_chosen_results/fading_unit_power/performance_common_bank/bank_margin/mnist"
replay=$(mktemp -d)
(cd "$search" && shasum -a 256 -c validation_trials.sha256)
tar -xzf "$search/validation_trials.tar.gz" -C "$replay"
cp -R "$search/manifests" "$replay/manifests"
conda run -n torch28 python -m fading_unit_power.performance_bank_pilot summarize \
    --output "$replay" --dataset mnist --margin-search
cmp "$search/bank_margin_mnist.json" "$replay/bank_margin_mnist.json"
cmp "$search/bank_margin_mnist.csv" "$replay/bank_margin_mnist.csv"
```
