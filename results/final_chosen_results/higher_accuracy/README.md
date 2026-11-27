# Deployment accuracy at smaller shared m (T1 with the fresh Q_{W|S})

> **Manuscript use.** The paper's main deployment table stays at m = 2000
> ([`../performance_table.tex`](../performance_table.tex), from `../data_dependent/only_fresh`). The
> runs here at m = 20, 200 and 1000 feed the bank-size table
> ([`../bank_size_table.tex`](../bank_size_table.tex)), which shows the lead growing as m shrinks.

Goal: in every row of the deployment table, the better of T1 and T2 should beat the better of ERM and
L2, by as wide a margin as possible. One channel-bank size m is shared by all eight rows and all
methods. T1 keeps the `fresh` data-dependent prior from [`../data_dependent/only_fresh`](../data_dependent/only_fresh).

**Result: T1 is the best method in all 8 rows at m = 20.** Seven rows win by 4.3–16.6 points. Moons
erasure is a tie within seed noise, at +0.04.

Table: [`performance/performance_table.tex`](performance/performance_table.tex). Per-seed accuracy:
[`performance/deployment_by_seed.csv`](performance/deployment_by_seed.csv). Gaps and selected variants:
[`performance/comparison.json`](performance/comparison.json).

## Test accuracy at m = 20 (%, mean over the archived paired seeds)

| row | ERM | L2 | T2 | T1 | gap to max(ERM, L2) | gap at m = 2000 | seeds T1 wins |
|---|---|---|---|---|---|---|---|
| Moons fading | 66.56 | 66.79 | 66.81 | **71.07** | **+4.27** | +0.17 | 3/3 |
| Moons erasure | 86.37 | 86.34 | 86.33 | **86.41** | +0.04 | +0.28 | 3/5 |
| MNIST fading | 63.40 | 64.91 | 61.28 | **78.08** | **+13.17** | +0.63 | 3/3 |
| MNIST erasure | 91.26 | 91.21 | 89.56 | **96.36** | **+5.10** | +0.13 | 5/5 |
| F-MNIST fading | 60.25 | 59.44 | 58.43 | **70.86** | **+10.61** | +0.80 | 3/3 |
| F-MNIST erasure | 79.66 | 79.67 | 80.91 | **86.42** | **+6.75** | +0.45 | 9/9 |
| CIFAR-10 fading | 34.94 | 40.48 | 35.15 | **57.12** | **+16.64** | +2.62 | 3/3 |
| CIFAR-10 erasure | 56.55 | 57.51 | 56.50 | **65.21** | **+7.70** | +0.94 | 5/5 |

The m = 2000 column is max(T1, T2) − max(ERM, L2) from `only_fresh`. "Seeds T1 wins" counts paired
seeds where T1 beats both ERM and L2 trained on the same seed. The perfect-training column of the
table does not depend on m and is copied from the archived table.

## Other shared bank sizes: m = 200 and m = 1000

Separate campaigns (`--m 200`, `--m 1000`) use the same protocol, with a screen at that m only. Both
Moons rows get the six-seed follow-up with the wider grids. Tables:
[`m200/performance/performance_table.tex`](m200/performance/performance_table.tex) and
[`m1000/performance/performance_table.tex`](m1000/performance/performance_table.tex).

Test lead of max(T1, T2) over max(ERM, L2), in points, with paired seeds won in parentheses:

| row | m = 20 | m = 200 | m = 1000 | m = 2000 (`only_fresh`) |
|---|---|---|---|---|
| Moons fading | **+4.27** (3/3) | +0.74 (3/3) | −0.09 (1/3) | +0.17 |
| Moons erasure | +0.04 (3/5) | −0.07 (2/5) | +0.02 (3/5), T2 | +0.28 |
| MNIST fading | **+13.17** (3/3) | +6.09 (3/3) | +1.29 (3/3) | +0.63 |
| MNIST erasure | **+5.10** (5/5) | +0.36 (5/5) | +0.14 (5/5) | +0.13 |
| F-MNIST fading | **+10.61** (3/3) | +4.48 (3/3) | +1.36 (3/3) | +0.80 |
| F-MNIST erasure | **+6.75** (9/9) | +0.33 (9/9) | +0.52 (9/9) | +0.45 |
| CIFAR-10 fading | **+16.64** (3/3) | +6.85 (3/3) | +3.70 (3/3) | +2.62 |
| CIFAR-10 erasure | **+7.70** (5/5) | +0.89 (4/5) | +1.23 (4/5) | +0.94 |
| rows won | 8/8 | 7/8 | 7/8 | 8/8 |
| mean lead | **+8.04** | +2.46 | +1.02 | +0.75 |

T1 is the better proposed method everywhere except Moons erasure at m = 1000, where T2 edges T1 by
0.003 points. T1's accuracy hardly changes with m: MNIST fading is 78.1% at all three sizes. The
lead grows as m shrinks because ERM, L2 and T2 lose accuracy. Moons erasure is a tie at every m.

Spartan: m = 200 ran at revision 4681a13 (job 35841053) in
`/data/gpfs/projects/punim2792/yangshuo/higher-accuracy-m200-20261127`; m = 1000 ran at 4a8ff2a
(job 35842542) in `/data/gpfs/projects/punim2792/yangshuo/higher-accuracy-m1000-20261127`. Fresh
stage-1 fits are linked from the data-dependent and m = 20 campaigns. The selected candidates are in
each folder's `performance/selection.json`.

## Why a small m

The fresh stage-1 mean μ_S is trained on S with a new P_tr draw for every minibatch, so it never
depends on the bank U or on m. Stage 2's channel term scales as 1/√(m−1), so a small bank keeps T1
near μ_S, which generalizes over the channel law. ERM and L2 see only the m channels in U and overfit
them as m shrinks. With the fixed prior Q_W, smaller m did not help T1
([earlier MNIST margin search](../fading_unit_power/performance_common_bank/bank_margin/mnist/README.md)).

Validation gaps from the screen, max(T1, T2) − max(ERM, L2), in points:

| m | worst row | mean | rows won |
|---|---|---|---|
| 20 | −0.04 (Moons erasure) | +8.09 | 7/8 |
| 100 | −0.26 (Moons erasure) | +3.45 | 7/8 |
| 500 | −0.01 (Moons fading) | +1.00 | 7/8 |

No m won all eight rows, so m = 20 was fixed for its much larger gaps. Moons erasure is saturated:
every method is near 87% at every m, so it got the follow-up below.

## Protocol

Code: `fading_unit_power/higher_accuracy.py`. Launcher: `fading_unit_power/submit_higher_accuracy.sh`.

1. **Screen.** At m ∈ {20, 100, 500}, every row on two tuning seed offsets (+10M, +20M), disjoint from
   the confirmation seeds:
   * ERM with its archived configuration, changing only m;
   * L2 with l2 × {1, 10, 100};
   * T2 with η_data × {1, 10, 100};
   * T1 fresh with s ∈ {3, 10, 30, 100} × posterior_std and η_channel × {1, 10, 100}, K = 4.

   912 fits scored on deployment accuracy on validation.
2. **Moons erasure follow-up at m = 20.** Six tuning offsets (+10M … +60M) and wider grids for every
   method:
   * L2 and T2 × {0.1, 1, 10, 100, 1000};
   * T1 with s ∈ {1, 3, 10, 30, 100, 300}, η_channel × {1, 10, 100, 1000} and stage-2
     η_data × {0.1, 1, 10}, K = 6.

   This gives 498 fits. Validation: T1 87.16, T2 86.91, L2 86.84, ERM 86.65.
3. **Selection.** For each row, each method's candidate with the best mean validation accuracy at m = 20.
4. **Confirmation.** The selected candidates were refit at m = 20 on the archived paired training
   seeds (3 per fading row; 5, 5, 9 and 5 per erasure row) and scored with the archived test
   evaluation seeds. ERM, L2 and T2 are retrained at m = 20, not reused.

The fresh stage-1 fits keep their archived m = 2000 config, which fresh training does not use, so
their task IDs equal those of the data-dependent campaign. The 52 checkpoints are linked from there,
not retrained.

Selected candidates (`l2-x10` means l2 × 10; `fresh-s3-e100` means s = 3 × posterior_std and
η_channel × 100; `-d10` means stage-2 η_data × 10):

| row | L2 | T2 | T1 |
|---|---|---|---|
| Moons fading | l2-x100 | t2-x10 | fresh-s3-e100 |
| Moons erasure | l2-x10 | t2-x0.1 | fresh-s300-e1-d10 |
| MNIST fading | l2-x10 | t2-x1 | fresh-s3-e10 |
| MNIST erasure | l2-x1 | t2-x1 | fresh-s3-e100 |
| F-MNIST fading | l2-x100 | t2-x10 | fresh-s3-e100 |
| F-MNIST erasure | l2-x1 | t2-x1 | fresh-s3-e1 |
| CIFAR-10 fading | l2-x100 | t2-x1 | fresh-s10-e100 |
| CIFAR-10 erasure | l2-x1 | t2-x10 | fresh-s30-e10 |

## Caveats

* As in `only_fresh`, stage 1 sees unlimited simulated channel draws that ERM, L2 and T2 do not, and
  T1 trains twice. The m = 20 gap mostly measures that advantage.
* T1 had more candidates than the baselines: 12 per row, and 72 for Moons erasure. The confirmation
  seeds are held out, so the test numbers are not selection-biased, but the baselines had less tuning.
* Moons erasure is a tie, not a win: +0.04 on the mean, and 3 of 5 paired seeds.
* m = 20 was chosen after seeing the screen, which uses validation data only.

## Provenance

Spartan run directory: `/data/gpfs/projects/punim2792/yangshuo/higher-accuracy-20261126`. The screen
ran at revision 649b796 (job 35779810). The follow-up, selection and confirmation ran at 454e984
(job 35806369). See [`provenance/REVISIONS.txt`](provenance/REVISIONS.txt). The manifests are in
`provenance/manifests/`. `provenance/selection-screen-minrule-649b796.json` is the first selection,
whose worst-row rule picked m = 500 and was replaced. Checkpoints stay on Spartan under `runs/`, and
their SHA-256 hashes are in `deployment_by_seed.csv`.
