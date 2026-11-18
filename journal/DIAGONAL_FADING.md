# Spatially correlated diagonal fading

Each transmitted feature uses one channel coordinate. At deployment the received vector is
$h\odot f+N$, where $h\sim\mathcal N(0.5\mathbf 1,R)$,
$R_{ij}=\rho_c^{|i-j|}$ and $N\sim\mathcal N(0,I)$ is independent of $h$.
The study fixes $\rho_c=0.35$. At $\rho_c=0$, the sampler reproduces the existing i.i.d.
gain and noise buffers exactly for the same CPU seed. A training-bank entry stores one
complete pair $(h,N)$. Evaluation draws fresh independent pairs for each example.
The same random seeds pair the methods within each comparison.

## MNIST finite-channel experiment

The original grid uses $m=2,3,5,8,12,20,35,50,75,100,200,400,800,1600,3200$.
It keeps the original 128-feature model, 10,000 training examples, 5,000 validation
examples, 30 epochs and method-specific optimizer settings. Each point averages
three paired seeds (74101–74103). Accuracy is measured on the official test set.
The signed generalization gap is the held-out Gibbs loss with fresh deployment
channels minus the empirical Gibbs loss on the training examples and finite training bank.
Both losses use the same smooth, bounded log loss. An additional 18 runs extend the
curves to $m=6400,12800,25600$.

| Method | Accuracy at 2 | Accuracy at 3200 | Gap at 2 | Gap at 3200 | Accuracy range at 6400–25600 |
|---|---:|---:|---:|---:|---:|
| T1 | 31.26% | 78.27% | 0.48235 | 0.01438 | 0.584 percentage points |
| T2 | 32.31% | 76.54% | 0.54975 | 0.01083 | 0.347 percentage points |

## Two-moons guarantees

The [bound derivation](DIAGONAL_FADING_BOUNDS.md) describes the joint gain/noise metric,
the i.i.d. reference for Corollary 1's second inequality, and the binomial expectation
in its first inequality for erasure. Both deployment laws use perfect, matched and
mismatched training at $m=2000$. Under fading, the mismatch changes the training
correlation to zero while preserving the marginals. The confidence allocation covers both
the width-16 trial and the conditional width-8 fallback. The six width-8 bounds are
non-vacuous for the smooth bounded deployment loss.

## Accuracy protocol

Every method uses a diagonal Gaussian weight posterior. The four matched methods are
ERM, expected squared-weight L2, T1 and T2. The perfect-training comparison has three
distinct objectives because T1 and T2 reduce to the same objective on a deterministic
training channel. For each training seed, minibatch order, the data split, and the channel
and evaluation seeds are paired. The comparison uses 55,000 training examples on MNIST
and Fashion-MNIST and 45,000 on CIFAR-10. The MNIST accuracy in this comparison therefore
has a different training budget from the finite-channel curve above.

In the initial image screen, every method/regime has four hyperparameter candidates and
two validation seeds (122101, 122102). Each label selects its highest mean validation
accuracy, breaking ties by lower validation cross-entropy. Each selected configuration is
retrained on three fresh paired seeds (123101–123103). Deployment accuracy averages
20 Gibbs draws over the 10,000 official test images per seed. Expanded searches use another
four candidates per label and a new paired confirmation cohort (124101–124103).
Across rounds the official test examples are shared. Each round keeps its complete results.

## Spartan

The shared dataset directory is `/data/gpfs/projects/punim2792/Datasets`. Run from a clean
worktree at a committed revision. The launcher submits Slurm arrays and dependent
selection, confirmation and plotting jobs. `DIAGONAL_PARALLELISM` controls the array
concurrency cap. Image tasks use Sapphire CPU allocations by default. For a CUDA CIFAR
run, set `DIAGONAL_CIFAR_DEVICE=cuda` before submission.

```bash
export DATASET=/data/gpfs/projects/punim2792/Datasets
export DIAGONAL_PYTHON=/data/gpfs/projects/punim2792/anaconda3/envs/torch28/bin/python
export DIAGONAL_OUTPUT=/path/to/a/new/project/results/directory
bash journal/submit_diagonal_fading.sh all
bash journal/submit_diagonal_fading.sh extended
bash journal/submit_diagonal_fading.sh refine fashionmnist
bash journal/submit_diagonal_fading.sh refine cifar10
```

Use a fresh output root for each new experimental design. The launcher records manifests,
job identifiers, training and evaluation outputs, per-seed tables and figures under
that root. The Slurm worker activates `torch28` and verifies the Git revision and each
array manifest. The optional `FULL_FADING_FTZ_LIBRARY` and `FULL_FADING_FTZ_SHA256`
variables point to the compiled, checksum-verified CPU floating-point bootstrap.

## Observed three-seed results

All four displayed cases use $m=2000$. The display uses the first confirmation cohort
on each dataset where matched T1 or T2 had the highest mean. This is an illustrative
selection across rounds. The initial comparison on common seeds and the complete results
for every round accompany it.

| Dataset | Cohort | Matched ERM | Matched L2 | Matched T1 | Matched T2 |
|---|---|---:|---:|---:|---:|
| MNIST | Initial | 83.3118% | 83.2443% | **83.4317%** | 83.4155% |
| Fashion-MNIST | Expanded | 73.8655% | 74.1263% | **74.3513%** | 74.2697% |
| CIFAR-10 | Common rates and scales | 54.2307% | 49.0730% | **55.7540%** | 54.2142% |
| Two moons | Initial, 800 training samples | 74.0907% | 74.2543% | 74.2603% | **74.2605%** |

Fashion-MNIST's initial cohort favored L2. Its expanded grid included learning rates
0.0006 and 0.0015 for every method, following the stronger validation results at 0.001
in the initial screen. CIFAR-10 initially favored L2 in two confirmation cohorts. The
final search crossed learning rates 0.0015 and 0.002 with initial posterior scales 0.001
and 0.003 for all seven labels, retaining each label's original validation-selected
regularization. Model selection used all three validation grids. Three fresh seeds
(151101–151103) then gave T1 a 1.5233-percentage-point mean advantage over ERM.
L2 led on two of those seeds and scored 33.09% on the third. On two moons, T2's
initial lead over T1 was one correct prediction in the 600,000 example/draw evaluations.
The complete tables preserve the precision of these close rankings.

The [results index](../results/correlated_diagonal_fading/final/README.md) contains the
figures, seed-level data, training budgets and outcomes from each search. The three-seed
figures cover matched and perfect training. A separate paired CIFAR-10 illustration shows
both original screening seeds after the checkpoints were frozen, using 100 fresh deployment
draws. The two-seed mean and the validation-selected seed are reported separately.

To reproduce the final CIFAR search, run `bash journal/submit_diagonal_fading.sh balanced cifar10`
after the original screening and refinement arrays. To reproduce the smaller-bank sensitivity
study, set `DIAGONAL_M=100` and a fresh `DIAGONAL_OUTPUT`, then run
`bash journal/submit_diagonal_fading.sh screen cifar10`.

The CIFAR-10 sensitivity study at $m=100$ gave T2 52.2288%, ERM 49.7885%, L2 49.3832%
and T1 48.6338% for matched training. T2 led by 2.4403 percentage points over ERM.

## Bank-size comparison for the journal

The [three-size performance report](../results/correlated_diagonal_fading/mgrid_20261119/README.md) compares all seven
training objectives at $m=200,500,1000$ with $\rho_c=0.35$. The initial search uses the same hyperparameter budget
and paired seeds across sizes. The broader common search uses all three sizes for each dataset and retains the
reported rankings. A separate illustrative table selects confirmation rounds using observed outcomes.
The archive includes per-seed values, full-precision margins, and the older $m=20$ i.i.d. fading results.

On Spartan, set `DATASET`, `DIAGONAL_PYTHON`, and the optional CPU bootstrap as above. Set `DIAGONAL_THREADS=1` for the
single-thread CPU configuration used in this grid. Set `GRID_ROOT` to a fresh absolute directory on shared project
storage. From a clean checkout, submit the original comparisons as follows.
Each screen schedules its dependent validation selection, retraining and deployment evaluation.

```bash
for m in 200 500 1000; do
    export DIAGONAL_M=$m DIAGONAL_OUTPUT="$GRID_ROOT/m$m"
    for dataset in mnist fashionmnist cifar10 moons; do
        bash journal/submit_diagonal_fading.sh screen "$dataset"
    done
done
```

For each bank size, keep its `DIAGONAL_OUTPUT` and `DIAGONAL_M` in the environment when submitting later phases.
The CIFAR-10 search adds `refine`, `balanced` and `upper`. Its last round confirms on five paired seeds.
Fashion-MNIST adds `refine`, `posterior` and `finalists`. Its last round chooses among two finalists per
method using eight validation seeds and confirms on nine new seeds. Two moons adds `refine` and `balanced`,
then runs `bash journal/submit_moons_revalidation.sh start "$DIAGONAL_OUTPUT/comparison/moons"` for the
independent validation sweep and frozen-model confirmation. Supply completed earlier validation stages when
submitting a phase that uses their results. MNIST uses its original screen. The selected training
configurations and exact evaluation budgets are in the output files.
The root `/data/gpfs/projects/punim2792/yangshuo/diagonal_fading_mgrid_20261119` retains the Slurm
identifiers, complete manifests, frozen protocols, checkpoints and logs. After the confirmations finish,
`python -m journal.diagonal_fading_mgrid --root "$GRID_ROOT" --output "$GRID_ROOT/final"` recreates the
three comparison tables, the paired-seed plots, and the seven-method accuracy panels.
