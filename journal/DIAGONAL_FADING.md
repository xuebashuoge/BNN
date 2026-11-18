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
