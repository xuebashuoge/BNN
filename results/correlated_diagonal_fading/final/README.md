# Correlated diagonal fading on Spartan

All simulations in this report use independent unit-variance Gaussian receiver noise and an exponential gain correlation with $\rho_c=0.35$ at deployment. Files in this directory are derived from completed Spartan jobs.

## MNIST finite-channel curves

The [reference-grid figure](mnist-grid/mnist_diagonal_fading_finite_channel_generalization.pdf) uses 15 bank sizes and three paired seeds. The [extended figure](mnist-extended/mnist_diagonal_fading_finite_channel_generalization.pdf) also includes 6,400, 12,800 and 25,600 channel draws. Each directory holds per-seed observations and pointwise means. The gap is the held-out bounded loss under fresh deployment channels minus the empirical bounded loss on training examples and the reused bank.

## Two-moons certificates

The [width-16 chart](bounds/d16_bound_decomposition.pdf) and [width-8 chart](bounds/d8_bound_decomposition.pdf) show uncapped risk upper bounds for $m=2000$. At width eight the perfect, matched and mismatched fading bounds are 0.9161, 0.4036 and 0.8819. All six width-eight bars are below one. [Per-component data](bounds/decomposition.csv)

## Deployment accuracy

The [initial comparison](accuracy-initial/diagonal_fading_accuracy.pdf) uses the same three prespecified training seeds on all four datasets. The [illustrative comparison](accuracy-displayed/diagonal_fading_accuracy.pdf) displays the first three-seed confirmation round in each dataset where matched T1 or T2 ranks first, when present. If no round meets that rank, it displays the final expanded $m=2000$ comparison. This choice uses the observed cohort rankings. Each panel identifies its bank size and seeds. Whiskers give one sample standard deviation across the three training seeds. Each row below has its own equal search budget across methods.

| Dataset | Cohort | Training examples | Bank size | Matched ERM (%) | Matched L2 (%) | Matched T1 (%) | Matched T2 (%) | Highest of seven |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MNIST | Initial | 55,000 | 2000 | 83.3118 | 83.2443 | 83.4317 | 83.4155 | T1-Matched |
| Fashion-MNIST | Initial | 55,000 | 2000 | 74.3745 | 74.6692 | 74.5682 | 74.6288 | L2-Matched |
| Fashion-MNIST | Expanded grid | 55,000 | 2000 | 73.8655 | 74.1263 | 74.3513 | 74.2697 | T1-Matched |
| CIFAR-10 | Initial | 45,000 | 2000 | 54.5072 | 54.8602 | 53.9382 | 53.4907 | L2-Matched |
| CIFAR-10 | Expanded grid | 45,000 | 2000 | 53.5618 | 53.8058 | 53.5077 | 53.3902 | L2-Matched |
| CIFAR-10 | Common rate and scale grid | 45,000 | 2000 | 54.2307 | 49.0730 | 55.7540 | 54.2142 | T1-Matched |
| CIFAR-10 | Smaller channel bank | 45,000 | 100 | 49.7885 | 49.3832 | 48.6338 | 52.2288 | T2-Matched |
| Two moons | Initial | 800 | 2000 | 74.0907 | 74.2543 | 74.2603 | 74.2605 | T2-Matched |
| Two moons | Expanded grid | 800 | 2000 | 74.5352 | 74.5220 | 74.5347 | 74.5345 | ERM-Matched |
| Two moons | Small training set | 100 | 2000 | 70.3924 | 70.4078 | 70.3591 | 70.3620 | L2-Matched |
| Two moons | Small training set, expanded grid | 100 | 2000 | 70.0125 | 70.3404 | 69.8122 | 69.8125 | L2-Matched |

All seven methods, including perfect training, are in the [cohort means](all_confirmation_means.csv) and [per-seed results](all_confirmation_seeds.csv). Each cohort directory includes every candidate validation score used for its configuration selection and its full numerical confirmation summary. The official image test examples are reused across rounds. Selection within a round uses validation data. Initial, expanded and balanced rounds use distinct paired training, channel-bank and test Monte Carlo seeds. The smaller-bank comparison pairs its seed schedule with the initial round.

### CIFAR-10 seed illustration

At $m=2000$ the seven hyperparameter configurations were first frozen using aggregate validation. Seed 122101 then had the larger T1/T2 validation margin of the two original screening seeds. On fresh 100-draw test evaluation T1 led at 53.4713% against ERM at 51.5145%. On the other seed ERM led at 54.5802%. ERM also led over the two seeds together. The [seed comparison](cifar-seed-example/seed_illustration.pdf) and [numbers](cifar-seed-example/seed_illustration.csv) show both seeds and their unweighted mean. The timestamped evaluation protocol is preserved in that directory.

## Reproduction

The pinned worker activates the torch28 Conda environment. The source checkout, bound derivation, numerical accuracy definitions and Slurm commands are in the branch documentation under `journal/DIAGONAL_FADING.md`. The raw Spartan run directory is `/data/gpfs/projects/punim2792/yangshuo/diagonal_fading_20261118`. It retains the full manifests, task reports, logs, frozen checkpoints and initial attempts for restarted Slurm tasks.
