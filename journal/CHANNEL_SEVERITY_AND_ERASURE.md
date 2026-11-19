# Harder diagonal fading and 50% feature erasure

The three main choices of channel-bank size are 200, 500 and 1000. The five comparison tables also include the existing m=2000 design on all four datasets and the CIFAR-10-only m=100 sensitivity design. Two moons uses an iid training sample of 800 and a width-16 split network. The image experiments use 55,000 training examples for MNIST/Fashion-MNIST and 45,000 for CIFAR-10, a width of 128, and the original split architectures.

## Diagonal fading

The model transmits $h\odot f+N$. The gain mean remains 0.5 and its marginal standard deviation remains one. Increasing $\rho_c$ from 0.35 to 0.85 raises the adjacent-gain correlation. Doubling $\sigma_N$ from 1 to 2 quadruples the additive noise power. The experiment uses a four-condition factorial test on each frozen original classifier, then retrains on the combined harder law. The fixed-checkpoint evaluation and recovery check share posterior and standardized channel streams. The transfer training preserves the original optimizer and model settings, finite-bank size and paired seeds. Both matched and perfect training use hard-channel validation to select the epoch.

For each original confirmation manifest, the transfer launcher is `bash journal/submit_fading_severity.sh SOURCE_MANIFEST ABSOLUTE_OUTPUT CPU_THREADS`. The seven-method report supplies per-seed scores, means and SDs. Fresh hard-validation searches are separate cohorts. Frozen-model tests and recovery use `journal/submit_diagonal_deployment.sh`; the run directories retain their frozen source protocols. `python -m journal.report_fading_severity --root SEVERITY_ROOT --output OUTPUT` assembles the effects once all required models are available.

## Feature erasure

The matched channel independently erases each coordinate with probability 0.50 without additive noise. Perfect-trained comparators use the identity link during training. Every erasure row reports test accuracy using fresh 50% erasure masks. The archived m=20 erasure winners supply the original per-label training configurations; their matched regularization strengths follow the original bank-size scaling. The new experiments use data seed 20261118 to align the data split with fading and five common training seeds for the anchor comparison. The historical perfect-training T1 and T2 configurations coincide after replacing the algorithm name because the two objectives have the same perfect-link specialization.

The launcher is `bash journal/submit_erasure_bank.sh DATASET_NAME M ABSOLUTE_OUTPUT CPU_THREADS`. It reads `ERASURE_SOURCE` for the assembled historical confirmation manifest. Both launchers use `DATASET` for the data location and `DIAGONAL_PYTHON` for the torch28 interpreter. The Slurm worker verifies the submitted Git revision and the scientific task verifies each array manifest by hash.

## Journal tables

Within a dataset and deployment law, the common comparison uses the same available validation/confirmation phase and paired seed cohort across the three principal bank sizes. The illustrative table uses test rankings to identify a round in which a matched proposed objective leads. It prefers one common winning round across all three sizes; otherwise the cohort is designated separately for each size. The transfer table always includes the original validation-selected hyperparameters. The full-precision source and seven-method data accompany these views.

Each displayed cell is mean deployment accuracy in percent ± sample SD across training seeds. The columns are matched ERM, L2, T1 and T2, and the shared perfect-training T1/T2 objective. The largest unrounded mean among the displayed columns is bold. Perfect ERM and L2 are retained in the CSV comparisons over all seven labels. Paired row-wise differences, convergence of the frozen-channel experiments, and statistical variability across training seeds are separate quantities.

## Artifacts

The [release index](../results/journal_channel_tables_20261119/README.md) links the common-protocol and illustrative
five-page PDFs, individual table images, exact data and manuscript-ready LaTeX fragments. The matched/perfect
training table sources and all confirmation cohorts are stored beside the tables. The [paired severity
report](../results/harder_diagonal_fading_20261119/README.md) covers the original checkpoints, four deployment
laws and their matched hard-trained counterparts.

On Spartan, the three raw roots in the project account's yangshuo directory are
`diagonal_fading_severity_20261119`, `erasure_mgrid_20261119`, and `journal_channel_tables_20261119`.
The release can be rebuilt with
`python -m journal.publish_channel_tables --fading SEVERITY_ROOT --erasure ERASURE_ROOT --output TABLE_OUTPUT`
using the torch28 environment.
