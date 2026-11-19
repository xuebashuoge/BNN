# Diagonal fading severity comparison

The three principal bank sizes are 200, 500 and 1000. The study also repeats the four-dataset m=2000 reference and the earlier CIFAR-10-only m=100 reference. MNIST and Fashion-MNIST use 55,000 training examples, CIFAR-10 uses 45,000, and two moons uses 800 with a 16-feature split.

## Channel conditions

The received signal is h ⊙ f + N, with diagonal gain, mean gain 0.5 and variance one per coordinate. The gain covariance is R[i,j] = rho_c^|i−j|. Noise is independent of the gains and has covariance noise_std² I. The reference is (rho_c, noise_std) = (0.35, 1). The combined harder channel uses (0.85, 2). The correlation-only (0.85, 1) and noise-only (0.35, 2) conditions are included in the paired fixed-model experiment. At a fixed input power, doubling noise standard deviation quadruples its power, a 6.02 dB increase. The diagonal entries' mean, variance and feature order stay fixed.

## Paired tests

1. Freeze all seven previously selected method configurations, training/validation splits and checkpoint identities. For every original matched- or perfect-trained checkpoint, evaluate the four deployment conditions with common posterior draws, standardized gain/noise innovations and held-out examples. The within-checkpoint contrasts isolate correlation, noise and their combination.
2. Transfer each previously selected configuration and its exact training/finite-bank seed schedule to the harder experiment. Matched methods train and deploy at (0.85, 2). Perfect comparators train with the identity channel, select epochs by hard-channel validation and deploy at (0.85, 2). The corresponding original data budgets, architectures, optimizer settings, epoch counts, runtime threads and native deployment draw counts are recorded by the paired protocol. All seven labels and all seeds enter the transfer comparison.
3. Evaluate the transferred/retrained models on the same hard-channel random streams as the fixed-checkpoint experiment, to measure the accuracy recovered by channel-aware training.
4. Additional hard-law searches, if used, select each label's hyperparameters using validation. They retain all seven objectives and separate results for their fresh paired confirmation cohorts. The primary transfer comparison remains visible for every bank size.

The original matched-method labels are ERM, L2, T1 and T2. Perfect training yields three distinct objectives (ERM, L2 and the shared T1/T2 objective). The method margin compares the better matched T1/T2 mean with the largest mean among the remaining five labels. All accuracy differences use percentage points. Candidate screening and confirmation use separate seeds. The official image test set is shared between experimental conditions.
