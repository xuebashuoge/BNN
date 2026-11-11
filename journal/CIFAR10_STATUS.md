# CIFAR-10 journal extension

This extension tests a compact split CNN on CIFAR-10 with ERM and L2 as the main competing baselines, proposed T1/T2 training, and perfect-training references. It preserves the shared smooth bounded loss and the single Gaussian-posterior interpretation. All real training runs on Spartan; local checks use synthetic tensors or checkpoint inference only.

The transmitter has three convolution/ReLU blocks (16, 32 and 64 output channels), two max-pooling stages, adaptive 4-by-4 pooling, and an affine/ReLU projection to 128 transmitted features. The receiver is a ten-class affine classifier. There are 156,074 network weights and biases; proposed models learn their diagonal-Gaussian means and scales. There is no batch normalization, dropout, CSI input, random augmentation or transmit-power normalization.

CIFAR data are checksum-verified and cached under `DATASET`. Pixel data are converted once to contiguous float32 CHW tensors with fixed `pixel/127.5-1` preprocessing. This avoids repeated PIL transforms while retaining a deterministic, data-independent transform. Main screening uses 10,000 training/5,000 validation examples; full-data comparison uses 45,000/5,000. The official test partition has 10,000 examples and is not opened by the training routine.

The initial CPU allocation covers small two-epoch developer pilots and one-epoch screen/full-shape pilots for each method/training law. GPU execution requires a completed matching CPU pilot, including architecture, dimensions and channel settings. A short GPU benchmark will be used if needed before sizing the larger stage. Resource times and observed results will be recorded here after measurement; pilot accuracy is not treated as a converged result.

The initial severe Gaussian gain/noise law matches the earlier image experiments, `(gain_mean,gain_std,noise_std)=(0.5,1,1)`. Moderate and near-ideal presets are available for sensitivity checks. Checkpoint selection uses validation accuracy with CE as tiebreak; main comparisons retain all seeds. Data-only PAC matched is not a main competitor; its perfect-training specialization is labelled Proposed Perfect.

CPU job **34837394** completed all 18 resource probes successfully in 44 seconds. The single-worker peak RSS was approximately 1.21 GiB, and Slurm's batch peak was 4.60 GiB. Screen-size epochs took approximately 2.5--2.7 seconds; full-size epochs 7.7--8.8 seconds, including two validation draws. These pilots are resource/execution checks, not converged performance comparisons. The record is `cifar_cpu_resources.json`.

Existing MNIST/FashionMNIST defaults remain MLPs. Architecture is part of the cache/resource signature, and the certificate sampler check is restricted to its unchanged IID-toy branch and shared splitting logic, so adding CIFAR does not silently invalidate or alter the previous toy experiment.
