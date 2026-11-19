# Joint fading and erasure bound decomposition

Each width uses five separately trained predictors per channel. The perfect-trained learner is certified with Corollary 1. T1 training is certified with Theorem 1 and T2 training with Theorem 2, under both matched and mismatched training. Within a stochastic training law the two learners share the same training data, bank and seeds and fit distinct posterior means. Training uses 0.02 times the corresponding complexity. Certification uses the full unweighted term.

Deployment erases each feature independently with probability 0.50. Matched training uses 0.50 and mismatch training uses 0.45. In the fading comparison the test correlation is 0.35, the mismatch training correlation is 0.15, and receiver noise has unit variance. All cases use the same two-moons design with 800 training examples, 200 validation examples, 120 epochs and a training bank of 2000.

## Uncapped upper bounds on smooth bounded deployment loss

| Channel | Width | Perfect (C1) | Matched T1 | Matched T2 | Mismatch T1 | Mismatch T2 |
|---|---:|---:|---:|---:|---:|---:|
| Erasure | 16 | 0.4293 | 0.4796 | 0.4028 | 0.6834 | 0.6060 |
| Erasure | 8 | 0.3704 | 0.4185 | 0.3588 | 0.5639 | 0.5027 |
| Fading | 16 | 1.5413 | 0.4808 | 0.4033 | 0.8980 | 0.8221 |
| Fading | 8 | 0.9176 | 0.4058 | 0.3608 | 0.6912 | 0.6455 |

The confidence budget is simultaneous across both channels, both widths and all five fits per channel/width. It allocates 0.00125 each to the assigned theorem event and its independent empirical Monte Carlo event in each of 20 cases. Each empirical estimate uses 20,000 independent bounded observations. Its one-sided Hoeffding correction is 0.0129273. Diamonds show held-out Gibbs loss estimated with 20 draws over 10,000 examples. All ten erasure upper bounds are below one. Perfect-trained fading at width 16 has the sole vacuous certificate.

For perfect-trained erasure, C1(i) uses the exact finite-binomial expectation of the Frobenius mask distance. Its channel penalty is 0.064601 at width 16 and 0.045200 at width 8. For mismatched training, the bounded-loss shift for T1 and T2 is sqrt(KL(test || train)/2). Here the KL is −d log(0.99)/2. The mismatch adds 0.200503 at width 16 and 0.141777 at width 8. The matched law has zero shift. Perfect-trained fading uses C1(ii).

[Erasure figure (PDF)](erasure_bound_decomposition.pdf) · [PNG](erasure_bound_decomposition.png) · [Erasure numerical components and held-out accuracies](erasure_bound_decomposition.csv)

[Two-channel figure (PDF)](fading_erasure_bound_decomposition.pdf) · [PNG](fading_erasure_bound_decomposition.png) · [Combined CSV](fading_erasure_bound_decomposition.csv) · [Verified case provenance](fading_erasure_bound_decomposition.json)

## Reproduction

Run `bash journal/submit_correlated_bounds.sh /absolute/shared/output 0.15 both` from the repository on Spartan. Set DATASET to the prepared data directory. The Slurm worker uses the torch28 environment. The joint run trained ten new erasure models. It restored byte-verified fading checkpoints from the preceding fading study and recomputed their certificates with the joint confidence budget. This changes the fading confidence terms. The weights, selected epochs, empirical losses and held-out scores match the original fading study.

Spartan job array 35404116 evaluated the 20 cases. Audit job 35404171 checked parameters, checkpoint fingerprints, paired stochastic banks, independently fitted means, numerical components and confidence. The outputs are stored in `/data/gpfs/projects/punim2792/yangshuo/erasure_fading_bound_decomposition_20261119`.
