# Journal experiment campaign

The target is to find and verify settings where the T1 or T2 proposed learner has the highest deployment test accuracy among the implemented comparison methods. Hyperparameters, seeds, channel-bank sizes and channel conditions may be explored. Every trial remains in the output; a test-selected or best-seed result is identified as exploratory, separately from confirmation using new training and evaluation seeds.

The common theoretical loss is

\[
\ell_\epsilon(w,c,z)=-\frac{\log[\epsilon+(1-\epsilon)p_w(y\mid x,c)]}{\log(1/\epsilon)},\qquad\epsilon=0.01.
\]

It lies in `[0,1]` uniformly over weights, channels and samples, so both sub-Gaussian variance proxies can be set to `1/4`. Training and theoretical-risk evaluation use this same loss. Raw cross entropy and test accuracy are reported separately. A raw-CE ERM reference is retained alongside ERM trained using the common bounded loss.

The screening stage uses two paired initialization/bank seeds, a fixed prior, fixed validation random numbers across epochs and checkpoint selection by validation accuracy with CE as a tiebreak. Finalist test inference uses fresh random numbers. Perfect training is included as its own reference condition and still evaluated under the same deployment channel.

The initial resource plan is one short CPU allocation with eight CPUs, four independent workers and two threads per worker. The measured single-worker MNIST pilot used under 0.6 GiB RSS and approximately 2.3 seconds per width-256 full-data epoch. Screening uses fewer samples, width 128 and 25 epochs. Twelve GiB host memory allows room for the four worker processes and coordinator; no GPU is requested.

The sequence is screening, full-data confirmation of selected configurations, finite-channel/data-size experiments, training/deployment mismatch, and inference-only bound decomposition. Selection results and whether the requested accuracy target is met will be recorded after the submitted jobs finish.

Screening job **34696908** completed successfully in 4 minutes 56 seconds: 104 training trials and 44 finalist test evaluations, no failures. The validation-selected MNIST T1 model averaged 70.818% test accuracy versus 70.817% for the strongest PAC baseline. The 0.001 percentage-point margin changes sign across seeds and is effectively a tie. T1 did not lead in the two toy conditions. This does not yet meet the target of a repeatable accuracy advantage.

The follow-up manifest combines 27 full-data MNIST confirmation trials (55,000 training examples, width 128, 100 epochs, seeds 9001–9003), 100 targeted smaller-bank trials, 102 finite-channel/data-size trials and 66 training-distribution mismatch trials. All 295 trials retain their individual stage labels, configurations and outcomes. Finalist inference uses ten fresh posterior/channel draws. The measured screening job peaked around 2 GiB RSS, so the follow-up requests eight CPUs and 8 GiB with a 50-minute limit.

For the bounded loss, a direct total-variation shift bound is also available without a channel-Lipschitz constant. Feature erasure uses exact TV computed by grouping masks by their erased-coordinate count. Perfect training gives shift `1-(1-p)^d`; a singular continuous deployment channel gives the universal, usually trivial bound 1. This is an additional bounded-loss consequence, not an unproved Lipschitz substitution. The raw KL remains infinite/undefined where appropriate.
