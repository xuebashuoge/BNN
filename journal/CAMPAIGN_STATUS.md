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
