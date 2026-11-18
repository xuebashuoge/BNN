"""Check continuous channel integration against raw gains, noise and exact deterministic ties."""
import unittest

import torch

from journal.diagonal_moons_revalidation import conditional_correct
from journal.full_fading import correlation_factor
from journal.theorem1 import ChannelBank, GaussianMLP


class MoonsRevalidationTests(unittest.TestCase):
    def test_binary_conditional_accuracy_agrees_with_sampled_deployment(self):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(8511921)
        model = GaussianMLP(2, 16, 2)
        theta = .4 * torch.randn(model.d, generator=rng)
        w1, b1, w2, b2 = model.split(theta)
        x = torch.randn(11, 2, generator=rng)
        y = torch.arange(len(x)) % 2
        for rho, sigma in ((0., 1.), (.35, 1.), (.85, 2.)):
            law = dict(kind="diagonal_fading", rho_c=rho, noise_std=sigma)
            bank = ChannelBank(16, 3, law, seed=13)
            exact = conditional_correct(x.double(), y, model, theta.double(), correlation_factor(rho, 16), sigma)
            estimates = []
            for inp, target in zip(x, y):
                hidden = (inp @ w1 + b1).relu().expand(120000, -1)
                logits = bank.apply(hidden, "deployment", law, rng) @ w2 + b2
                estimates.append((logits.argmax(1) == target).double().mean().item())
            torch.testing.assert_close(torch.tensor(estimates, dtype=torch.float64), exact, rtol=0, atol=.007)
        zero_theta = torch.zeros_like(theta.double())
        ties = conditional_correct(x.double(), y, model, zero_theta, correlation_factor(.35, 16))
        torch.testing.assert_close(ties, (y == 0).double())


if __name__ == "__main__":
    unittest.main()
