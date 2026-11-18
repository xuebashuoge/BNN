"""Regression tests for the correlated diagonal Gaussian channel and its iid limit."""
import unittest

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from journal.theorem1 import ChannelBank, channel_shift_diagnostic, validate_channel


def law(rho):
    return {"kind": "diagonal_fading", "rho_c": rho}


def covariance(width, rho):
    indices = torch.arange(width)
    return torch.tensor(rho, dtype=torch.float64) ** (indices[:, None] - indices).abs()


class DiagonalFadingTests(unittest.TestCase):
    def test_sampled_gains_and_noise_have_requested_joint_law(self):
        width, rho = 6, .7
        bank = ChannelBank(width, 50000, law(rho), seed=438)
        observations = torch.cat((bank.gain, bank.noise), dim=1).double()
        mean = torch.cat((observations.new_full((width,), .5), observations.new_zeros(width)))
        joint_covariance = torch.block_diag(covariance(width, rho), torch.eye(width, dtype=torch.float64))
        torch.testing.assert_close(observations.mean(0), mean, atol=.02, rtol=0)
        torch.testing.assert_close(observations.T.cov(), joint_covariance, atol=.025, rtol=0)

    def test_bank_and_fresh_actions_apply_elementwise_gain_then_noise(self):
        bank = ChannelBank(4, 17, law(.25), seed=7)
        features = torch.tensor([[0., 2., -3., 5.], [3., 0., -2., 2.], [5., 3., -2., 0.]])
        generator = torch.Generator().manual_seed(81)
        indices = torch.randint(17, (len(features),), generator=generator)
        torch.testing.assert_close(bank.apply(features, generator=generator.manual_seed(81)),
                                   bank.gain[indices] * features + bank.noise[indices])

        # Fresh draws can come from a different rho than the training bank.
        generator.manual_seed(91)
        factor = torch.linalg.cholesky(covariance(4, .8)).float()
        gain = .5 + torch.randn(features.shape, generator=generator) @ factor.T
        noise = torch.randn(features.shape, generator=generator)
        received = bank.apply(features, "deployment", law(.8), generator.manual_seed(91))
        torch.testing.assert_close(received, features * gain + noise)
        torch.testing.assert_close(bank.apply(features, "perfect"), features)

    def test_iid_limit_matches_seeded_legacy_channel(self):
        iid = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)
        diagonal, legacy = (ChannelBank(8, 24, config, seed=308) for config in (law(0.), iid))
        self.assertTrue(torch.equal(diagonal.gain, legacy.gain))
        self.assertTrue(torch.equal(diagonal.noise, legacy.noise))
        for deployment, training in ((law(0.), iid), (iid, law(0.))):
            shift = channel_shift_diagnostic(deployment, training, 8)
            self.assertEqual(shift["channel_shift_complexity"], 0.)
        self.assertEqual(channel_shift_diagnostic(law(.75), iid, 8),
                         channel_shift_diagnostic(law(.75), law(0.), 8))

    def test_analytic_shift_matches_dense_gaussian_kl(self):
        for width, p, q in ((1, .2, .9), (4, .95, .2), (8, .2, .95), (6, .7, 0.), (3, .5, .5)):
            with self.subTest(width=width, deployment=p, training=q):
                mean = torch.full((width,), .5, dtype=torch.float64)
                expected = kl_divergence(MultivariateNormal(mean, covariance(width, p)),
                                         MultivariateNormal(mean, covariance(width, q))).item()
                actual = channel_shift_diagnostic(law(p), law(q), width)
                self.assertAlmostEqual(actual["channel_kl_deployment_to_training"], expected, places=10)

    def test_rejects_invalid_correlations_and_incompatible_parameters(self):
        for rho in (-.01, 1, 1.2, float("nan"), True, float("inf"), "0.5"):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                validate_channel(law(rho))
        with self.assertRaises(ValueError):
            validate_channel({**law(.5), "noise_std": .4})


if __name__ == "__main__":
    unittest.main()
