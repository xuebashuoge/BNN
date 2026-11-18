"""Regression checks for correlated full fading and its bank protocol."""
import unittest

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from journal.full_fading import correlation_factor, correlation_matrix, draw_full_fading
from journal.run_experiment import channel_bank_fingerprint, normalize_config
from journal.theorem1 import ChannelBank, channel_shift_diagnostic, validate_channel


class FullFadingTests(unittest.TestCase):
    def test_seeded_rectangular_draw_agrees_with_matrix_normal_formula(self):
        rx = correlation_factor(.3, 3)
        tx = correlation_factor(-.2, 2)
        generator = torch.Generator().manual_seed(24)
        gain, noise = draw_full_fading(7, rx, tx, .6, generator)
        generator.manual_seed(24)
        white = torch.randn((7, 3, 2), dtype=torch.float64, generator=generator)
        expected_noise = .6 * torch.randn((7, 3), dtype=torch.float64, generator=generator)
        torch.testing.assert_close(gain, .5 + rx @ white @ tx.T)
        torch.testing.assert_close(noise, expected_noise)

    def test_dense_bank_has_requested_entry_marginals_and_covariance(self):
        config = dict(kind="full_fading", rx_correlation=.35, tx_correlation=-.2, noise_std=0.)
        bank = ChannelBank(3, 24000, config, 79)
        gains = bank.gain.flatten(1).double()
        covariance = torch.kron(correlation_matrix(.35, 3), correlation_matrix(-.2, 3))
        torch.testing.assert_close(gains.mean(0), torch.full((9,), .5, dtype=gains.dtype), rtol=0, atol=.025)
        torch.testing.assert_close(gains.T.cov(), covariance, rtol=0, atol=.035)
        self.assertFalse(bank.noise.any())

    def test_bank_orientation_and_fresh_projected_law(self):
        config = dict(kind="full_fading", rx_correlation=.3, tx_correlation=.2, noise_std=.4)
        bank = ChannelBank(3, 1, config, 23)
        hidden = torch.tensor([[1., -2., 3.], [-1., 2., 5.]])
        expected = hidden @ bank.gain[0].T + bank.noise[0]
        torch.testing.assert_close(bank.apply(hidden), expected)
        generator = torch.Generator().manual_seed(54)
        fading = torch.randn(hidden.shape, generator=generator)
        noise = torch.randn(hidden.shape, generator=generator)
        rx = correlation_factor(.3, 3).float()
        rt = correlation_matrix(.2, 3).float()
        scale = (hidden * (hidden @ rt)).sum(-1, keepdim=True).sqrt()
        expected = .5 * hidden.sum(-1, keepdim=True) + scale * (fading @ rx.T) + .4 * noise
        torch.testing.assert_close(bank.apply(hidden, "deployment", generator=generator.manual_seed(54)), expected)

    def test_shift_kl_matches_joint_gaussian_reference(self):
        p = dict(kind="full_fading", rx_correlation=.25, tx_correlation=-.2, noise_std=.8)
        q = dict(kind="full_fading", rx_correlation=.2, tx_correlation=.1, noise_std=1.)
        width = 3

        def law(config):
            rt = correlation_matrix(config["tx_correlation"], width)
            rr = correlation_matrix(config["rx_correlation"], width)
            noise = config["noise_std"]**2 * torch.eye(width, dtype=torch.float64)
            mean = torch.cat((rr.new_full((width**2,), .5), rr.new_zeros(width)))
            return MultivariateNormal(mean, torch.block_diag(torch.kron(rr, rt), noise))

        diagnostic = channel_shift_diagnostic(p, q, width)
        self.assertAlmostEqual(diagnostic["channel_kl_deployment_to_training"], kl_divergence(law(p), law(q)).item())
        self.assertEqual(channel_shift_diagnostic(q, q, width)["channel_shift_complexity"], 0.)
        self.assertIsNone(channel_shift_diagnostic(p, {"kind": "perfect"}, width)["channel_kl_deployment_to_training"])

    def test_explicit_correlations_are_validated_and_fingerprinted(self):
        config = dict(kind="full_fading", rx_correlation=[[1., .4], [.4, 1.]])
        first, second = (ChannelBank(2, 8, config, 2) for _ in range(2))
        self.assertEqual(channel_bank_fingerprint(first), channel_bank_fingerprint(second))
        second.gain[3, 0, 1] += .1
        self.assertNotEqual(channel_bank_fingerprint(first), channel_bank_fingerprint(second))
        with self.assertRaisesRegex(ValueError, "size"):
            normalize_config(dict(width=3, deployment_channel=config))
        for matrix in ([[2., 0.], [0., 1.]], [[1., .3], [.4, 1.]], [[1., 1.], [1., 1.]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                validate_channel({**config, "rx_correlation": matrix})


if __name__ == "__main__":
    unittest.main()
