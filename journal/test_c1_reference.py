"""Independent integration, channel sampling and uniform-loss gradient checks."""
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
import torch

from journal.c1_reference import (centered_lognormal_kl, erasure_first_shift,
    lognormal_distance_benchmark, lognormal_logit_channel_terms,
    log_rayleigh_mgf, sample_centered_lognormal, smooth_loss_lipschitz)


class C1ReferenceTests(unittest.TestCase):
    def test_exact_kl_matches_independent_transformed_density_integration(self):
        for sigma, shape, s in ((.5, .1, .5), (.5, .3, .47), (1., .7, 1.4)):
            scale = sigma / math.sqrt(math.expm1(shape**2))

            def log_density_ratio(z):
                y = math.exp(shape * z - shape**2 / 2)
                noise = scale * (y - 1)
                log_p = (-.5 * z**2 - .5 * math.log(2 * math.pi) -
                         math.log(scale * shape) - math.log(y))
                log_q = -.5 * math.log(2 * math.pi * s**2) - noise**2 / (2 * s**2)
                return (log_p - log_q) * math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)

            numerical = 2 * quad(log_density_ratio, -12, 12, epsabs=1e-11)[0]
            self.assertAlmostEqual(centered_lognormal_kl(sigma, shape, s), numerical, places=10)

    def test_rayleigh_mgf_matches_independent_radial_integration(self):
        for sigma in (0., .1, .25, .5, 1., 2.):
            t = smooth_loss_lipschitz() * sigma
            integral = quad(lambda r: r * math.exp(t * r - r**2 / 2),
                            0, math.inf, epsabs=1e-11)[0]
            self.assertAlmostEqual(log_rayleigh_mgf(t), math.log(integral), places=11)
        # A direct exp(t²/2) would overflow, while the log expression is finite.
        self.assertTrue(math.isfinite(log_rayleigh_mgf(100.)))

    def test_analytic_terms_use_no_distance_benchmark_or_random_numbers(self):
        with patch("journal.c1_reference.lognormal_distance_benchmark", side_effect=AssertionError), \
                patch("numpy.random.default_rng", side_effect=AssertionError):
            row = lognormal_logit_channel_terms(.5)
        self.assertAlmostEqual(row["log_exponential_moment"], .19760937122662084, places=12)
        self.assertAlmostEqual(row["channel_kl"], .015004166663194492, places=12)
        self.assertAlmostEqual(row["c1_second_shift_upper"], .21261353788981532, places=12)
        self.assertLess(row["c1_second_shift_upper"], row["jensen_shift_upper"])
        self.assertGreater(lognormal_logit_channel_terms(.1)["c1_second_shift_upper"],
                           lognormal_logit_channel_terms(.1)["jensen_shift_upper"])

    def test_zero_noise_is_separate_from_positive_noise_shape_penalty(self):
        row = lognormal_logit_channel_terms(0.)
        for key in ("channel_kl", "reference_std", "log_exponential_moment",
                    "c1_second_shift_upper", "jensen_shift_upper"):
            self.assertEqual(row[key], 0.)
        self.assertEqual(row["kl_to_perfect_training"], "zero")
        near = lognormal_logit_channel_terms(1e-12)
        self.assertGreater(near["channel_kl"], .015)
        self.assertEqual(near["kl_to_perfect_training"], "infinite")
        self.assertLess(centered_lognormal_kl(.5, 1e-4), 2e-8)

    def test_erasure_term_matches_explicit_mask_probability(self):
        self.assertEqual(erasure_first_shift(8, 0.), 0.)
        self.assertEqual(erasure_first_shift(8, 1.), 1.)
        self.assertAlmostEqual(erasure_first_shift(8, .1),
                               sum(math.comb(8, k) * .1**k * .9**(8-k) for k in range(1, 9)))
        self.assertAlmostEqual(erasure_first_shift(8, .5), 1 - .5**8)

    def test_channel_samples_have_declared_moments_and_skew(self):
        sigma, shape = .7, .3
        noise = sample_centered_lognormal(np.random.default_rng(3109), (200_000, 2), sigma, shape)
        self.assertLess(np.max(np.abs(noise.mean(axis=0))), .006)
        self.assertLess(np.max(np.abs(noise.var(axis=0) - sigma**2)), .008)
        self.assertLess(abs(np.cov(noise, rowvar=False)[0, 1]), .005)
        self.assertTrue(np.all(noise > -sigma / math.sqrt(math.expm1(shape**2))))
        skew = np.mean((noise - noise.mean())**3) / np.std(noise)**3
        expected_skew = (math.exp(shape**2) + 2) * math.sqrt(math.expm1(shape**2))
        self.assertLess(abs(skew - expected_skew), .06)
        np.testing.assert_array_equal(
            sample_centered_lognormal(np.random.default_rng(1), (4, 2), sigma=0.), np.zeros((4, 2)))

    def test_distance_mc_is_reproducible_and_preserves_diagnostic_status(self):
        first = lognormal_distance_benchmark(samples=30_001, seed=813, batch_size=8_000)
        second = lognormal_distance_benchmark(samples=30_001, seed=813, batch_size=8_000)
        self.assertEqual(first, second)
        self.assertIn("not a certified", first["status"])
        self.assertGreater(first["unit_scale_mean_norm_standard_error"], 0.)
        # Cauchy-Schwarz/Jensen guarantees E||N||<=sqrt(2); this MC comparison
        # is a broad check of the intended two-dimensional diagnostic.
        self.assertLess(first["unit_scale_mean_norm"], math.sqrt(2.))
        self.assertGreater(first["unit_scale_mean_norm"], 1.1)

    def test_uniform_logit_lipschitz_constant_matches_analytic_gradient(self):
        generator = torch.Generator().manual_seed(519)
        epsilon = .01
        for classes in (2, 10):
            logits = (torch.randn((300, classes), generator=generator, dtype=torch.float64) * 7).requires_grad_()
            labels = torch.arange(len(logits)) % classes
            probabilities = logits.softmax(-1)
            py = probabilities[torch.arange(len(logits)), labels]
            loss = -torch.log(epsilon + (1 - epsilon) * py) / math.log(1 / epsilon)
            gradient = torch.autograd.grad(loss.sum(), logits)[0]
            one_hot = torch.nn.functional.one_hot(labels, classes)
            factor = (1 - epsilon) * py / (epsilon + (1 - epsilon) * py)
            analytic = factor[:, None] * (probabilities - one_hot) / math.log(1 / epsilon)
            torch.testing.assert_close(gradient, analytic)
            self.assertLessEqual(gradient.norm(dim=1).max().item(), smooth_loss_lipschitz(epsilon) + 1e-12)

    def test_invalid_channel_and_reference_parameters_are_rejected(self):
        for call in (lambda: lognormal_logit_channel_terms(-1.),
                     lambda: lognormal_logit_channel_terms(float("nan")),
                     lambda: lognormal_logit_channel_terms(1., shape=0.),
                     lambda: lognormal_logit_channel_terms(0., shape=-1.),
                     lambda: centered_lognormal_kl(0.),
                     lambda: centered_lognormal_kl(1., reference_std=0.),
                     lambda: smooth_loss_lipschitz(0.), lambda: smooth_loss_lipschitz(1.),
                     lambda: log_rayleigh_mgf(-1.), lambda: log_rayleigh_mgf(float("inf")),
                     lambda: erasure_first_shift(True, .1), lambda: erasure_first_shift(8, 1.1),
                     lambda: sample_centered_lognormal(np.random.default_rng(1), (2, 0)),
                     lambda: lognormal_distance_benchmark(samples=1),
                     lambda: lognormal_distance_benchmark(batch_size=0)):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
