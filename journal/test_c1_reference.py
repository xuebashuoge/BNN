"""Independent chi-MGF integrals, mean-norm integration, and Lipschitz checks."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
import torch

from journal.evaluate_c1_reference import (erasure_first_shift, gaussian_logit_channel_terms,
    gaussian_reference_terms, log_chi_mgf, smooth_loss_lipschitz,
    centered_lognormal_reference_terms, lognormal_logit_channel_terms)
from journal.run_experiment import bounded_losses_from_ce


class ReferenceFormulaTests(unittest.TestCase):
    def test_centered_lognormal_kl_matches_independent_normal_integration(self):
        sigma, shape, s = .5, .1, .47
        variance = math.expm1(shape*shape)
        scale = sigma/math.sqrt(variance)
        # Integrate log p(N(z))-log q(N(z)) under standard-normal z.
        def log_ratio(z):
            y = math.exp(shape*z-shape*shape/2)
            noise = scale*(y-1)
            log_p = -.5*z*z-.5*math.log(2*math.pi)-math.log(scale*shape)-math.log(y)
            log_q = -.5*math.log(2*math.pi*s*s)-noise*noise/(2*s*s)
            return (log_p-log_q)*math.exp(-z*z/2)/math.sqrt(2*math.pi)
        numerical = 2*quad(log_ratio, -12, 12, epsabs=1e-11)[0]
        result = centered_lognormal_reference_terms(sigma, shape, s, .3)
        self.assertAlmostEqual(result["channel_kl"], numerical, places=10)

    def test_lognormal_analytic_bound_uses_no_distance_benchmark(self):
        without = lognormal_logit_channel_terms(.5, .1)
        with_benchmark = lognormal_logit_channel_terms(.5, .1,
            benchmark=dict(unit_scale_mean_norm=1.25, unit_scale_mean_norm_standard_error=.001))
        self.assertEqual(without["c1_second_shift_upper"], with_benchmark["c1_second_shift_upper"])
        self.assertIsNone(without["direct_first_shift_term"])
        self.assertLess(without["c1_second_shift_upper"], without["jensen_shift_upper"])
        self.assertAlmostEqual(centered_lognormal_reference_terms(.5, 1e-4, .5, .3)["channel_kl"], 0., places=6)

    def test_chi_mgf_matches_independent_scalar_quadrature(self):
        for dimension in (1, 2, 3, 8):
            for t in (0., .1, .7):
                log_constant = (1-dimension/2)*math.log(2) - gammaln(dimension/2)
                integral = quad(lambda r: math.exp(t*r-r*r/2+log_constant) * r**(dimension-1),
                                0, math.inf, epsabs=1e-11)[0]
                self.assertAlmostEqual(log_chi_mgf(t, dimension), math.log(integral), places=10)

    def test_equal_isotropic_reference_has_zero_gaussian_kl(self):
        result = gaussian_reference_terms(np.eye(2) * .25, .5, .3)
        self.assertAlmostEqual(result["channel_kl"], 0.)
        self.assertAlmostEqual(result["c1_second_shift_upper"], log_chi_mgf(.15, 2))

    def test_direct_mean_norm_matches_independent_angular_integration(self):
        for rho in (0., .3, -.7):
            sigma = .8
            row = gaussian_logit_channel_terms(sigma, rho)
            angular = quad(lambda a: math.sqrt((1+rho)*math.cos(a)**2+(1-rho)*math.sin(a)**2),
                           0, 2*math.pi, epsabs=1e-11)[0] / (2*math.pi)
            expected = smooth_loss_lipschitz(.01) * sigma * math.sqrt(math.pi/2) * angular
            self.assertAlmostEqual(row["direct_first_shift_term"], expected, places=10)

    def test_second_is_relaxation_and_reference_optimization_helps(self):
        for sigma in (.01, .1, .5, 2.):
            row = gaussian_logit_channel_terms(sigma, .3)
            self.assertGreaterEqual(row["c1_second_shift_upper"], row["direct_first_shift_term"]-1e-12)
            self.assertLessEqual(row["c1_second_shift_upper"], row["reference_std_equal_to_sigma_upper"]+1e-10)
            self.assertEqual(row["kl_to_perfect_training"], "infinite")
            self.assertLess(row["reference_std"], sigma)
        self.assertEqual(gaussian_logit_channel_terms(0., .3)["c1_second_shift_upper"], 0.)

    def test_uniform_smooth_loss_logit_gradient_constant(self):
        config = dict(training_loss="bounded_ce_smooth", loss_epsilon=.01)
        generator = torch.Generator().manual_seed(519)
        for classes in (2, 10):
            logits = (torch.randn((50, classes), generator=generator, dtype=torch.float64) * 10).requires_grad_()
            labels = torch.arange(50) % classes
            ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = bounded_losses_from_ce(ce, config)
            gradient = torch.autograd.grad(loss.sum(), logits)[0]
            self.assertLessEqual(gradient.norm(dim=1).max().item(), smooth_loss_lipschitz(.01)+1e-12)

    def test_erasure_discrete_metric_matches_nonidentity_probability(self):
        self.assertEqual(erasure_first_shift(8, 0.), 0.)
        self.assertEqual(erasure_first_shift(8, 1.), 1.)
        self.assertAlmostEqual(erasure_first_shift(8, .5), 1-2**-8)

    def test_invalid_formulas_are_rejected(self):
        for call in (lambda: log_chi_mgf(-1, 2), lambda: log_chi_mgf(.2, 0),
                     lambda: gaussian_logit_channel_terms(-1, .3),
                     lambda: gaussian_logit_channel_terms(1, 1),
                     lambda: gaussian_reference_terms([[1, 2], [2, 1]], 1, 1),
                     lambda: smooth_loss_lipschitz(0), lambda: erasure_first_shift(0, .5)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
