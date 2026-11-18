import math
import unittest

from scipy.integrate import quad
from scipy.stats import ncx2

from journal.diagonal_fading_bounds import design, perfect_shift
from journal.run_experiment import make_models


class DiagonalFadingBoundTests(unittest.TestCase):
    def test_second_inequality_uses_the_joint_metric_and_iid_reference(self):
        for width in (8, 16):
            config = design(width, "fading", "perfect", 2000, 1, 1)
            model, _ = make_models(config, 2, 2)
            row = perfect_shift(config["deployment_channel"], width, model, config["loss_epsilon"])
            k = math.sqrt(1 + model.feature_norm**2) * model.channel_lipschitz_constant(.01)
            self.assertAlmostEqual(row["uniform_lipschitz"], k)
            self.assertAlmostEqual(row["reference_kl"], -(width-1)/2 * math.log1p(-.35**2))
            # Independent integration checks the analytic upper bound on the joint norm.
            squared_norm = ncx2(2*width, width/4)
            moment, _ = quad(lambda z: math.exp(k*math.sqrt(z)) * squared_norm.pdf(z), 0., math.inf)
            self.assertGreater(row["reference_log_mgf_upper"], math.log(moment))
            self.assertLess(row["reference_log_mgf_upper"] - math.log(moment), .01)
            self.assertEqual(row["upper"], row["reference_kl"] + row["reference_log_mgf_upper"])

    def test_higher_noise_reference_includes_both_gaussian_scales(self):
        config = design(8, "fading", "perfect", 2000, 1, 1)
        model, _ = make_models(config, 2, 2)
        law = dict(kind="diagonal_fading", rho_c=.85, noise_std=2.)
        row = perfect_shift(law, 8, model, .01)
        a = row["quadratic_parameter"]
        # Exact quadratic moments for independent shifted gains and variance-four noise.
        from scipy.stats import chi2
        gain = ncx2(8, 2.)
        noise = chi2(8)
        mg, _ = quad(lambda z: math.exp(a*z/2) * gain.pdf(z), 0., 140.)
        mn, _ = quad(lambda z: math.exp(a*4*z/2) * noise.pdf(z), 0., 140.)
        self.assertAlmostEqual(row["reference_log_mgf_upper"],
                               row["uniform_lipschitz"]**2/(2*a)+math.log(mg*mn), places=9)
        self.assertAlmostEqual(row["reference_kl"], -3.5*math.log1p(-.85**2))

    def test_first_inequality_erasure_and_iid_limits(self):
        config = design(8, "erasure", "perfect", 2000, 1, 1)
        model, _ = make_models(config, 2, 2)
        erasure = perfect_shift(dict(kind="outage", outage_prob=1.), 8, model, .01)
        self.assertEqual(erasure["method"], "C1 first")
        self.assertEqual(erasure["expectation"], math.sqrt(8.))
        iid = perfect_shift(dict(kind="diagonal_fading", rho_c=0.), 8, model, .01)
        self.assertEqual(iid["reference_kl"], 0.)
        self.assertGreater(iid["reference_log_mgf_upper"], 0.)


if __name__ == "__main__":
    unittest.main()
