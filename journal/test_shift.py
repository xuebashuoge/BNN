"""Bounded-loss channel shift checks; no training or dataset access."""

import itertools
import math
import unittest

from journal.theorem1 import channel_shift_diagnostic


class BoundedShiftTests(unittest.TestCase):
    def test_single_erasure_coordinate_is_absolute_probability_difference(self):
        result = channel_shift_diagnostic({"kind": "outage", "outage_prob": .2},
                                          {"kind": "outage", "outage_prob": .7}, 1)
        self.assertAlmostEqual(result["channel_total_variation_exact"], .5)
        self.assertAlmostEqual(result["channel_shift_complexity"], .5)
        self.assertEqual(result["shift_method"], "exact_total_variation_feature_erasure")

    def test_perfect_training_has_exact_finite_shift_despite_infinite_kl(self):
        for probability in (0., .1, .5, 1.):
            result = channel_shift_diagnostic({"kind": "outage", "outage_prob": probability},
                                              {"kind": "perfect"}, 8)
            self.assertAlmostEqual(result["channel_shift_complexity"], 1 - (1-probability)**8)
            if probability:
                self.assertIsNone(result["channel_kl_deployment_to_training"])
            else:
                self.assertEqual(result["channel_kl_deployment_to_training"], 0.)

    def test_grouped_tv_matches_explicit_mask_enumeration_and_is_symmetric(self):
        p, q, width = .2, .6, 4
        brute_force = .5 * math.fsum(
            abs(p**sum(mask) * (1-p)**(width-sum(mask)) -
                q**sum(mask) * (1-q)**(width-sum(mask)))
            for mask in itertools.product((0, 1), repeat=width))
        p_config, q_config = {"kind": "outage", "outage_prob": p}, {"kind": "outage", "outage_prob": q}
        forward = channel_shift_diagnostic(p_config, q_config, width)
        reverse = channel_shift_diagnostic(q_config, p_config, width)
        self.assertAlmostEqual(forward["channel_total_variation_exact"], brute_force)
        self.assertAlmostEqual(forward["channel_shift_complexity"], reverse["channel_shift_complexity"])

    def test_gaussian_identical_zero_and_singular_pair_universal_one(self):
        gaussian = {"kind": "gaussian", "gain_mean": .5, "gain_std": 1., "noise_std": 1.}
        matched = channel_shift_diagnostic(gaussian, gaussian, 256)
        self.assertEqual(matched["channel_shift_complexity"], 0.)
        singular = channel_shift_diagnostic(gaussian, {"kind": "perfect"}, 256)
        self.assertEqual(singular["channel_shift_complexity"], 1.)
        self.assertEqual(singular["shift_method"], "universal_bounded_loss_shift")
        self.assertIsNone(singular["channel_kl_deployment_to_training"])

    def test_gaussian_pinsker_is_capped_and_endpoint_erasure_laws_work(self):
        severe = channel_shift_diagnostic({"kind": "gaussian", "gain_mean": 10.},
                                          {"kind": "gaussian", "gain_mean": 1.}, 64)
        self.assertEqual(severe["channel_shift_complexity"], 1.)
        self.assertGreater(severe["channel_shift_pinsker_bound"], 1.)
        opposite = channel_shift_diagnostic({"kind": "outage", "outage_prob": 1.},
                                            {"kind": "outage", "outage_prob": 0.}, 256)
        self.assertEqual(opposite["channel_shift_complexity"], 1.)


if __name__ == "__main__":
    unittest.main()
