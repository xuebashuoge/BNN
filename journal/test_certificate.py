"""Confidence accounting checks; no checkpoint inference or training."""

import math
import unittest

from journal.certify_checkpoint import confidence_allocation


class CertificateConfidenceTests(unittest.TestCase):
    def test_union_budget_covers_two_bounds_and_all_setups(self):
        result = confidence_allocation(.05, 21, 100)
        self.assertEqual(result["simultaneous_theorems"], 2)
        self.assertAlmostEqual(2 * 21 * result["posterior_failure_per_theorem_setup"]
                               + result["monte_carlo_failure"], .05)
        self.assertAlmostEqual(result["empirical_mc_upper_radius"], math.sqrt(math.log(100) / 200))

    def test_more_samples_shrink_mc_error_and_invalid_budgets_fail(self):
        small = confidence_allocation(.05, 21, 100)
        large = confidence_allocation(.05, 21, 400)
        self.assertAlmostEqual(large["empirical_mc_upper_radius"], small["empirical_mc_upper_radius"] / 2)
        for args in ((0., 21, 100), (.05, 0, 100), (.05, 2.5, 100), (.05, 21, 0)):
            with self.assertRaises(ValueError):
                confidence_allocation(*args)


if __name__ == "__main__":
    unittest.main()
