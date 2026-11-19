import math
import unittest

from journal.correlated_bound_study import CASES, FAMILY_SIZE, learner_config
from journal.diagonal_fading_bounds import perfect_shift
from journal.full_fading_bounds import risk_decomposition
from journal.run_experiment import make_models


class CorrelatedCertificateTests(unittest.TestCase):
    def test_objectives_and_certificates_follow_the_five_case_design(self):
        self.assertEqual(FAMILY_SIZE, 10)
        for case, (regime, method, certificate) in CASES.items():
            cfg = learner_config(8, case)
            self.assertEqual(cfg["method"], method)
            self.assertEqual(cfg["deployment_channel"]["rho_c"], .35)
            self.assertEqual(cfg["eta_channel"], .02 if certificate == "T1" else 0.)
            self.assertEqual(cfg["eta_data"], .02)
            model, _ = make_models(cfg, 2, 2)
            empirical = dict(mean=.15, iid_observations=20000)
            result = risk_decomposition(cfg, model, empirical, FAMILY_SIZE, perfect_shift,
                                        "T2" if method == "joint" else "T1")
            terms = result["components"]
            self.assertEqual(result["theory"], {"T1":"Theorem 1", "T2":"Theorem 2", "C1":"Corollary 1"}[certificate])
            self.assertAlmostEqual(result["uncapped_risk_upper"], sum(terms.values()))
            self.assertAlmostEqual(result["confidence"]["failure_each_theorem_and_mc"], .0025)
            if regime == "perfect":
                self.assertEqual(terms["channel"], 0.)
                self.assertEqual(result["channel_shift"]["method"], "C1 second")
            if method == "joint":
                kl = result["posterior_kl"]
                expected = math.sqrt((.5/799 + .5/1999)*(kl + math.log(math.sqrt(800*2000)/.0025)))
                self.assertAlmostEqual(terms["joint"], expected)
                self.assertNotIn("sample", terms)
                self.assertNotIn("channel", terms)
            else:
                self.assertNotIn("joint", terms)
            if regime != "perfect":
                expected_shift = 0. if regime == "matched" else .28538650829
                self.assertAlmostEqual(terms["shift"], expected_shift)

    def test_mismatch_is_a_correlated_alternative(self):
        for rho in (0, .35, 1., float("nan")):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                learner_config(16, "t2_mismatch", rho)


if __name__ == "__main__":
    unittest.main()
