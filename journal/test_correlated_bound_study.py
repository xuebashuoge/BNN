import math
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from journal.correlated_bound_study import CASES, FAMILY_SIZE, JOINT_FAMILY_SIZE, learner_config, run
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

    def test_erasure_learners_and_joint_confidence(self):
        self.assertEqual(JOINT_FAMILY_SIZE, 20)
        for width in (8, 16):
            for case, (regime, method, certificate) in CASES.items():
                with self.subTest(width=width, case=case):
                    cfg = learner_config(width, case, family="erasure")
                    self.assertEqual(cfg["deployment_channel"], {"kind": "outage", "outage_prob": .5})
                    expected_train = ({"kind": "perfect"} if regime == "perfect" else
                                      {"kind": "outage", "outage_prob": .5 if regime == "matched" else .45})
                    self.assertEqual(cfg["train_channel"], expected_train)
                    self.assertEqual(cfg["method"], method)
                    with tempfile.TemporaryDirectory() as folder, patch(
                            "journal.correlated_bound_study.evaluate_case") as evaluate:
                        run(Path(folder), width, case, family="erasure", both_channels=True)
                    args, kwargs = evaluate.call_args
                    self.assertEqual(args[0], cfg)
                    self.assertEqual(args[1], Path(folder) / "erasure" / f"d{width}" / case)
                    self.assertEqual(args[5], JOINT_FAMILY_SIZE)
                    self.assertEqual(args[6]["certificate"], certificate)
                    self.assertEqual(kwargs["stochastic_theorem"], "T2" if method == "joint" else "T1")
                    model, _ = make_models(cfg, 2, 2)
                    result = risk_decomposition(cfg, model, {"mean": .15, "iid_observations": 20000},
                                                JOINT_FAMILY_SIZE, perfect_shift, kwargs["stochastic_theorem"])
                    self.assertEqual(result["confidence"]["failure_each_theorem_and_mc"], .00125)
                    if regime == "perfect":
                        self.assertEqual(result["channel_shift"]["method"], "C1 first")
                        distance = sum(math.comb(width,k)*math.sqrt(k) for k in range(width+1))/2**width
                        shift = math.sqrt(2)*.075/math.log(100)*distance
                    else:
                        shift = math.sqrt(-width*math.log(.99)/4) if regime == "mismatch" else 0.
                    self.assertAlmostEqual(result["components"]["shift"], shift)

    def test_mismatch_is_a_correlated_alternative(self):
        for rho in (0, .35, 1., float("nan")):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                learner_config(16, "t2_mismatch", rho)


if __name__ == "__main__":
    unittest.main()
