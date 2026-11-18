from argparse import Namespace
import math
from pathlib import Path
import unittest

from scipy.special import ndtr
import torch

from journal.c1_full_fading import log_gaussian_norm_moment_upper
from journal.full_fading_bounds import FAMILY_SIZE, case_path, design, risk_decomposition
from journal.run_experiment import make_models


class FullFadingBoundsTests(unittest.TestCase):
    def test_closed_moment_upper_bounds_exact_scalar_gaussian_cgf(self):
        for mean, std, t in ((0., 1., .04), (1., .5, .1), (-2., 3., 1.)):
            direct = .5 * (std*t)**2 + math.log(
                math.exp(mean*t) * ndtr(mean/std + std*t)
                + math.exp(-mean*t) * ndtr(-mean/std + std*t))
            upper = log_gaussian_norm_moment_upper(
                t, torch.tensor([mean], dtype=torch.float64),
                torch.tensor([std**2], dtype=torch.float64))
            self.assertGreaterEqual(upper, direct)

    def test_compact_confirmation_stays_separate_from_fixed_cohort(self):
        setup = dict(fallback=True, dataset="moons", family="fading", regime="perfect",
                     m=2000, device="cpu", threads=1, epochs=None)
        fixed = design(Namespace(**setup, fallback_profile="certificate"))
        confirmed = design(Namespace(**setup, fallback_profile="validation"))
        for key in ("seed", "channel_seed", "eval_seed", "data_seed"):
            self.assertNotEqual(fixed[key], confirmed[key])
        for key in ("prior_std", "width", "m", "deployment_channel"):
            self.assertEqual(fixed[key], confirmed[key])
        root = Path("results")
        self.assertNotEqual(case_path(root, "moons", "fading", "perfect", True),
                            case_path(root, "moons", "fading", "perfect", True, "validation"))
        model, _ = make_models(confirmed, input_dim=2, classes=2)
        empirical = dict(mean=.15, iid_observations=10000)
        original = risk_decomposition(confirmed, model, empirical)
        supplemental = risk_decomposition(confirmed, model, empirical, family_size=6)
        self.assertEqual(original["confidence"]["family_size"], FAMILY_SIZE)
        self.assertAlmostEqual(supplemental["confidence"]["failure_each_theorem_and_mc"], .05/12)
        self.assertEqual(supplemental["channel_shift"]["method"], "C1 second")
        self.assertAlmostEqual(supplemental["components"]["shift"], .5074008502666063)
        self.assertEqual(supplemental["components"]["channel"], 0.)
        self.assertEqual(supplemental["uncapped_risk_upper"], sum(supplemental["components"].values()))
