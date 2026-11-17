"""Check simultaneous confidence and finite-KL model conventions; no training."""
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from journal.evaluate_empirical_bank import bounds_from_empirical, confidence_allocation, evaluate_entry
from journal.run_experiment import digest, load_data, make_models, normalize_config
from journal.theorem1 import GaussianMLP


class EmpiricalBankTests(unittest.TestCase):
    def test_joint_budget_covers_every_mc_and_both_theorem_events(self):
        result = confidence_allocation(.05, 51, 10)
        self.assertAlmostEqual(result["posterior_failure_per_theorem_run"], .04 / (2 * 51))
        self.assertAlmostEqual(result["monte_carlo_failure_per_run"], .01 / 51)
        self.assertAlmostEqual(result["posterior_union_failure"] + result["monte_carlo_union_failure"], .05)
        self.assertAlmostEqual(result["empirical_mc_upper_radius"], math.sqrt(math.log(5100) / 20))
        self.assertAlmostEqual(confidence_allocation(.05, 51, 40)["empirical_mc_upper_radius"],
                               result["empirical_mc_upper_radius"] / 2)

    def test_deterministic_predictor_never_receives_gaussian_kl_certificate(self):
        model = GaussianMLP(2, 8, 2, stochastic=False)
        with mock.patch("journal.evaluate_empirical_bank.gaussian_kl", side_effect=AssertionError("Dirac KL is not Gaussian KL")):
            result = bounds_from_empirical(model, {}, {}, {})
        self.assertIsNone(result["theorem_1"])
        self.assertIsNone(result["theorem_2"])
        self.assertEqual(result["final_posterior_complexities"], {})

    def test_perfect_gaussian_posterior_has_no_channel_complexity_and_restores_shift(self):
        model = GaussianMLP(2, 8, 2, stochastic=True, posterior_std=.2)
        config = normalize_config(dict(dataset="moons", width=8, train_size=512,
            train_channel=dict(kind="perfect"),
            deployment_channel=dict(kind="outage", outage_prob=.1)))
        allocation = confidence_allocation(.05, 1, 100)
        result = bounds_from_empirical(model, config, dict(bounded_loss=.1), allocation)
        complexity = result["final_posterior_complexities"]
        self.assertEqual(complexity["channel_complexity"], 0.)
        self.assertAlmostEqual(complexity["sample_complexity"], complexity["joint_complexity"])
        expected = min(1., .1 + allocation["empirical_mc_upper_radius"]) + (1-.9**8) + complexity["sample_complexity"]
        self.assertAlmostEqual(result["theorem_1"]["population_risk_upper_uncapped"], expected)
        self.assertEqual(result["theorem_1"], result["theorem_2"])

    def test_invalid_counts_do_not_silently_change_confidence(self):
        for args in ((0., 1, 10), (.05, 0, 10), (.05, 2.5, 10), (.05, 2, 0), (.05, True, 10)):
            with self.assertRaises(ValueError):
                confidence_allocation(*args)

    def test_frozen_checkpoint_evaluation_preserves_sources_and_never_loads_test_data(self):
        config = normalize_config(dict(dataset="moons", width=8, train_size=32,
            val_size=8, epochs=1, m=3, batch_size=16, training_loss="bounded_ce_smooth"))
        _, _, _, input_dim, classes, split = load_data(config, include_test=False)
        model, bank = make_models(config, input_dim, classes)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "frozen_s0"
            checkpoint.mkdir()
            metadata = dict(config=config, split=split, selected_epoch=1,
                training_channel_bank_hash=digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()}))
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
            original = {name: (checkpoint / name).read_bytes() for name in ("metadata.json", "weights.pt")}
            with mock.patch("journal.evaluate_empirical_bank.load_data", wraps=load_data) as loader:
                report = evaluate_entry(dict(task_id=checkpoint.name, checkpoint=str(checkpoint),
                    output=str(checkpoint / "empirical_bank"), seed=0),
                    draws=2, delta=.05, total_runs=51, threads=1)
            loader.assert_called_once()
            self.assertFalse(loader.call_args.kwargs["include_test"])
            self.assertEqual(report["empirical_gibbs_mc"]["channel_source"], "bank")
            self.assertEqual(report["empirical_gibbs_mc"]["examples"], 32)
            self.assertAlmostEqual(report["empirical_gibbs_mc"]["zero_one_loss"],
                                   1-report["empirical_gibbs_mc"]["accuracy"])
            self.assertTrue(report["no_training"])
            self.assertTrue((checkpoint / "empirical_bank" / "evaluation.json").is_file())
            self.assertEqual(original, {name: (checkpoint / name).read_bytes() for name in original})


if __name__ == "__main__":
    unittest.main()
