"""Confidence-family and frozen-source tests for all-method bound reporting."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from journal.bayesian_bounds import allocation, collect_trials, corrected_bounds, fingerprint, observation
from journal.run_experiment import bounded_loss_description, digest, load_data, make_models, normalize_config


class BayesianBoundsTests(unittest.TestCase):
    def test_joint_failure_budget_and_mc_convergence(self):
        budget = allocation(.05, 320, 25)
        self.assertAlmostEqual(budget["union_failure"], .05)
        self.assertAlmostEqual(budget["failure_per_theorem_run"], .04 / 640)
        self.assertAlmostEqual(budget["empirical_mc_upper_radius"],
                               2 * allocation(.05, 320, 100)["empirical_mc_upper_radius"])
        for args in ((.05, True, 10), (.05, 0, 10), (1., 2, 10), (.05, 2, 1.5)):
            with self.assertRaises(ValueError):
                allocation(*args)

    def test_perfect_channel_terms_collapse_and_shift_remains(self):
        config = normalize_config(dict(dataset="moons", predictor="gaussian", method="l2", width=8,
            train_size=512, m=1, train_channel={"kind": "perfect"},
            deployment_channel={"kind": "outage", "outage_prob": .1}))
        bound = corrected_bounds(1., config, .1, allocation(.05, 1, 10000))
        self.assertEqual(bound["channel_complexity"], 0.)
        self.assertEqual(bound["theorem_1"], bound["theorem_2"])
        self.assertAlmostEqual(bound["shift"]["channel_shift_complexity"], 1 - .9**8)

    def test_vacuity_is_visible_and_training_coefficients_do_not_scale_bounds(self):
        config = normalize_config(dict(dataset="moons", predictor="gaussian", train_size=32, m=2))
        first = corrected_bounds(100000., config, .2, allocation(.05, 2, 10))
        second = corrected_bounds(100000., {**config, "eta_channel": 0., "eta_data": 0., "l2": 0.},
                                  .2, allocation(.05, 2, 10))
        self.assertEqual(first, second)
        self.assertGreater(first["theorem_1"]["rhs_uncapped"], 1.)
        self.assertEqual(first["theorem_1"]["population_risk_upper"], 1.)
        self.assertFalse(first["theorem_1"]["nonvacuous"])

    def test_missing_seed_or_legacy_baseline_cannot_be_silently_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            entries = [dict(task_id=f"task_{seed}", config=dict(dataset="moons", predictor="gaussian", seed=seed))
                       for seed in (1, 2)]
            manifest.write_text("".join(json.dumps(item) + "\n" for item in entries))
            rows = [{**entries[0], "test_status": "complete"}]
            (root / "test_results.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "coverage"):
                collect_trials([(manifest, root)])
            rows.append({**entries[1], "test_status": "complete"})
            (root / "test_results.json").write_text(json.dumps(rows))
            self.assertEqual(len(collect_trials([(manifest, root)])), 2)
            for item in entries:
                item["config"]["predictor"] = "legacy"
            manifest.write_text("".join(json.dumps(item) + "\n" for item in entries))
            (root / "test_results.json").write_text(json.dumps([
                {**entry, "test_status": "complete"} for entry in entries]))
            with self.assertRaisesRegex(ValueError, "all-Bayesian"):
                collect_trials([(manifest, root)])

    def test_gaussian_erm_and_l2_receive_bounds_from_actual_saved_tensors(self):
        for method in ("erm", "l2"):
            config = normalize_config(dict(dataset="moons", predictor="gaussian", method=method,
                width=8, train_size=32, val_size=8, training_loss="bounded_ce_smooth", m=3))
            _, _, _, dimension, classes, split = load_data(config, include_test=False)
            model, bank = make_models(config, dimension, classes)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "trial"
                checkpoint.mkdir()
                metadata = dict(config=config, split=split,
                    training_channel_bank_hash=digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()}))
                (checkpoint / "metadata.json").write_text(json.dumps(metadata))
                torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
                empirical = dict(channel_source="bank", examples=32,
                                 bounded_loss_definition=bounded_loss_description(config), draws=2,
                                 bounded_loss=.2, predictor=dict(stochastic=True))
                heldout = {**empirical, "channel_source": "deployment", "accuracy": .7, "bounded_loss": .3}
                report = dict(checkpoint=str(checkpoint),
                              evaluation_config={**config, "eval_seed": config["eval_seed"] + 1000003, "eval_draws": 2},
                              split="test", risk_scope="full", empirical_gibbs_mc=empirical, heldout_gibbs_mc=heldout,
                              checkpoint_source_fingerprints={key: fingerprint(checkpoint / filename) for key, filename in
                                  (("metadata", "metadata.json"), ("weights", "weights.pt"))},
                              checkpoint_verification=dict(saved_channel_bank_hash=metadata["training_channel_bank_hash"],
                                  bank_hash_matches_metadata=True,
                                  split_hash_matches=dict(train_indices_hash=True, validation_indices_hash=True)))
                report_path = checkpoint / "evaluation.json"
                report_path.write_text(json.dumps(report))
                entry = dict(task_id="trial", group="moons", method_label=method, stage="confirm", config=config)
                result = observation(entry, {"test_report": str(report_path)}, root, 2, .05)
                self.assertTrue(math.isfinite(result["posterior_kl_float64"]))
                self.assertGreater(result["theorem_1"]["rhs_uncapped"], 0)
                with self.assertRaisesRegex(ValueError, "declared training configuration"):
                    observation({**entry, "config": {**config, "lr": .3}}, {"test_report": str(report_path)}, root, 2, .05)
                report["evaluation_config"]["eval_seed"] = config["eval_seed"]
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "fresh MC stream"):
                    observation(entry, {"test_report": str(report_path)}, root, 2, .05)
                report["evaluation_config"]["eval_seed"] += 1000003
                with torch.no_grad():
                    model.mu.add_(.1)
                torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "source fingerprints"):
                    observation(entry, {"test_report": str(report_path)}, root, 2, .05)
                bank.noise.add_(1)
                torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
                report["checkpoint_source_fingerprints"]["weights"] = fingerprint(checkpoint / "weights.pt")
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "bank hash"):
                    observation(entry, {"test_report": str(report_path)}, root, 2, .05)


if __name__ == "__main__":
    unittest.main()
