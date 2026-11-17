"""Check the fixed all-method toy certificate declaration without training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal.build_bayesian_certificate import SEEDS, diagnostic_entries, write_manifest
from journal.build_bayesian_campaign import CHANNELS, SEEDS as MAIN_SEEDS
from journal.campaign import validate_entries
from journal.run_experiment import make_models, normalize_config


class SmallBayesianCertificateTests(unittest.TestCase):
    def test_complete_paired_fixed_matrix_and_separate_scientific_scope(self):
        entries = diagnostic_entries()
        self.assertEqual(len(entries), 48)
        validate_entries(entries)
        self.assertEqual(len({item["task_id"] for item in entries}), 48)
        counts = Counter((item["group"], item["method_label"]) for item in entries)
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {3})
        self.assertFalse(set(SEEDS) & {seed for seeds in MAIN_SEEDS.values() for seed in seeds})
        for item in entries:
            config = normalize_config(item["config"])
            self.assertEqual(config["predictor"], "gaussian")
            self.assertEqual(config["dataset"], "moons")
            self.assertEqual(config["moons_sampler"], "iid")
            self.assertEqual((config["width"], config["train_size"], config["val_size"], config["m"]),
                             (8, 512, 512, 2000))
            self.assertEqual((config["epochs"], config["lr"], config["posterior_std"],
                              config["prior_std"], config["rho_lr_scale"]), (150, .003, .1, 1., .1))
            self.assertEqual(config["eval_draws"], 20)
            self.assertEqual(config["l2"], 1e-4)
            self.assertEqual(config["channel_seed"], config["seed"] + 10000)
            self.assertEqual(config["eval_seed"], config["seed"] + 20000)
            self.assertEqual(item["experiment"], "main")
            self.assertEqual(item["scientific_setup"], "small_model_bound_diagnostic")
            self.assertFalse(item["main_accuracy_table"])
            self.assertFalse(item["hyperparameter_search"])
            law = CHANNELS[item["channel_name"]]
            self.assertEqual(config["deployment_channel"], law)
            self.assertEqual(config["train_channel"], {"kind": "perfect"} if
                             item["training_regime"] == "perfect" else law)
            if config["method"] == "proposed":
                self.assertEqual((config["eta_channel"], config["eta_data"]), (.01, .03))
            if config["method"] == "joint":
                self.assertEqual(config["eta_data"], .03)

    def test_all_algorithms_share_same_gaussian_initial_model_and_bank(self):
        # Model/bank construction uses only synthetic RNG, no data loaders or
        # optimization. Pairing must survive differences in algorithm labels.
        entries = diagnostic_entries()
        paired = defaultdict(list)
        for item in entries:
            paired[(item["channel_name"], item["training_regime"], item["config"]["seed"])].append(item)
        import torch
        for key, items in paired.items():
            model0, bank0 = make_models(normalize_config(items[0]["config"]), 2, 2)
            self.assertTrue(model0.stochastic)
            self.assertEqual(model0.d, 42)
            for item in items[1:]:
                model, bank = make_models(normalize_config(item["config"]), 2, 2)
                self.assertTrue(model.stochastic)
                self.assertTrue(torch.equal(model.mu, model0.mu), key)
                self.assertTrue(torch.equal(model.rho, model0.rho), key)
                self.assertTrue(torch.equal(bank.gain, bank0.gain), key)
                self.assertTrue(torch.equal(bank.noise, bank0.noise), key)

    def test_manifest_generation_never_opens_data_and_records_hash(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("journal.run_experiment.load_data", side_effect=AssertionError("Data access forbidden")):
            output = Path(directory, "certificate.jsonl")
            plan = write_manifest(output)
            self.assertEqual(plan["tasks"], 48)
            self.assertEqual(plan["seeds"], list(SEEDS))
            self.assertFalse(plan["test_based_selection"])
            self.assertFalse(plan["hyperparameter_search"])
            self.assertEqual(plan["manifest_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(plan, json.loads(output.with_suffix(".plan.json").read_text()))


if __name__ == "__main__":
    unittest.main()
