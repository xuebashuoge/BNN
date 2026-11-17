"""Frozen-checkpoint C1 inference, immutable protocol and cache validation."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from journal.c1_reference import lognormal_distance_benchmark, lognormal_logit_channel_terms
from journal.evaluate_c1_bayesian import (confidence, content_sha, derive_channels,
    evaluate_one, family_entries, fingerprint, prepare_protocol, run, validate_budget)
from journal.run_experiment import digest, load_data, make_models, normalize_config


class C1BayesianEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign, self.output = self.root / "campaign", self.root / "c1"
        self.campaign.mkdir()
        self.entries = family_entries()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          cwd=Path(__file__).resolve().parents[1]).strip()
        config = normalize_config(self.entries[0]["config"])
        train, validation, _, input_dim, classes, split = load_data(config, include_test=False)
        # Real IID training data and model, with a small held-out fixture to
        # keep the protocol test fast. No optimization or dataset download.
        self.data = train, validation, TensorDataset(*train[:16]), input_dim, classes, split
        torch.manual_seed(819)
        for entry in self.entries:
            config = normalize_config(entry["config"])
            checkpoint = self.campaign / entry["task_id"]
            checkpoint.mkdir()
            model, bank = make_models(config, 2, 2)
            metadata = dict(config=config, split=split, git_sha=revision, git_dirty=False,
                            training_channel_bank_hash=digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()}),
                            selected_epoch=1, total_seconds=1., peak_rss_mb=1.)
            self.write(checkpoint / "metadata.json", metadata)
            self.write(checkpoint / "task_config.json", config)
            self.write(checkpoint / "manifest_entry.json", entry)
            self.write(checkpoint / "history.json", [dict(epoch=epoch, validation=dict(accuracy=.7, cross_entropy=.5))
                       for epoch in range(1, config["epochs"] + 1)])
            torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def evaluate(self, index=0, **overrides):
        kwargs = dict(empirical_draws=4, test_draws=3, threads=1, index=index)
        kwargs.update(overrides)
        with patch("journal.evaluate_c1_bayesian.load_data", return_value=self.data):
            return run(self.campaign, self.output, **kwargs)

    def row_path(self, index=0):
        return self.output / "rows" / (self.entries[index]["task_id"] + ".json")

    def test_actual_frozen_gaussian_inference_never_constructs_optimizer(self):
        before = {str(path): fingerprint(path) for path in self.campaign.rglob("*") if path.is_file()}
        with patch("torch.optim.Adam", side_effect=AssertionError("no training")), \
                patch("torch.optim.SGD", side_effect=AssertionError("no training")):
            row = self.evaluate()
        self.assertTrue(row["no_training"])
        self.assertEqual(row["source_config"]["predictor"], "gaussian")
        self.assertEqual(len(row["empirical_replicates"]), 4)
        self.assertEqual(len(row["lognormal"]), 6)
        self.assertEqual(row["erasure"][0]["replicates"], row["lognormal"][0]["replicates"])
        for channel in row["lognormal"]:
            terms = lognormal_logit_channel_terms(channel["severity"])
            self.assertEqual(channel["shift_second_closed"], terms["c1_second_shift_upper"])
            self.assertFalse(channel["channel_guarantee_uses_distance_mc"])
            self.assertIsNone(channel["first_risk_upper"])
        self.assertEqual(before, {str(path): fingerprint(path) for path in self.campaign.rglob("*") if path.is_file()})

    def test_exact_resume_opens_no_data_and_rejects_changed_draws_or_budget(self):
        row = self.evaluate()
        with patch("journal.evaluate_c1_bayesian.load_data", side_effect=AssertionError("no new inference")):
            self.assertEqual(run(self.campaign, self.output, 4, 3, threads=1, index=0), row)
            for kwargs in (dict(empirical_draws=5), dict(delta=.025), dict(test_draws=4)):
                options = dict(empirical_draws=4, test_draws=3, threads=1, index=0)
                options.update(kwargs)
                with self.assertRaisesRegex(ValueError, "protocol"):
                    run(self.campaign, self.output, **options)

    def test_cache_rejects_tampering_even_with_recomputed_result_checksum(self):
        row = self.evaluate()
        changed = copy.deepcopy(row)
        changed["lognormal"][3]["risk_upper_uncapped"] -= .01
        self.write(self.row_path(), changed)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.evaluate()
        changed["result_sha256"] = content_sha({key: value for key, value in changed.items() if key != "result_sha256"})
        self.write(self.row_path(), changed)
        with self.assertRaisesRegex(ValueError, "inconsistent channel"):
            self.evaluate()

    def test_protocol_rejects_other_checkpoint_hash_or_diagnostic_changes(self):
        self.evaluate()
        other = self.campaign / self.entries[-1]["task_id"] / "history.json"
        original = other.read_text()
        other.write_text(original + "\n")
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.evaluate()
        other.write_text(original)
        diagnostic_path = self.output / "distance_benchmark.json"
        diagnostic = json.loads(diagnostic_path.read_text())
        diagnostic["unit_scale_mean_norm"] *= 1.01
        self.write(diagnostic_path, diagnostic)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.evaluate()

    def test_benchmark_creation_is_serialized_and_analytic_bounds_ignore_its_mean(self):
        budget = confidence(.05, 4, 3)
        with patch("journal.c1_reference.lognormal_distance_benchmark", wraps=lognormal_distance_benchmark) as sampler:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(prepare_protocol, self.entries, self.campaign, self.output, budget, 1)
                           for _ in range(2)]
                first, second = [future.result() for future in futures]
            self.assertEqual(sampler.call_count, 1)
        self.assertEqual(first, second)
        benchmark = first[1]
        altered = {**benchmark, "unit_scale_mean_norm": benchmark["unit_scale_mean_norm"] * 1.2}
        draws = [[dict(bounded_risk=.2, accuracy=.75, cross_entropy=.6)] for _ in range(6)]
        _, original_rows = derive_channels(normalize_config(self.entries[0]["config"]), .5, benchmark, draws, draws)
        _, altered_rows = derive_channels(normalize_config(self.entries[0]["config"]), .5, altered, draws, draws)
        for original, changed in zip(original_rows, altered_rows):
            self.assertEqual(original["risk_upper_uncapped"], changed["risk_upper_uncapped"])
            self.assertEqual(original["shift_second_closed"], changed["shift_second_closed"])
        self.assertNotEqual(original_rows[3]["shift_first_mc_diagnostic"], altered_rows[3]["shift_first_mc_diagnostic"])

    def test_family_config_bank_and_split_admissibility_are_enforced(self):
        bad_entry = copy.deepcopy(self.entries[0])
        bad_entry["config"]["predictor"] = "deterministic"
        with self.assertRaisesRegex(ValueError, "outside.*family"):
            evaluate_one(bad_entry, self.campaign, self.output, confidence(.05, 4, 3),
                         lognormal_distance_benchmark(), threads=1)
        metadata_path = self.campaign / self.entries[0]["task_id"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        original_metadata = copy.deepcopy(metadata)
        metadata["split"]["train_indices_hash"] = "wrong"
        self.write(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "splits"):
            self.evaluate()
        self.write(metadata_path, original_metadata)
        self.output = self.root / "invalid-bank"
        weights_path = metadata_path.with_name("weights.pt")
        state = torch.load(weights_path, weights_only=True)
        state["bank"]["gain"][0, 0] = .5
        torch.save(state, weights_path)
        with self.assertRaisesRegex(ValueError, "bank"):
            self.evaluate()

    def test_simultaneous_failure_budget_and_aggregate_coverage(self):
        budget = confidence(.05, 1000, 100)
        self.assertAlmostEqual(budget["data_failure_total"], .04)
        self.assertAlmostEqual(budget["empirical_mc_failure_total"], .01)
        self.assertAlmostEqual(12 * (budget["data_failure_per_run"] + budget["empirical_mc_failure_per_run"]), .05)
        with self.assertRaisesRegex(ValueError, "confidence allocation"):
            validate_budget({**budget, "empirical_mc_radius": 0.})
        self.evaluate()
        with self.assertRaisesRegex(ValueError, "every declared posterior"):
            self.evaluate(index=None, aggregate_only=True)
        for index in range(1, 12):
            self.evaluate(index=index)
        data = self.evaluate(index=None, aggregate_only=True)
        self.assertEqual(len(data["reports"]), 12)
        self.assertIn("no joint confidence", data["confidence_scope"])
        changed = json.loads(self.row_path(11).read_text())
        changed["data_complexity"] = 0.
        self.write(self.row_path(11), changed)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.evaluate(index=None, aggregate_only=True)


if __name__ == "__main__":
    unittest.main()
