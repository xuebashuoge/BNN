"""Fresh bank MC, frozen provenance, family coverage and safe certificate resume."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from journal.bayesian_bounds import allocation, corrected_bounds, fingerprint
from journal.build_bayesian_certificate import diagnostic_entries
from journal.evaluate_bayesian_certificate import SEED_OFFSET, declared_family, fresh_seed, run
from journal.run_experiment import (bounded_loss_description, digest, evaluate, load_data,
                                    make_models, normalize_config)


class BayesianCertificateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.output = self.root / "certificate"
        self.manifest = self.root / "manifest.jsonl"
        self.entries = diagnostic_entries()
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in self.entries))
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           cwd=Path(__file__).resolve().parents[1]).strip()
        _, _, _, _, _, split = load_data(normalize_config(self.entries[0]["config"]), include_test=False)
        self.rows = []
        for entry in self.entries:
            config = normalize_config(entry["config"])
            checkpoint = self.campaign / entry["task_id"]
            checkpoint.mkdir()
            model, bank = make_models(config, 2, 2)
            bank_hash = digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()})
            metadata = dict(config=config, split=split, git_sha=revision, git_dirty=False,
                            training_channel_bank_hash=bank_hash, selected_epoch=1,
                            total_seconds=1., peak_rss_mb=1.)
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            (checkpoint / "task_config.json").write_text(json.dumps(config))
            (checkpoint / "manifest_entry.json").write_text(json.dumps(entry))
            (checkpoint / "history.json").write_text(json.dumps([
                dict(epoch=epoch, validation=dict(accuracy=.7, cross_entropy=.5))
                for epoch in range(1, config["epochs"] + 1)]))
            torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
            empirical = dict(channel_source="bank", examples=512,
                             bounded_loss_definition=bounded_loss_description(config), draws=20,
                             bounded_loss=.2, accuracy=.7, cross_entropy=.5, predictor=dict(stochastic=True))
            heldout = {**empirical, "channel_source": "deployment", "examples": 10000,
                       "bounded_loss": .25, "accuracy": .75, "cross_entropy": .6}
            report = dict(checkpoint=str(checkpoint),
                evaluation_config={**config, "eval_seed": config["eval_seed"] + 1000003},
                split="test", risk_scope="full", empirical_gibbs_mc=empirical, heldout_gibbs_mc=heldout,
                signed_gap_decomposition_mc={metric: dict(channel_generalization=.01,
                    data_generalization=.02, channel_shift=gap - .03, sum=gap, observed_gap=gap)
                    for metric, gap in (("bounded_loss", .05), ("cross_entropy", .1))},
                checkpoint_source_fingerprints={key: fingerprint(checkpoint / filename) for key, filename in
                    (("metadata", "metadata.json"), ("weights", "weights.pt"))},
                checkpoint_verification=dict(saved_channel_bank_hash=bank_hash, bank_hash_matches_metadata=True,
                    split_hash_matches=dict(train_indices_hash=True, validation_indices_hash=True)))
            report_path = checkpoint / "test_full" / "evaluation.json"
            report_path.parent.mkdir()
            report_path.write_text(json.dumps(report))
            self.rows.append({**entry, "seed": config["seed"], "status": "complete", "test_status": "complete",
                              "test_report": str(report_path), "test_accuracy": .75, "test_cross_entropy": .6})
        self.write_results()

    def tearDown(self):
        self.temporary.cleanup()

    def write_results(self):
        (self.campaign / "test_results.json").write_text(json.dumps(self.rows))

    def evaluate_first(self, **overrides):
        return run(self.manifest, self.campaign, self.output, draws=4, threads=1, index=0, **overrides)

    def test_fresh_mc_uses_only_saved_training_bank_and_unweighted_corrected_bounds(self):
        checkpoint = self.campaign / self.entries[0]["task_id"]
        before = {str(path.relative_to(checkpoint)): fingerprint(path)
                  for path in checkpoint.rglob("*") if path.is_file()}
        with patch("journal.evaluate_bayesian_certificate.load_data", wraps=load_data) as loader, \
                patch("journal.evaluate_bayesian_certificate.evaluate", wraps=evaluate) as evaluator:
            result = self.evaluate_first()
        self.assertEqual(loader.call_count, 1)
        self.assertIs(loader.call_args.kwargs["include_test"], False)
        self.assertEqual(evaluator.call_args.kwargs["source"], "bank")
        self.assertEqual(evaluator.call_args.kwargs["draws"], 4)
        self.assertEqual(result["certificate_evaluation_seed"],
                         self.entries[0]["config"]["eval_seed"] + SEED_OFFSET)
        self.assertEqual(result["certificate_empirical_gibbs_mc"]["examples"], 512)
        self.assertEqual(result["full_test_draws"], 20)
        self.assertFalse(result["verification"]["test_data_opened_for_certificate"])
        expected = corrected_bounds(result["posterior_kl_float64"], result["config"],
                                    result["empirical_bounded_risk"], allocation(.025, 48, 4))
        self.assertEqual(result["theorem_1"], expected["theorem_1"])
        self.assertEqual(result["theorem_2"], expected["theorem_2"])
        self.assertAlmostEqual(result["confidence_allocation"]["union_failure"], .025)
        self.assertEqual(before, {str(path.relative_to(checkpoint)): fingerprint(path)
                                  for path in checkpoint.rglob("*") if path.is_file()})

    def test_exact_resume_opens_no_data_and_draw_or_budget_changes_are_rejected(self):
        original = self.evaluate_first()
        path = Path(original["certificate_report"])
        before = fingerprint(path)
        with patch("journal.evaluate_bayesian_certificate.evaluate", side_effect=AssertionError("no new MC")), \
                patch("journal.evaluate_bayesian_certificate.load_data", side_effect=AssertionError("no data reload")):
            self.assertEqual(self.evaluate_first(resume=True), original)
            with self.assertRaisesRegex(ValueError, "different draws, family, budget"):
                run(self.manifest, self.campaign, self.output, draws=8, threads=1, index=0, resume=True)
            with self.assertRaisesRegex(ValueError, "fixed failure budget"):
                run(self.manifest, self.campaign, self.output, draws=4, delta=.05, threads=1, index=0, resume=True)
        self.assertEqual(fingerprint(path), before)

    def test_incomplete_test_family_and_incomplete_training_prevent_all_mc(self):
        with patch("journal.evaluate_bayesian_certificate.evaluate", side_effect=AssertionError("no MC")):
            missing = self.rows.pop()
            self.write_results()
            with self.assertRaisesRegex(ValueError, "coverage"):
                self.evaluate_first()
            self.rows.append(missing)
            self.write_results()
            checkpoint = self.campaign / self.entries[-1]["task_id"]
            (checkpoint / "history.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "complete matching training"):
                self.evaluate_first()

    def test_stale_checkpoint_source_cannot_reuse_prior_mc(self):
        original = self.evaluate_first()
        checkpoint = self.campaign / self.entries[0]["task_id"]
        state = torch.load(checkpoint / "weights.pt", weights_only=True)
        state["model"]["mu"][0] += .1
        torch.save(state, checkpoint / "weights.pt")
        with patch("journal.evaluate_bayesian_certificate.evaluate", side_effect=AssertionError("no MC")):
            with self.assertRaisesRegex(ValueError, "source fingerprints"):
                self.evaluate_first(resume=True)
        self.assertEqual(json.loads(Path(original["certificate_report"]).read_text()), original)

    def test_corrupt_cached_metrics_are_rejected_and_aggregate_requires_all_rows(self):
        original = self.evaluate_first()
        path = Path(original["certificate_report"])
        cached = json.loads(path.read_text())
        cached["accuracy"] += .01
        path.write_text(json.dumps(cached))
        with self.assertRaisesRegex(ValueError, "full-test identity or metrics"):
            self.evaluate_first(resume=True)
        path.write_text(json.dumps(original))
        with patch("journal.evaluate_bayesian_certificate.evaluate", side_effect=AssertionError("no MC")):
            with self.assertRaisesRegex(ValueError, "all 48 complete"):
                run(self.manifest, self.campaign, self.output, draws=4, threads=1, aggregate_only=True)

    def test_stream_overlap_and_changed_training_source_prevent_new_mc(self):
        entry = self.entries[0]
        report_path = Path(self.rows[0]["test_report"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["eval_seed"] = entry["config"]["eval_seed"] + SEED_OFFSET
        report_path.write_text(json.dumps(report))
        with patch("journal.evaluate_bayesian_certificate.evaluate", side_effect=AssertionError("no MC")):
            with self.assertRaisesRegex(ValueError, "overlap validation or full-test"):
                self.evaluate_first()
        with self.assertRaisesRegex(ValueError, "overlap validation or full-test"):
            fresh_seed({**entry["config"], "epochs": 2, "validation_seed_policy": "fixed"},
                       {"eval_seed": entry["config"]["eval_seed"] + SEED_OFFSET - 1})
        with patch("journal.evaluate_bayesian_certificate.verify_training_sources",
                   side_effect=ValueError("training source changed")), \
                patch("journal.evaluate_bayesian_certificate.load_data", side_effect=AssertionError("no data")):
            with self.assertRaisesRegex(ValueError, "training source changed"):
                self.evaluate_first()

    def test_foreign_prescribed_hyperparameters_are_rejected(self):
        self.entries[0]["config"]["lr"] = .5
        self.rows[0]["config"]["lr"] = .5
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in self.entries))
        self.write_results()
        with self.assertRaisesRegex(ValueError, "prespecified training configuration"):
            declared_family(self.manifest, self.campaign)


if __name__ == "__main__":
    unittest.main()
