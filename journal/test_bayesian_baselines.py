"""Common Gaussian predictors, L2 moments, and legacy checkpoint compatibility."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from journal.certify_checkpoint import certify
from journal.run_experiment import (complexities, digest, evaluate_checkpoint, make_models,
                                    normalize_config, training_penalty,
                                    uses_gaussian_predictor)
from journal.run_task import require_cpu_pilots


class BayesianBaselineTests(unittest.TestCase):
    def config(self, **overrides):
        return normalize_config({"dataset": "moons", "width": 8,
                                 "train_size": 8, "val_size": 4, "batch_size": 4,
                                 "epochs": 1, "m": 4, "eval_draws": 2,
                                 "predictor": "gaussian", **overrides})

    def test_predictor_choice_is_explicit_and_backwards_compatible(self):
        self.assertEqual(normalize_config({})["predictor"], "legacy")
        for invalid in ("bayesian", "deterministic", "", None, True, {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config({"predictor": invalid})
        for method in ("erm", "l2", "stochastic_erm", "pac_bayes", "proposed", "joint"):
            with self.subTest(method=method):
                gaussian, _ = make_models(self.config(method=method), 2, 2)
                self.assertTrue(gaussian.stochastic)
                self.assertTrue(gaussian.rho.requires_grad)
                legacy_config = self.config(method=method, predictor="legacy")
                legacy_config.pop("predictor")  # Config saved before this field existed.
                legacy, _ = make_models(legacy_config, 2, 2)
                self.assertEqual(legacy.stochastic, method not in {"erm", "l2"})
                self.assertEqual(uses_gaussian_predictor(legacy_config), legacy.stochastic)

    def test_all_four_algorithms_share_initial_posterior_and_bank(self):
        reference = None
        for method in ("erm", "l2", "proposed", "joint"):
            with self.subTest(method=method):
                config = self.config(method=method)
                model, bank = make_models(config, 2, 2)
                state = (model.mu.detach().clone(), model.rho.detach().clone(), bank.gain.clone(), bank.noise.clone())
                if reference is None:
                    reference = state
                for actual, expected in zip(state, reference):
                    self.assertTrue(torch.equal(actual, expected))
                first = model.sample(torch.Generator().manual_seed(5))
                second = model.sample(torch.Generator().manual_seed(6))
                self.assertFalse(torch.equal(first, second))
                terms = complexities(model, config)
                self.assertEqual(set(terms), {"kl", "channel_complexity", "sample_complexity", "joint_complexity"})
                self.assertTrue(all(torch.isfinite(value) and value >= 0 for value in terms.values()))

    def test_gaussian_l2_penalizes_both_mean_and_variance(self):
        config = self.config(method="l2", l2=.03, posterior_std=.2)
        model, _ = make_models(config, 2, 2)
        penalty = training_penalty(model, config)
        expected = config["l2"] * (model.mu.square().sum() + F.softplus(model.rho).square().sum())
        self.assertTrue(torch.allclose(penalty, expected))
        penalty.backward()
        self.assertTrue(torch.allclose(model.mu.grad, 2 * config["l2"] * model.mu))
        self.assertTrue(torch.all(model.rho.grad > 0))
        deterministic, _ = make_models({**config, "predictor": "legacy"}, 2, 2)
        self.assertTrue(torch.equal(training_penalty(deterministic, config),
                                    config["l2"] * deterministic.mu.square().sum()))
        for method in ("erm", "stochastic_erm"):
            self.assertEqual(training_penalty(model, {**config, "method": method}).item(), 0.)

    def test_gaussian_erm_empirical_loss_trains_posterior_scale(self):
        config = self.config(method="erm")
        model, bank = make_models(config, 2, 2)
        model.train()
        x = torch.tensor([[.1, .8], [.7, -.3], [1.2, .2], [-.2, .5]])
        logits = model(x, model.sample(torch.Generator().manual_seed(10)), bank,
                       "bank", generator=torch.Generator().manual_seed(11))
        F.cross_entropy(logits, torch.tensor([0, 1, 1, 0])).backward()
        self.assertIsNotNone(model.rho.grad)
        self.assertTrue(torch.isfinite(model.rho.grad).all())
        self.assertGreater(model.rho.grad.abs().sum().item(), 0.)

    def test_checkpoint_evaluation_preserves_legacy_and_gaussian_families(self):
        data = TensorDataset(torch.tensor([[.1, .8], [.7, -.3], [1.2, .2], [-.2, .5]]),
                             torch.tensor([0, 1, 1, 0]))
        split = {"train_indices_hash": "fixed-training", "validation_indices_hash": "fixed-validation"}
        for predictor in ("legacy", "gaussian"):
            with self.subTest(predictor=predictor), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = self.config(method="erm", predictor=predictor)
                model, bank = make_models(config, 2, 2)
                saved_config = dict(config)
                if predictor == "legacy":
                    saved_config.pop("predictor")
                metadata = {"config": saved_config, "split": dict(split), "selected_epoch": 1,
                            "certificate_note": "test checkpoint"}
                if predictor == "gaussian":
                    metadata["training_channel_bank_hash"] = digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()})
                else:
                    metadata["split"].pop("validation_indices_hash")
                (root / "metadata.json").write_text(json.dumps(metadata))
                torch.save({"model": model.state_dict(), "bank": bank.state_dict()}, root / "weights.pt")
                with patch("journal.run_experiment.load_data", return_value=(data, data, data, 2, 2, split)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_checkpoint(root, None, root / "evaluation", "validation", "full")
                stochastic = predictor == "gaussian"
                self.assertEqual(result["heldout_gibbs_mc"]["predictor"]["stochastic"], stochastic)
                terms = result["final_posterior_complexities"]
                self.assertEqual(bool(terms), stochastic)
                self.assertTrue(all(math.isfinite(value) for value in terms.values()))
                self.assertAlmostEqual(result["signed_gap_decomposition_mc"]["bounded_loss"]["sum"],
                                       result["bounded_gap_mc"])
                self.assertEqual(result["checkpoint_verification"]["bank_hash_matches_metadata"],
                                 True if stochastic else None)

    def test_checkpoint_evaluation_detects_validation_split_or_bank_tampering(self):
        data = TensorDataset(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))
        split = {"train_indices_hash": "training", "validation_indices_hash": "validation"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(method="l2")
            model, bank = make_models(config, 2, 2)
            state = {"model": model.state_dict(), "bank": bank.state_dict()}
            torch.save(state, root / "weights.pt")
            metadata = {"config": config, "split": dict(split), "certificate_note": "test",
                        "training_channel_bank_hash": digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()})}
            metadata["split"]["validation_indices_hash"] = "changed"
            (root / "metadata.json").write_text(json.dumps(metadata))
            with patch("journal.run_experiment.load_data", return_value=(data, data, data, 2, 2, split)):
                with self.assertRaisesRegex(ValueError, "validation_indices_hash"):
                    evaluate_checkpoint(root, None, root / "evaluation", "validation")
                metadata["split"] = dict(split)
                (root / "metadata.json").write_text(json.dumps(metadata))
                state["bank"]["gain"][0, 0] += 1
                torch.save(state, root / "weights.pt")
                with self.assertRaisesRegex(ValueError, "recorded bank hash"):
                    evaluate_checkpoint(root, None, root / "evaluation", "validation")

    def test_certificate_family_gate_accepts_gaussian_erm_and_l2(self):
        for method in ("erm", "l2"):
            for predictor in ("legacy", "gaussian"):
                with self.subTest(method=method, predictor=predictor), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    config = self.config(method=method, predictor=predictor,
                                         training_loss="bounded_ce_smooth")
                    (root / "metadata.json").write_text(json.dumps({"config": config, "git_sha": "test"}))
                    with patch("journal.certify_checkpoint.verify_sampler_source",
                               side_effect=RuntimeError("family gate passed")) as verify:
                        if predictor == "gaussian":
                            with self.assertRaisesRegex(RuntimeError, "family gate passed"):
                                certify(root, root / "certificate", 2, .05, 1, 1, 1)
                            verify.assert_called_once()
                        else:
                            with self.assertRaisesRegex(ValueError, "deterministic Dirac"):
                                certify(root, root / "certificate", 2, .05, 1, 1, 1)
                            verify.assert_not_called()

    def test_deterministic_pilot_cannot_cover_gaussian_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_config = self.config(method="erm", predictor="legacy")
            old_config.pop("predictor")
            (root / "metadata.json").write_text(json.dumps({"config": old_config,
                "total_seconds": 1., "peak_rss_mb": 1., "selected_epoch": 1}))
            (root / "history.json").write_text(json.dumps([
                {"epoch": 1, "validation": {"cross_entropy": .5}}]))
            (root / "weights.pt").write_bytes(b"existence sentinel")
            legacy = {"task_id": "legacy", "config": old_config}
            self.assertEqual(len(require_cpu_pilots([legacy], root)), 1)
            gaussian = {"task_id": "gaussian", "config": {**old_config, "predictor": "gaussian"}}
            with self.assertRaisesRegex(ValueError, "No completed matching CPU pilot"):
                require_cpu_pilots([gaussian], root)


if __name__ == "__main__":
    unittest.main()
