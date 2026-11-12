"""Optimizer-family checks using synthetic tensors; no optimizer updates."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from journal.campaign import completed_trial
from journal.run_experiment import make_models, make_optimizer, normalize_config
from journal.theorem1 import gaussian_kl


class PosteriorOptimizerTests(unittest.TestCase):
    def config(self, **overrides):
        return normalize_config(dict(dataset="moons", architecture="mlp", width=8,
                                     train_size=512, val_size=128, epochs=1, lr=.003,
                                     **overrides))

    def test_scale_validation_and_legacy_default(self):
        self.assertEqual(normalize_config({})["rho_lr_scale"], 1.)
        for value in (-1, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config(dict(rho_lr_scale=value))

    def test_default_keeps_single_group_and_both_trainable(self):
        config = self.config()
        model, _ = make_models(config, 2, 2)
        optimizer = make_optimizer(model, config)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], .003)
        self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]},
                         {id(model.mu), id(model.rho)})

    def test_slow_scale_uses_independent_rate(self):
        config = self.config(rho_lr_scale=.03)
        model, _ = make_models(config, 2, 2)
        groups = make_optimizer(model, config).param_groups
        self.assertEqual(len(groups), 2)
        self.assertIs(groups[0]["params"][0], model.mu)
        self.assertIs(groups[1]["params"][0], model.rho)
        self.assertAlmostEqual(groups[0]["lr"], .003)
        self.assertAlmostEqual(groups[1]["lr"], .00009)

    def test_fixed_scale_retains_gaussian_sampling_and_finite_kl(self):
        config = self.config(rho_lr_scale=0.)
        model, _ = make_models(config, 2, 2)
        self.assertTrue(model.stochastic)
        self.assertFalse(model.rho.requires_grad)
        self.assertTrue(model.mu.requires_grad)
        a = model.sample(torch.Generator().manual_seed(2))
        b = model.sample(torch.Generator().manual_seed(3))
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.isfinite(gaussian_kl(model.mu, model.rho)))
        parameters = make_optimizer(model, config).param_groups[0]["params"]
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], model.mu)

    def test_old_checkpoints_normalize_to_default_scale(self):
        config = self.config()
        old = {key: value for key, value in config.items() if key != "rho_lr_scale"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(json.dumps(dict(
                config=old, selected_epoch=1, total_seconds=1., peak_rss_mb=1., git_sha="old")))
            (root / "history.json").write_text(json.dumps([
                dict(epoch=1, validation=dict(accuracy=.5, cross_entropy=1.))]))
            (root / "weights.pt").write_bytes(b"synthetic existence sentinel")
            self.assertIsNotNone(completed_trial(root, config))
            self.assertIsNone(completed_trial(root, {**config, "rho_lr_scale": .1}))


if __name__ == "__main__":
    unittest.main()
