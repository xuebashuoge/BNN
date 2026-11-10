"""Small mathematical/protocol checks; no dataset downloads or training."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.distributions import Independent, Normal, kl_divergence
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from journal.theorem1 import ChannelBank, GaussianMLP, gaussian_kl, theorem1_terms, joint_complexity
from journal.run_experiment import DEFAULTS, empirical_objective, evaluate, read_config, sample_moons_iid
from journal.run_task import read_manifest, require_cpu_pilots


class TheoremProtocolTests(unittest.TestCase):
    def test_iid_moons_is_reproducible_and_unconstrained_by_class_counts(self):
        x, y = sample_moons_iid(101, .3, 31)
        x_again, y_again = sample_moons_iid(101, .3, 31)
        self.assertEqual(x.shape, (101, 2))
        self.assertEqual(y.shape, (101,))
        self.assertTrue((x == x_again).all() and (y == y_again).all())
        self.assertEqual(set(y.tolist()), {0, 1})
        other_x, other_y = sample_moons_iid(101, .3, 32)
        self.assertFalse((x == other_x).all())
        self.assertFalse((y == other_y).all())

    def test_bounded_training_loss_has_correct_units_and_gradient(self):
        logits = torch.tensor([[2., -2.], [-10., 10.]], requires_grad=True)
        targets = torch.tensor([0, 0])
        raw = empirical_objective(logits, targets, {**DEFAULTS, "training_loss": "raw_ce"})
        torch.testing.assert_close(raw, F.cross_entropy(logits, targets))
        bounded = empirical_objective(logits, targets,
                                      {**DEFAULTS, "training_loss": "bounded_ce", "clip_ce": 2.})
        self.assertGreaterEqual(bounded.item(), 0.)
        self.assertLessEqual(bounded.item(), 1.)
        bounded.backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(2))
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.)

    def test_kl_matches_distribution_reference_and_gradient(self):
        mu = torch.tensor([.2, -.4, .9], dtype=torch.float64, requires_grad=True)
        rho = torch.tensor([-.8, .3, 1.1], dtype=torch.float64, requires_grad=True)
        expected = kl_divergence(Independent(Normal(mu, F.softplus(rho)), 1),
                                 Independent(Normal(torch.zeros_like(mu), torch.full_like(mu, 1.7)), 1))
        actual = gaussian_kl(mu, rho, 1.7)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(mu.grad).all() and torch.isfinite(rho.grad).all())

    def test_separate_formula_and_confidence(self):
        kl = torch.tensor(2., dtype=torch.float64)
        terms = theorem1_terms(kl, n=101, m=21, delta=.05)
        self.assertAlmostEqual(terms["channel_complexity"].item(), math.sqrt(.5 / 20 * (2 + math.log(math.sqrt(21) / .025))))
        self.assertAlmostEqual(terms["sample_complexity"].item(), math.sqrt(.5 / 100 * (2 + math.log(math.sqrt(101) / .025))))
        self.assertGreater(theorem1_terms(kl * 0, 101, 21)["sample_complexity"].item(), 0)

    def test_perfect_case_has_no_channel_dependence(self):
        kl = torch.tensor(3., dtype=torch.float64)
        small = theorem1_terms(kl, n=81, m=1, perfect_training=True)
        large = theorem1_terms(kl, n=81, m=1000, perfect_training=True)
        self.assertEqual(small["channel_complexity"].item(), 0.)
        torch.testing.assert_close(small["sample_complexity"], large["sample_complexity"])
        self.assertAlmostEqual(small["sample_complexity"].item(), math.sqrt(.5 / 80 * (3 + math.log(9 / .05))))
        torch.testing.assert_close(joint_complexity(kl, 81, 1, perfect_training=True), small["sample_complexity"])

    def test_bank_is_fixed_seeded_and_cartesian_sampling_unbiased(self):
        bank = ChannelBank(2, 3, {"kind": "gaussian"}, 123)
        gain, noise = bank.gain.clone(), bank.noise.clone()
        identical = ChannelBank(2, 3, {"kind": "gaussian"}, 123)
        different = ChannelBank(2, 3, {"kind": "gaussian"}, 124)
        torch.testing.assert_close(gain, identical.gain)
        self.assertFalse(torch.equal(gain, different.gain))
        # Each data row repeats equally. Independent sampled bank indices then
        # approximate the exact average over all data/channel pairs.
        data = torch.tensor([[.2, .5], [.8, -.4]])
        exact = (data[:, None, :] * gain[None, :, :] + noise[None, :, :]).square().mean()
        estimate = bank.apply(data.repeat(30000, 1), generator=torch.Generator().manual_seed(31)).square().mean()
        self.assertLess(abs(exact.item() - estimate.item()), .004)
        torch.testing.assert_close(bank.gain, gain)
        torch.testing.assert_close(bank.noise, noise)

    def test_weight_is_fixed_across_complete_inference_replicate(self):
        class SpyMLP(GaussianMLP):
            def __init__(self):
                super().__init__(2, 4, 2)
                self.samples = 0
                self.weight_calls = []
            def sample(self, generator=None):
                self.samples += 1
                return super().sample(generator)
            def forward(self, x, theta, *args, **kwargs):
                self.weight_calls.append(theta.clone())
                return super().forward(x, theta, *args, **kwargs)
        model = SpyMLP()
        bank = ChannelBank(4, 5, {"kind": "perfect"}, 2)
        loader = DataLoader(TensorDataset(torch.ones(7, 2), torch.zeros(7, dtype=torch.long)), batch_size=3)
        config = {**DEFAULTS, "device": "cpu", "deployment_channel": {"kind": "perfect"}}
        stats = evaluate(model, bank, loader, config, draws=4, seed=101)
        self.assertEqual(model.samples, 4)
        self.assertEqual(len(model.weight_calls), 12)
        for start in range(0, 12, 3):
            torch.testing.assert_close(model.weight_calls[start], model.weight_calls[start + 1])
            torch.testing.assert_close(model.weight_calls[start], model.weight_calls[start + 2])
        self.assertGreaterEqual(stats["bounded_loss"], 0.)
        self.assertLessEqual(stats["bounded_loss"], 1.)

    def test_invalid_configs_and_duplicate_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"m": 1, "train_channel": {"kind": "outage"}}))
            with self.assertRaises(ValueError):
                read_config(str(path))
            path.write_text(json.dumps({"m": 1, "train_channel": {"kind": "perfect"}}))
            self.assertEqual(read_config(str(path))["m"], 1)
            row = json.dumps({"task_id": "same", "config": {}})
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_gpu_guard_rejects_incomplete_or_different_cpu_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**DEFAULTS, "dataset": "moons", "train_size": 800, "val_size": 200,
                      "width": 64, "epochs": 2, "device": "cpu"}
            item = {"task_id": "gpu_trial", "config": {**config, "device": "cuda", "epochs": 150}}
            meta = {"config": config, "selected_epoch": 2}
            (root / "metadata.json").write_text(json.dumps(meta))
            (root / "weights.pt").write_bytes(b"synthetic existence check")
            (root / "history.json").write_text(json.dumps([
                {"epoch": 1, "validation": {"cross_entropy": .8}},
                {"epoch": 2, "validation": {"cross_entropy": .7}}]))
            with self.assertRaises(ValueError):
                require_cpu_pilots([item], root)
            meta.update(total_seconds=1.2, peak_rss_mb=600)
            (root / "metadata.json").write_text(json.dumps(meta))
            self.assertEqual(require_cpu_pilots([item], root)[0]["task_id"], "gpu_trial")
            item["config"]["width"] = 256
            with self.assertRaises(ValueError):
                require_cpu_pilots([item], root)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
