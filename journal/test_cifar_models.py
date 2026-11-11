"""Synthetic-tensor CIFAR model checks; no dataset access or training."""

import math
import unittest

import torch
from torch.nn import functional as F

from journal.cifar_models import GaussianSplitCNN
from journal.theorem1 import ChannelBank, gaussian_kl


class CifarModelTests(unittest.TestCase):
    def test_parameter_layout_is_complete_and_uses_views(self):
        for width, count in ((64, 89834), (128, 156074)):
            model = GaussianSplitCNN(width=width)
            self.assertEqual(model.d, count)
            parts = model.split(model.mu)
            self.assertEqual(sum(x.numel() for x in parts), count)
            self.assertEqual(tuple(x.shape for x in parts), model.parameter_shapes)
            pointer = model.mu.untyped_storage().data_ptr()
            self.assertTrue(all(x.untyped_storage().data_ptr() == pointer for x in parts))
            for bias in parts[1::2]:
                torch.testing.assert_close(bias, torch.zeros_like(bias))
        with self.assertRaises(ValueError):
            model.split(torch.ones(model.d + 1))

    def test_deterministic_limit_and_perfect_channel(self):
        torch.manual_seed(11)
        model = GaussianSplitCNN(width=64, stochastic=False)
        bank = ChannelBank(64, 5, {"kind": "perfect"}, 21)
        inputs = torch.randn(3, 3, 32, 32)
        model.eval()
        self.assertIs(model.sample(), model.mu)
        self.assertFalse(model.rho.requires_grad)
        perfect = model(inputs, model.sample(), bank, source="perfect")
        bank_output = model(inputs, model.sample(), bank, generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(perfect, bank_output)
        self.assertEqual(perfect.shape, (3, 10))
        self.assertTrue(torch.isfinite(perfect).all())
        self.assertTrue(math.isfinite(model.last_feature_power))
        self.assertGreaterEqual(model.last_feature_power, 0.)

    def test_stochastic_reparameterization_and_finite_gradients(self):
        torch.manual_seed(12)
        model = GaussianSplitCNN(width=64, posterior_std=.003)
        bank = ChannelBank(64, 5, {"kind": "gaussian", "gain_mean": .5,
                                  "gain_std": 1., "noise_std": 1.}, 21)
        theta = model.sample(torch.Generator().manual_seed(41))
        repeated = model.sample(torch.Generator().manual_seed(41))
        torch.testing.assert_close(theta, repeated)
        self.assertFalse(torch.equal(theta, model.sample(torch.Generator().manual_seed(42))))
        output = model(torch.randn(2, 3, 32, 32), theta, bank,
                       generator=torch.Generator().manual_seed(7))
        # Mathematical differentiation check only: no optimizer or update.
        loss = F.cross_entropy(output, torch.tensor([1, 4]))
        ce_gradients = torch.autograd.grad(loss, (model.mu, model.rho), retain_graph=True)
        self.assertTrue(all(torch.isfinite(value).all() for value in ce_gradients))
        self.assertTrue(all(value.abs().sum().item() > 0 for value in ce_gradients))
        self.assertTrue(all(weight.abs().sum().item() > 0
                            for weight in model.split(ce_gradients[0])[::2]))
        loss = loss + 1e-6 * gaussian_kl(model.mu, model.rho)
        loss.backward()
        self.assertTrue(torch.isfinite(model.mu.grad).all())
        self.assertTrue(torch.isfinite(model.rho.grad).all())
        self.assertGreater(model.mu.grad.abs().sum().item(), 0.)
        self.assertGreater(model.rho.grad.abs().sum().item(), 0.)

    def test_shape_and_configuration_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            GaussianSplitCNN(width=0)
        with self.assertRaises(ValueError):
            GaussianSplitCNN(posterior_std=0)
        model = GaussianSplitCNN(width=64)
        bank = ChannelBank(32, 5, {"kind": "perfect"}, 1)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3072), model.mu, bank)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 32, 32), model.mu, bank)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
