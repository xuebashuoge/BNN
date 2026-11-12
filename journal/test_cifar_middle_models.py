"""Middle-split CNN mathematical checks; no datasets or optimizer updates."""

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from journal.cifar_models import GaussianMiddleSplitCNN
from journal.theorem1 import ChannelBank, gaussian_kl


def explicit_transmitter(x, parts):
    w1, b1, w2, b2, wl, bl = parts[:6]
    x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w2, b2, padding=1)), 2)
    return F.conv2d(F.adaptive_avg_pool2d(x, (4, 4)), wl, bl)


def explicit_receiver(latent, parts):
    wr, br, wf, bf, wo, bo = parts[6:]
    x = F.relu(F.conv2d(latent, wr, br, padding=1)).flatten(1)
    return F.linear(F.relu(F.linear(x, wf, bf)), wo, bo)


class CifarMiddleModelTests(unittest.TestCase):
    def test_shapes_counts_initialization_and_storage_views(self):
        for width, total, tx, rx in ((16, 22331, 5121, 17210),
                                     (64, 73838, 5220, 68618),
                                     (128, 142514, 5352, 137162)):
            with self.subTest(width=width):
                model = GaussianMiddleSplitCNN(width=width, posterior_std=.003)
                self.assertEqual(model.architecture, "cnn_mid")
                self.assertEqual(model.transmitted_shape, (width // 16, 4, 4))
                self.assertEqual(math.prod(model.transmitted_shape), width)
                self.assertEqual(model.d, total)
                self.assertEqual(model.transmitter_parameter_count, tx)
                self.assertEqual(model.receiver_parameter_count, rx)
                self.assertEqual(tx + rx, total)
                self.assertIn(f"to {width} channel symbols", model.split_description)
                parts = model.split(model.mu)
                self.assertEqual(tuple(tuple(x.shape) for x in parts), model.parameter_shapes)
                self.assertEqual(sum(x.numel() for x in parts), total)
                pointer = model.mu.untyped_storage().data_ptr()
                self.assertTrue(all(x.untyped_storage().data_ptr() == pointer for x in parts))
                for weight, bias in zip(parts[::2], parts[1::2]):
                    torch.testing.assert_close(bias, torch.zeros_like(bias))
                    self.assertLessEqual(weight.detach().abs().max().item(),
                                         1 / math.sqrt(math.prod(weight.shape[1:])))
                torch.testing.assert_close(F.softplus(model.rho), torch.full_like(model.rho, .003))
        with self.assertRaises(ValueError):
            model.split(torch.zeros(model.d + 1))
        with self.assertRaises(ValueError):
            model.split(torch.zeros(1, model.d))

    def test_perfect_channel_equals_explicit_transmitter_then_receiver(self):
        torch.manual_seed(81)
        model = GaussianMiddleSplitCNN(width=64, stochastic=False)
        model.eval()
        inputs = torch.randn(3, 3, 32, 32)
        bank = ChannelBank(64, 5, {"kind": "perfect"}, 11)
        self.assertIs(model.sample(), model.mu)
        self.assertFalse(model.rho.requires_grad)
        parts = model.split(model.mu)
        latent = explicit_transmitter(inputs, parts)
        expected = explicit_receiver(latent, parts)
        perfect = model(inputs, model.mu, bank, source="perfect")
        via_bank = model(inputs, model.mu, bank, generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(perfect, expected)
        torch.testing.assert_close(via_bank, expected)
        self.assertEqual(tuple(perfect.shape), (3, 10))
        self.assertTrue(torch.isfinite(perfect).all())
        self.assertAlmostEqual(model.last_feature_power, latent.detach().square().mean().item())

    def test_bank_acts_once_on_signed_flat_spatial_latent(self):
        torch.manual_seed(82)
        model = GaussianMiddleSplitCNN(width=64, stochastic=False)
        model.eval()
        parts = model.split(model.mu)
        with torch.no_grad():
            # Both signs must survive the linear 1x1 projection.
            parts[4].zero_()
            parts[5].copy_(torch.linspace(-2., 2., 4))
        inputs = torch.randn(2, 3, 32, 32)
        latent = explicit_transmitter(inputs, parts)
        self.assertTrue((latent < 0).any())
        bank = ChannelBank(64, 3, {"kind": "perfect"}, 11)
        with torch.no_grad():
            bank.gain.copy_(torch.linspace(-1., 1., 64).expand(3, -1))
            bank.noise.copy_(torch.linspace(.3, -.1, 64).expand(3, -1))
        generator = torch.Generator().manual_seed(19)
        with patch.object(bank, "apply", wraps=bank.apply) as apply:
            output = model(inputs, model.mu, bank, generator=generator)
        apply.assert_called_once()
        transmitted, source, channel, actual_generator = apply.call_args.args
        self.assertEqual(tuple(transmitted.shape), (2, 64))
        torch.testing.assert_close(transmitted, latent.flatten(1))
        self.assertEqual(source, "bank")
        self.assertIsNone(channel)
        self.assertIs(actual_generator, generator)
        recovered = (latent.flatten(1) * bank.gain[0] + bank.noise[0]).reshape(2, 4, 4, 4)
        torch.testing.assert_close(output, explicit_receiver(recovered, parts))
        self.assertAlmostEqual(model.last_feature_power, latent.detach().square().mean().item())

    def test_ce_gradients_reach_both_sides_and_all_weight_layers(self):
        torch.manual_seed(83)
        model = GaussianMiddleSplitCNN(width=64, posterior_std=.003)
        bank = ChannelBank(64, 5, {"kind": "gaussian", "gain_mean": .5,
                                  "gain_std": 1., "noise_std": 1.}, 11)
        output = model(torch.randn(2, 3, 32, 32),
                       model.sample(torch.Generator().manual_seed(18)), bank,
                       generator=torch.Generator().manual_seed(19))
        gradients = torch.autograd.grad(F.cross_entropy(output, torch.tensor([1, 4])),
                                        (model.mu, model.rho))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient[:model.transmitter_parameter_count].abs().sum().item(), 0.)
            self.assertGreater(gradient[model.transmitter_parameter_count:].abs().sum().item(), 0.)
            self.assertTrue(all(weight.abs().sum().item() > 0
                                for weight in model.split(gradient)[::2]))
        # Existing flat Gaussian KL applies to the entire transmitter/receiver.
        kl = gaussian_kl(model.mu, model.rho)
        self.assertTrue(torch.isfinite(kl))
        self.assertGreater(kl.item(), 0.)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in torch.autograd.grad(kl, (model.mu, model.rho))))

    def test_stochastic_weights_and_channel_streams_are_reproducible(self):
        torch.manual_seed(84)
        model = GaussianMiddleSplitCNN(width=64)
        inputs = torch.randn(2, 3, 32, 32)
        bank = ChannelBank(64, 5, {"kind": "outage", "outage_prob": .5}, 11)
        theta = model.sample(torch.Generator().manual_seed(18))
        repeated = model.sample(torch.Generator().manual_seed(18))
        torch.testing.assert_close(theta, repeated)
        self.assertFalse(torch.equal(theta, model.sample(torch.Generator().manual_seed(19))))
        first = model(inputs, theta, bank, generator=torch.Generator().manual_seed(20))
        second = model(inputs, repeated, bank, generator=torch.Generator().manual_seed(20))
        torch.testing.assert_close(first, second)

    def test_invalid_dimensions_and_scales_are_rejected(self):
        for width in (0, -16, 15, 17, 1.5, True):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "multiple of 16"):
                GaussianMiddleSplitCNN(width=width)
        for classes in (0, 1, 1.5, True):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                GaussianMiddleSplitCNN(classes=classes)
        for scale in (0., -1., float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                GaussianMiddleSplitCNN(posterior_std=scale)
        model = GaussianMiddleSplitCNN(width=64)
        bank = ChannelBank(64, 5, {"kind": "perfect"}, 11)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3072), model.mu, bank)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 32, 32), model.mu,
                  ChannelBank(128, 5, {"kind": "perfect"}, 11))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
