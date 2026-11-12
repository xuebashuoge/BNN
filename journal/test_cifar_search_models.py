"""Synthetic architecture-search checks; no datasets or optimizer updates."""

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from journal.cifar_models import GaussianMiddleSplitCNN, GaussianSplitCNN
from journal.cifar_search_models import (CIFAR_SEARCH_MODELS, GaussianMiddleSplitReLUCNN,
                                         GaussianReceiverConvCNN, GaussianSpatialThreeConvCNN)
from journal.theorem1 import ChannelBank, gaussian_kl


def explicit_transmitter(model, inputs, parts):
    if isinstance(model, GaussianMiddleSplitReLUCNN):
        w1, b1, w2, b2, wl, bl = parts[:6]
        x = F.max_pool2d(F.relu(F.conv2d(inputs, w1, b1, padding=1)), 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, w2, b2, padding=1)), 2)
        return F.relu(F.conv2d(F.adaptive_avg_pool2d(x, (4, 4)), wl, bl)).flatten(1)
    w1, b1, w2, b2, w3, b3, wl, bl = parts[:8]
    x = F.max_pool2d(F.relu(F.conv2d(inputs, w1, b1, padding=1)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w2, b2, padding=1)), 2)
    x = F.adaptive_avg_pool2d(F.relu(F.conv2d(x, w3, b3, padding=1)), (4, 4))
    if isinstance(model, GaussianReceiverConvCNN):
        return F.relu(F.linear(x.flatten(1), wl, bl))
    return F.relu(F.conv2d(x, wl, bl)).flatten(1)


def explicit_receiver(model, recovered, parts):
    wr, br, wf, bf, wo, bo = parts[6:] if isinstance(model, GaussianMiddleSplitReLUCNN) else parts[8:]
    x = recovered.reshape(len(recovered), *model.transmitted_shape)
    x = F.relu(F.conv2d(x, wr, br, padding=1)).flatten(1)
    return F.linear(F.relu(F.linear(x, wf, bf)), wo, bo)


class CifarSearchModelTests(unittest.TestCase):
    def test_counts_layout_views_and_initialization(self):
        cases = ((GaussianReceiverConvCNN, 128, 224074, 154784, 69290),
                 (GaussianReceiverConvCNN, 64, 157322, 89184, 68138),
                 (GaussianSpatialThreeConvCNN, 128, 161266, 24104, 137162),
                 (GaussianSpatialThreeConvCNN, 64, 158702, 23844, 134858))
        for model_class, width, count, tx, rx in cases:
            with self.subTest(model=model_class.__name__, width=width):
                model = model_class(width=width, posterior_std=.003)
                self.assertIs(CIFAR_SEARCH_MODELS[model.architecture], model_class)
                self.assertEqual(model.transmitted_shape, (width // 16, 4, 4))
                self.assertEqual(math.prod(model.transmitted_shape), width)
                self.assertEqual(model.d, count)
                self.assertEqual(model.transmitter_parameter_count, tx)
                self.assertEqual(model.receiver_parameter_count, rx)
                self.assertEqual(tx + rx, count)
                self.assertIn(f"{width} channel symbols", model.split_description)
                parts = model.split(model.mu)
                self.assertEqual(tuple(tuple(x.shape) for x in parts), model.parameter_shapes)
                self.assertEqual(tuple(parts[10].shape), (128, model.receiver_channels * 16))
                self.assertEqual(tuple(parts[12].shape), (10, 128))
                pointer = model.mu.untyped_storage().data_ptr()
                self.assertTrue(all(x.untyped_storage().data_ptr() == pointer for x in parts))
                self.assertEqual(sum(x.numel() for x in parts), count)
                for weight, bias in zip(parts[::2], parts[1::2]):
                    torch.testing.assert_close(bias, torch.zeros_like(bias))
                    self.assertLessEqual(weight.detach().abs().max().item(),
                                         1 / math.sqrt(math.prod(weight.shape[1:])))
                torch.testing.assert_close(F.softplus(model.rho), torch.full_like(model.rho, .003))

    def test_perfect_forward_is_explicit_transmitter_then_receiver(self):
        for model_class in CIFAR_SEARCH_MODELS.values():
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(91)
                model = model_class(width=128, stochastic=False).eval()
                bank = ChannelBank(128, 3, {"kind": "perfect"}, 11)
                inputs = torch.randn(2, 3, 32, 32)
                parts = model.split(model.mu)
                transmitted = explicit_transmitter(model, inputs, parts)
                self.assertTrue((transmitted >= 0).all())
                self.assertIs(model.sample(), model.mu)
                self.assertFalse(model.rho.requires_grad)
                expected = explicit_receiver(model, transmitted, parts)
                output = model(inputs, model.mu, bank, source="perfect")
                torch.testing.assert_close(output, expected)
                self.assertEqual(tuple(output.shape), (2, 10))
                self.assertTrue(torch.isfinite(output).all())
                self.assertAlmostEqual(model.last_feature_power,
                                       transmitted.detach().square().mean().item())

    def test_channel_acts_once_at_exact_flat_bottleneck(self):
        for model_class in CIFAR_SEARCH_MODELS.values():
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(92)
                model = model_class(width=128, stochastic=False).eval()
                bank = ChannelBank(128, 3, {"kind": "perfect"}, 11)
                with torch.no_grad():
                    bank.gain.copy_(torch.linspace(-1., 1., 128).expand(3, -1))
                    bank.noise.copy_(torch.linspace(.2, -.3, 128).expand(3, -1))
                inputs = torch.randn(2, 3, 32, 32)
                parts = model.split(model.mu)
                latent = explicit_transmitter(model, inputs, parts)
                generator = torch.Generator().manual_seed(19)
                with patch.object(bank, "apply", wraps=bank.apply) as apply:
                    output = model(inputs, model.mu, bank, generator=generator)
                apply.assert_called_once()
                actual, source, channel, stream = apply.call_args.args
                self.assertEqual(tuple(actual.shape), (2, 128))
                torch.testing.assert_close(actual, latent)
                self.assertEqual(source, "bank")
                self.assertIsNone(channel)
                self.assertIs(stream, generator)
                recovered = latent * bank.gain[0] + bank.noise[0]
                torch.testing.assert_close(output, explicit_receiver(model, recovered, parts))
                self.assertAlmostEqual(model.last_feature_power, latent.detach().square().mean().item())

    def test_rxconv_retains_original_encoder_exactly(self):
        torch.manual_seed(93)
        original = GaussianSplitCNN(width=128, stochastic=False)
        candidate = GaussianReceiverConvCNN(width=128, stochastic=False)
        original_tx = sum(part.numel() for part in original.split(original.mu)[:8])
        self.assertEqual(original_tx, candidate.transmitter_parameter_count)
        with torch.no_grad():
            candidate.mu[:original_tx].copy_(original.mu[:original_tx])
        inputs = torch.randn(2, 3, 32, 32)
        bank = ChannelBank(128, 3, {"kind": "perfect"}, 11)
        with patch.object(bank, "apply", wraps=bank.apply) as apply:
            original(inputs, original.mu, bank, source="perfect")
            original_latent = apply.call_args.args[0].detach().clone()
            candidate(inputs, candidate.mu, bank, source="perfect")
            candidate_latent = apply.call_args.args[0]
        torch.testing.assert_close(original_latent, candidate_latent)

    def test_middle_relu_preserves_parameters_and_rectifies_only_the_latent(self):
        torch.manual_seed(96)
        original = GaussianMiddleSplitCNN(width=128, stochastic=False)
        torch.manual_seed(96)
        candidate = GaussianMiddleSplitReLUCNN(width=128, stochastic=False).eval()
        self.assertIs(CIFAR_SEARCH_MODELS["cnn_mid_relu"], GaussianMiddleSplitReLUCNN)
        self.assertIsInstance(candidate, GaussianMiddleSplitCNN)
        self.assertEqual(candidate.parameter_shapes, original.parameter_shapes)
        self.assertEqual(candidate.d, 142514)
        self.assertEqual(candidate.transmitter_parameter_count, 5352)
        self.assertEqual(candidate.receiver_parameter_count, 137162)
        self.assertTrue(candidate.nonnegative_transmitted_features)
        self.assertIn("nonnegative channel symbols", candidate.split_description)
        torch.testing.assert_close(candidate.mu, original.mu)
        torch.testing.assert_close(candidate.rho, original.rho)
        parts = candidate.split(candidate.mu)
        with torch.no_grad():
            parts[4].zero_()
            parts[5].copy_(torch.linspace(-2., 2., 8))
        inputs = torch.randn(2, 3, 32, 32)
        bank = ChannelBank(128, 3, {"kind": "perfect"}, 11)
        with patch.object(bank, "apply", wraps=bank.apply) as apply:
            original(inputs, candidate.mu, bank, source="perfect")
            signed_latent = apply.call_args.args[0].detach().clone()
            output = candidate(inputs, candidate.mu, bank, source="perfect")
            nonnegative_latent = apply.call_args.args[0]
        self.assertTrue((signed_latent < 0).any())
        self.assertEqual(tuple(nonnegative_latent.shape), (2, 128))
        self.assertTrue((nonnegative_latent >= 0).all())
        torch.testing.assert_close(nonnegative_latent, F.relu(signed_latent))
        torch.testing.assert_close(output, explicit_receiver(candidate, nonnegative_latent, parts))
        self.assertAlmostEqual(candidate.last_feature_power,
                               nonnegative_latent.detach().square().mean().item())

    def test_ce_gradients_and_existing_flat_kl_cover_both_sides(self):
        for model_class in CIFAR_SEARCH_MODELS.values():
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(94)
                model = model_class(width=128, posterior_std=.003)
                bank = ChannelBank(128, 5, {"kind": "gaussian", "gain_mean": .5,
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
                kl = gaussian_kl(model.mu, model.rho)
                self.assertTrue(torch.isfinite(kl))
                self.assertGreater(kl.item(), 0.)
                self.assertTrue(all(torch.isfinite(value).all()
                                    for value in torch.autograd.grad(kl, (model.mu, model.rho))))

    def test_sampling_and_channel_evaluation_are_reproducible(self):
        for model_class in CIFAR_SEARCH_MODELS.values():
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(95)
                model = model_class(width=128)
                inputs = torch.randn(2, 3, 32, 32)
                bank = ChannelBank(128, 5, {"kind": "outage", "outage_prob": .5}, 11)
                first = model.sample(torch.Generator().manual_seed(18))
                second = model.sample(torch.Generator().manual_seed(18))
                torch.testing.assert_close(first, second)
                self.assertFalse(torch.equal(first, model.sample(torch.Generator().manual_seed(19))))
                torch.testing.assert_close(
                    model(inputs, first, bank, generator=torch.Generator().manual_seed(20)),
                    model(inputs, second, bank, generator=torch.Generator().manual_seed(20)))

    def test_invalid_inputs_and_parameters_fail_explicitly(self):
        for model_class in CIFAR_SEARCH_MODELS.values():
            for width in (0, -16, 15, 17, 1.5, True):
                with self.subTest(model=model_class.__name__, width=width), self.assertRaises(ValueError):
                    model_class(width=width)
            for classes in (0, 1, 1.5, True):
                with self.subTest(model=model_class.__name__, classes=classes), self.assertRaises(ValueError):
                    model_class(classes=classes)
            for scale in (0., -1., float("nan"), float("inf")):
                with self.subTest(model=model_class.__name__, scale=scale), self.assertRaises(ValueError):
                    model_class(posterior_std=scale)
            model = model_class(width=128)
            bank = ChannelBank(128, 3, {"kind": "perfect"}, 11)
            with self.assertRaises(ValueError):
                model.split(torch.zeros(model.d + 1))
            with self.assertRaises(ValueError):
                model.split(torch.zeros(1, model.d))
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 3072), model.mu, bank)
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 3, 32, 32), model.mu,
                      ChannelBank(64, 3, {"kind": "perfect"}, 11))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
