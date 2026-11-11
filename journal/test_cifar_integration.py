"""CIFAR integration checks using tiny synthetic data; no training or download."""

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from journal.certify_checkpoint import sampler_source_signature, verify_sampler_source
from journal.run_experiment import load_data, make_models, normalize_config
from journal.run_task import PILOT_MATCH_FIELDS


class CifarIntegrationTests(unittest.TestCase):
    def test_tensor_preprocessing_split_and_test_isolation(self):
        calls = []
        class TinyCifar:
            def __init__(self, root, train, download):
                calls.append(train)
                self.data = np.zeros((12 if train else 4, 32, 32, 3), dtype=np.uint8)
                self.data[:, 0, 0, 1] = 255
                self.targets = [index % 10 for index in range(len(self.data))]
        config = normalize_config(dict(dataset="cifar10", architecture="cnn", train_size=8, val_size=4))
        with patch("torchvision.datasets.CIFAR10", TinyCifar), patch.dict("os.environ", {"DATASET": "/synthetic"}):
            train, val, test, input_dim, classes, split = load_data(config, include_test=False)
            self.assertEqual(calls, [True])
            self.assertIsNone(test)
            self.assertEqual((len(train), len(val), input_dim, classes), (8, 4, 3072, 10))
            self.assertFalse(set(train.indices) & set(val.indices))
            x, _ = train[0]
            self.assertEqual(tuple(x.shape), (3, 32, 32))
            self.assertTrue(x.is_contiguous())
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(x[1, 0, 0].item(), 1.)
            self.assertEqual(x[0, 0, 0].item(), -1.)
            _, _, test, _, _, repeated = load_data(config, include_test=True)
            self.assertEqual(calls, [True, True, False])
            self.assertEqual(len(test), 4)
            self.assertEqual(split["train_indices_hash"], repeated["train_indices_hash"])

    def test_model_dispatch_and_architecture_guard(self):
        config = normalize_config(dict(dataset="cifar10", architecture="cnn", width=16))
        model, bank = make_models(config, 3072, 10)
        self.assertEqual(type(model).__name__, "GaussianSplitCNN")
        self.assertEqual(bank.width, 16)
        self.assertIn("architecture", PILOT_MATCH_FIELDS)
        self.assertEqual(normalize_config({})["architecture"], "mlp")
        with self.assertRaises(ValueError):
            normalize_config(dict(dataset="mnist", architecture="cnn"))

    def test_image_only_source_edits_do_not_change_toy_fingerprint(self):
        def node(image, toy="1", split="2"):
            code = f'def load_data(config):\n    if config["dataset"] == "moons":\n        x = {toy}\n    else:\n        x = {image}\n    indices = {split}\n    return x, indices\n'
            return ast.parse(code).body[0]
        self.assertEqual(sampler_source_signature(node("2")), sampler_source_signature(node("3")))
        self.assertNotEqual(sampler_source_signature(node("2")), sampler_source_signature(node("2", toy="4")))
        self.assertNotEqual(sampler_source_signature(node("2")), sampler_source_signature(node("2", split="5")))

    def test_existing_iid_checkpoint_sampler_remains_valid(self):
        sha = subprocess.check_output(["git", "rev-parse", "69fa1d1"], text=True).strip()
        self.assertTrue(all(verify_sampler_source(sha).values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
