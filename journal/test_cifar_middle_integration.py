"""Middle-split dispatch and pilot isolation checks; no dataset training."""
import json
from pathlib import Path
import tempfile
import unittest

from journal.run_experiment import make_models, normalize_config
from journal.run_task import require_cpu_pilots


class MiddleCifarIntegrationTests(unittest.TestCase):
    def test_middle_dispatch_and_transmission_budget(self):
        config = normalize_config(dict(dataset="cifar10", architecture="cnn_mid", width=128))
        model, bank = make_models(config, 3072, 10)
        self.assertEqual(type(model).__name__, "GaussianMiddleSplitCNN")
        self.assertEqual(model.transmitted_shape, (8, 4, 4))
        self.assertEqual(bank.width, 128)
        self.assertEqual(model.d, 142514)
        self.assertEqual(model.transmitter_parameter_count + model.receiver_parameter_count, model.d)
        old, _ = make_models({**config, "architecture": "cnn"}, 3072, 10)
        self.assertEqual(type(old).__name__, "GaussianSplitCNN")
        self.assertEqual(old.d, 156074)

    def test_invalid_spatial_width_or_dataset_rejected(self):
        for width in (0, 15, 127, 129, 128., True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                normalize_config(dict(dataset="cifar10", architecture="cnn_mid", width=width))
        with self.assertRaises(ValueError):
            normalize_config(dict(dataset="mnist", architecture="cnn_mid", width=128))

    def test_old_head_split_pilot_cannot_authorize_middle_split(self):
        old = normalize_config(dict(dataset="cifar10", architecture="cnn", width=128,
                                    method="erm", epochs=1, train_size=10000,
                                    val_size=5000, device="cpu", val_draws=2))
        target = dict(task_id="middle_target", config={**old, "architecture": "cnn_mid"})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            directory.mkdir()
            metadata = dict(config=old, total_seconds=1., peak_rss_mb=100., selected_epoch=1)
            (directory / "metadata.json").write_text(json.dumps(metadata))
            (directory / "history.json").write_text(json.dumps([
                dict(epoch=1, validation=dict(cross_entropy=1.))]))
            (directory / "weights.pt").write_bytes(b"synthetic existence sentinel")
            with self.assertRaisesRegex(ValueError, "No completed matching CPU pilot"):
                require_cpu_pilots([target], temporary)
            metadata["config"]["architecture"] = "cnn_mid"
            (directory / "metadata.json").write_text(json.dumps(metadata))
            matches = require_cpu_pilots([target], temporary)
            self.assertEqual(matches[0]["cpu_pilots"], [str(directory)])


if __name__ == "__main__":
    unittest.main()
