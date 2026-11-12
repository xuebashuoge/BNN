"""Synthetic saved-metadata aggregation; no model, dataset or optimizer runs."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal import aggregate_checkpoints as aggregator
from journal.run_experiment import normalize_config


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def checkpoint_fixture(root, count=4):
    """Create two paired methods with explicitly mixed actual thread counts."""
    manifest = root / "manifest.jsonl"
    output = root / "checkpoints"
    output.mkdir()
    entries = []
    for index in range(count):
        method = "erm" if index < 2 else "proposed"
        seed = index % 2 + 1
        config = dict(dataset="cifar10", architecture="cnn_mid", method=method,
                      train_size=45000, val_size=5000, width=128, epochs=100,
                      threads=seed, seed=seed, channel_seed=10000 + seed,
                      eval_seed=20000 + seed)
        # Older checkpoints/manifests omit rho_lr_scale; normalization must
        # recover the unchanged 1.0 default, rather than rejecting the cache.
        if index % 2:
            config["rho_lr_scale"] = 1.
        entry = dict(task_id=f"synthetic_{method}_s{seed}", group="synthetic_cifar",
                     method_label="ERM" if method == "erm" else "T1",
                     family="baseline" if method == "erm" else "proposed",
                     variant_id=method, stage="synthetic", mode="train", config=config)
        entries.append(entry)
        directory = output / entry["task_id"]
        directory.mkdir()
        saved_config = normalize_config(config)
        if not index % 2:
            saved_config.pop("rho_lr_scale")
        write_json(directory / "metadata.json", dict(
            config=saved_config, selected_epoch=50, total_seconds=3.25,
            peak_rss_mb=1234., git_sha="a" * 40))
        write_json(directory / "history.json", [dict(
            epoch=epoch, validation=dict(accuracy=.7 + .01 * seed, cross_entropy=.8 - .01 * seed))
            for epoch in range(1, 101)])
        # Intentionally not a deserializable torch checkpoint: metadata-only
        # aggregation may check existence but must never read model tensors.
        (directory / "weights.pt").write_bytes(b"synthetic checkpoint existence sentinel\n")
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return manifest, output, entries


class AggregateCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        self.verify = self.guards.enter_context(patch.object(aggregator, "verify_submission_snapshot"))
        for name in ("load_data", "train", "evaluate_checkpoint"):
            self.guards.enter_context(patch(f"journal.run_experiment.{name}",
                side_effect=AssertionError("Aggregation must not load data, train, or infer")))
        self.guards.enter_context(patch("torch.load",
            side_effect=AssertionError("Aggregation must not deserialize model weights")))

    def test_complete_cases_preserve_execution_threads_and_normalize_old_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output, entries = checkpoint_fixture(Path(temporary))
            rows = aggregator.aggregate(manifest, output)
            self.assertEqual(len(rows), 4)
            expected_threads = {entry["task_id"]: entry["config"]["threads"] for entry in entries}
            self.assertEqual({row["task_id"]: row["config"]["threads"] for row in rows}, expected_threads)
            self.assertEqual(set(expected_threads.values()), {1, 2})
            self.assertTrue(all(row["config"]["rho_lr_scale"] == 1. for row in rows))
            self.assertTrue(all(row["status"] == "complete" and row["resumed"] for row in rows))
            self.assertTrue(all(row["selected_epoch"] == 50 for row in rows))
            self.assertEqual(json.loads((output / "training_results.json").read_text()), rows)
            snapshot = json.loads((output / "aggregation_snapshot.json").read_text())
            self.assertEqual(snapshot["completed_trials"], 4)
            self.assertEqual(snapshot["training_threads_by_task"], expected_threads)
            self.assertEqual(snapshot["new_optimization"], 0)
            self.assertIs(snapshot["checkpoint_inference"], False)
            self.assertIn("Withheld", snapshot["selection"])
            self.assertNotIn("selected_variants", snapshot)
            self.assertEqual(len(snapshot["variant_validation_summaries"]), 2)
            self.assertTrue(all(item["complete_seed_set"] for item in snapshot["variant_validation_summaries"]))
            self.assertEqual(self.verify.call_count, 2)

    def test_config_mismatches_are_rejected_without_replacing_execution_settings(self):
        for field, wrong in (("threads", 2), ("rho_lr_scale", .1), ("lr", .002)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                manifest, output, entries = checkpoint_fixture(Path(temporary), count=2)
                path = output / entries[0]["task_id"] / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata["config"][field] = wrong
                write_json(path, metadata)
                with self.assertRaisesRegex(ValueError, "Incomplete or mismatched checkpoint"):
                    aggregator.aggregate(manifest, output)
                self.assertFalse((output / "training_results.json").exists())
                self.assertFalse((output / "aggregation_snapshot.json").exists())

    def test_incomplete_last_case_withholds_all_publication(self):
        for missing in ("history_epoch", "weights"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                manifest, output, entries = checkpoint_fixture(Path(temporary))
                directory = output / entries[-1]["task_id"]
                if missing == "history_epoch":
                    history = json.loads((directory / "history.json").read_text())
                    write_json(directory / "history.json", history[:-1])
                else:
                    (directory / "weights.pt").unlink()
                with self.assertRaisesRegex(ValueError, entries[-1]["task_id"]):
                    aggregator.aggregate(manifest, output)
                self.assertFalse((output / "training_results.json").exists())
                self.assertFalse((output / "aggregation_snapshot.json").exists())

    def test_nonfinite_completion_or_selected_validation_metrics_are_rejected(self):
        for invalid in ("completion_time", "validation_accuracy"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                manifest, output, entries = checkpoint_fixture(Path(temporary), count=2)
                directory = output / entries[-1]["task_id"]
                if invalid == "completion_time":
                    path = directory / "metadata.json"
                    value = json.loads(path.read_text())
                    value["total_seconds"] = float("inf")
                else:
                    path = directory / "history.json"
                    value = json.loads(path.read_text())
                    value[49]["validation"]["accuracy"] = float("nan")
                write_json(path, value)
                with self.assertRaisesRegex(ValueError, "Incomplete or mismatched checkpoint"):
                    aggregator.aggregate(manifest, output)
                self.assertFalse((output / "training_results.json").exists())
                self.assertFalse((output / "aggregation_snapshot.json").exists())


if __name__ == "__main__":
    unittest.main()
