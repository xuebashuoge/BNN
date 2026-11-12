"""Metadata-only pool batching, isolation, and failure checks; no training."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal import batch_campaign as batch
from journal.build_campaign import make_entry


def entries(total=10):
    config = dict(dataset="moons", architecture="mlp", method="erm", width=4,
                  train_size=20, val_size=10, epochs=1, m=5,
                  train_channel={"kind": "perfect"}, deployment_channel={"kind": "perfect"})
    return [make_entry(config, seed, "synthetic", "ERM", "pool_test") for seed in range(total)]


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class BatchCampaignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.jsonl"
        self.rows = entries()
        write_manifest(self.manifest, self.rows)
        self.output = self.root / "output"
        self.enterContext(patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}))
        self.enterContext(patch.object(batch, "allowed_cpus", return_value=list(range(8))))
        self.guard = self.enterContext(patch.object(batch, "verify_submission_snapshot"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def complete(entry, output_root, threads, resume, device):
        assert threads == 2 and resume is True and device == "cpu"
        return dict(task_id=entry["task_id"], status="complete", resumed=True)

    def test_contiguous_coverage_tail_and_cache_positions(self):
        self.rows[1]["already_cached"] = True
        slices = [batch.select_batch(self.rows, index) for index in range(3)]
        self.assertEqual([len(items) for items in slices], [4, 4, 2])
        self.assertEqual([item for items in slices for item in items], self.rows)
        self.assertEqual(slices[0][1]["task_id"], self.rows[1]["task_id"])
        self.assertEqual(batch.batch_count(24), 6)
        self.assertEqual(batch.batch_count(28), 7)
        for index in (-1, 3, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                batch.select_batch(self.rows, index)

    def test_reports_isolate_batches_and_leave_global_reports_untouched(self):
        self.output.mkdir()
        global_report = self.output / "training_results.json"
        global_report.write_text("keep global aggregation untouched")
        with patch.object(batch, "train_trial", side_effect=self.complete) as trial:
            first = batch.run_batch(self.manifest, self.output, 0)
            first_path = self.output / "pool_reports/batch_0/training_results.json"
            first_bytes = first_path.read_bytes()
            second = batch.run_batch(self.manifest, self.output, 1)
        self.assertEqual([row["task_id"] for row in first], [row["task_id"] for row in self.rows[:4]])
        self.assertEqual([row["task_id"] for row in second], [row["task_id"] for row in self.rows[4:8]])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(global_report.read_text(), "keep global aggregation untouched")
        self.assertEqual(trial.call_count, 8)
        self.assertTrue(all(call.args[1] == self.output for call in trial.call_args_list))
        self.assertEqual(self.guard.call_count, 4)

    def test_all_outcomes_retained_and_cli_exits_nonzero_after_failure(self):
        def trial(entry, output_root, threads, resume, device):
            if entry["task_id"] == self.rows[1]["task_id"]:
                raise RuntimeError("synthetic launcher failure")
            if entry["task_id"] == self.rows[2]["task_id"]:
                return dict(task_id=entry["task_id"], status="failed", error="synthetic trial failure")
            return self.complete(entry, output_root, threads, resume, device)

        argv = ["batch_campaign", "--manifest", str(self.manifest), "--output-root", str(self.output),
                "--batch-index", "0"]
        with patch.object(batch, "train_trial", side_effect=trial) as calls, patch("sys.argv", argv):
            self.assertEqual(batch.main(), 1)
        saved = json.loads((self.output / "pool_reports/batch_0/training_results.json").read_text())
        self.assertEqual(calls.call_count, 4)
        self.assertEqual([row["task_id"] for row in saved], [row["task_id"] for row in self.rows[:4]])
        self.assertEqual([row["status"] for row in saved], ["complete", "failed", "failed", "complete"])
        self.assertIn("synthetic launcher failure", saved[1]["error"])
        self.assertEqual(saved[1]["config"]["seed"], self.rows[1]["config"]["seed"])
        self.assertEqual(Path(saved[1]["checkpoint"]).name, self.rows[1]["task_id"])

    def test_submission_guard_precedes_any_trial_or_report(self):
        self.guard.side_effect = RuntimeError("changed manifest")
        with patch.object(batch, "train_trial") as trial, self.assertRaises(RuntimeError):
            batch.run_batch(self.manifest, self.output, 0)
        trial.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_resumed_batch_preserves_prior_attempt_reports(self):
        with patch.object(batch, "train_trial", side_effect=self.complete):
            batch.run_batch(self.manifest, self.output, 0)
            batch.run_batch(self.manifest, self.output, 0)
        root = self.output / "pool_reports/batch_0"
        archives = list((root / "previous_attempts").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual(len(json.loads((archives[0] / "training_results.json").read_text())), 4)
        snapshot = json.loads((root / "batch_snapshot.json").read_text())
        self.assertEqual(snapshot["task_ids"], [row["task_id"] for row in self.rows[:4]])
        self.assertFalse(snapshot["cache_filtering"])

    def test_changed_master_cannot_overwrite_pool_reports(self):
        with patch.object(batch, "train_trial", side_effect=self.complete) as trial:
            batch.run_batch(self.manifest, self.output, 0)
            report = self.output / "pool_reports/batch_0/training_results.json"
            before = report.read_bytes()
            write_manifest(self.manifest, self.rows + entries(11)[-1:])
            with self.assertRaisesRegex(ValueError, "different manifest/batch"):
                batch.run_batch(self.manifest, self.output, 0)
        self.assertEqual(report.read_bytes(), before)
        self.assertEqual(trial.call_count, 4)

    def test_cpu_budget_checks_allocation_and_actual_mask(self):
        self.assertEqual(batch.validate_cpu_budget(4, 2), 8)
        for workers, threads in ((5, 2), (0, 2), (4, 0), (True, 2)):
            with self.subTest(workers=workers, threads=threads), self.assertRaises(ValueError):
                batch.validate_cpu_budget(workers, threads)
        with patch.object(batch, "allowed_cpus", return_value=[0, 1]), self.assertRaisesRegex(ValueError, "affinity"):
            batch.validate_cpu_budget(4, 2)

    def test_import_induced_affinity_narrowing_stops_before_children(self):
        masks = [list(range(8)), list(range(8)), [0]]
        with patch.object(batch, "allowed_cpus", side_effect=masks), patch.object(batch, "train_trial") as trial:
            with self.assertRaisesRegex(ValueError, "affinity"):
                batch.run_batch(self.manifest, self.output, 0)
        trial.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_validate_only_has_no_training_or_output_writes(self):
        argv = ["batch_campaign", "--manifest", str(self.manifest), "--output-root", str(self.output), "--validate-only"]
        with patch.object(batch, "train_trial") as trial, patch("sys.argv", argv):
            self.assertEqual(batch.main(), 0)
        trial.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
