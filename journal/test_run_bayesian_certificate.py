"""Test the auxiliary certificate orchestration without data or optimization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from journal.build_bayesian_certificate import diagnostic_entries
from journal.run_bayesian_certificate import pilot_entries, run


class CertificatePilotTests(unittest.TestCase):
    def test_pilots_cover_every_shape_and_only_change_epochs_and_rng_seeds(self):
        entries = diagnostic_entries()
        pilots = pilot_entries(entries)
        self.assertEqual(len(pilots), 16)
        self.assertEqual(len({item["task_id"] for item in pilots}), 16)
        source = {(item["group"], item["method_label"]): item for item in entries}
        ignored = {"epochs", "seed", "channel_seed", "eval_seed"}
        for item in pilots:
            parent = source[(item["group"], item["method_label"])]
            self.assertEqual({key: value for key, value in item["config"].items() if key not in ignored},
                             {key: value for key, value in parent["config"].items() if key not in ignored})
            self.assertEqual(item["config"]["epochs"], 1)
            self.assertEqual(item["config"]["seed"], 76100)
            self.assertEqual(item["config"]["channel_seed"], 86100)
            self.assertEqual(item["config"]["eval_seed"], 96100)
            self.assertFalse(item["scientific_result"])
            self.assertEqual(item["role"], "resource_pilot")
            self.assertEqual(item["family"], parent["family"])


class CertificateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.finished_indices = set()
        self.active_mc = 0
        self.peak_mc = 0
        self.lock = threading.Lock()
        self.fail_mc_index = None
        self.failed_evaluation = False
        self.incomplete_pilot = False

    def fake_command(self, args, log, environment):
        with self.lock:
            self.commands.append((list(args), dict(environment)))
        if args[0] == "journal.campaign":
            output = Path(args[args.index("--output-root") + 1])
            output.mkdir(parents=True, exist_ok=True)
            if output.name == "pilot":
                count = 15 if self.incomplete_pilot else 16
                (output / "training_results.json").write_text(json.dumps([
                    dict(task_id=f"pilot{index}", status="complete", peak_rss_mb=256., seconds=1.)
                    for index in range(count)]))
            else:
                (output / "summary.json").write_text(json.dumps(dict(
                    completed_trials=48, failed_trials=[],
                    failed_evaluations=[dict(task_id="missing")] if self.failed_evaluation else [])))
        elif args[0] == "journal.evaluate_bayesian_certificate":
            if "--index" in args:
                index = int(args[args.index("--index") + 1])
                with self.lock:
                    self.active_mc += 1
                    self.peak_mc = max(self.peak_mc, self.active_mc)
                try:
                    # A short synthetic wait exercises bounded overlap; no
                    # inference, subprocess, data access or training occurs.
                    time.sleep(.001)
                    if index == self.fail_mc_index:
                        raise subprocess.CalledProcessError(1, args)
                    with self.lock:
                        self.finished_indices.add(index)
                finally:
                    with self.lock:
                        self.active_mc -= 1
            else:
                self.assertIn("--aggregate-only", args)
                self.assertEqual(self.finished_indices, set(range(48)))
                self.assertEqual(self.active_mc, 0)
        elif args[0] == "journal.plot_bayesian_certificate":
            self.assertEqual(self.finished_indices, set(range(48)))

    def run_mock(self, root):
        manifest = Path(root, "certificate.jsonl")
        manifest.write_text("".join(json.dumps(item) + "\n" for item in diagnostic_entries()))
        output = Path(root, "output")
        with patch("journal.run_bayesian_certificate.command", side_effect=self.fake_command), \
                patch("journal.batch_campaign.validate_cpu_budget", return_value=8), \
                patch("journal.run_task.verify_submission_snapshot") as source_guard, \
                patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "4096", "JOURNAL_GIT_SHA": "aux_source_sha"}):
            state = run(manifest, output, workers=4, threads=2)
        source_guard.assert_called_once_with(manifest)
        return manifest, output, state

    def test_phase_order_pilot_no_test_then_all48_mc_before_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, output, state = self.run_mock(root)
            self.assertEqual(state["status"], "complete")
            args = [call for call, environment in self.commands]
            self.assertEqual(args[0][0], "journal.campaign")
            self.assertEqual(args[0][args[0].index("--evaluate-top") + 1], "0")
            self.assertNotIn("--full-risks", args[0])
            self.assertEqual(args[1][0], "journal.campaign")
            self.assertEqual(args[1][args[1].index("--evaluate-top") + 1], "1")
            self.assertEqual(args[1][args[1].index("--eval-draws") + 1], "20")
            self.assertIn("--full-risks", args[1])
            mc = [call for call in args if "--index" in call]
            self.assertEqual(len(mc), 48)
            self.assertEqual({int(call[call.index("--index") + 1]) for call in mc}, set(range(48)))
            self.assertLessEqual(self.peak_mc, 4)
            self.assertGreater(self.peak_mc, 1)
            for call in mc:
                self.assertEqual(call[call.index("--draws") + 1], "1000")
                self.assertEqual(call[call.index("--delta") + 1], ".025")
                self.assertIn("--resume", call)
            self.assertIn("--aggregate-only", args[-2])
            self.assertEqual(args[-1][0], "journal.plot_bayesian_certificate")
            self.assertEqual(self.commands[0][1]["JOURNAL_MANIFEST_SHA256"],
                             hashlib.sha256((output / "pilot.jsonl").read_bytes()).hexdigest())
            for call, environment in self.commands[1:]:
                self.assertEqual(environment["JOURNAL_MANIFEST_SHA256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")

    def test_failed_mc_does_not_publish_aggregate_or_figures(self):
        self.fail_mc_index = 7
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_mock(root)
            self.assertFalse(any("--aggregate-only" in args for args, env in self.commands))
            self.assertFalse(any(args[0] == "journal.plot_bayesian_certificate" for args, env in self.commands))
            status = json.loads(Path(root, "output", "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("CalledProcessError", status["error"])

    def frozen_campaign(self, root):
        output = Path(root, "output")
        campaign = output / "campaign"
        campaign.mkdir(parents=True)
        (campaign / "summary.json").write_text(json.dumps(dict(
            completed_trials=48, failed_trials=[], failed_evaluations=[])))
        # Preserve intentional ordering/whitespace: protocol identity hashes
        # raw bytes, so a semantically identical rewrite must also be avoided.
        raw = b'[\n  {"task_id":"frozen_b"}, {"task_id":"frozen_a"}\n]\n'
        (campaign / "test_results.json").write_bytes(raw)
        (output / "bounds").mkdir()
        (output / "bounds" / "certificate_family.json").write_text('{"frozen":true}\n')
        return campaign, raw

    def test_complete_resume_validates_and_preserves_frozen_test_source_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            campaign, raw = self.frozen_campaign(root)
            with patch("journal.evaluate_bayesian_certificate.declared_family", return_value=[]) as validate:
                manifest, output, state = self.run_mock(root)
            validate.assert_called_once_with(manifest, campaign.resolve())
            self.assertTrue(state["resumed_complete_campaign"])
            self.assertEqual(state["status"], "complete")
            self.assertEqual((campaign / "test_results.json").read_bytes(), raw)
            campaigns = [args for args, env in self.commands if args[0] == "journal.campaign"]
            self.assertEqual(len(campaigns), 1)
            self.assertEqual(Path(campaigns[0][campaigns[0].index("--output-root") + 1]).name, "pilot")
            self.assertEqual(self.finished_indices, set(range(48)))

    def test_invalid_frozen_source_refuses_repair_or_new_mc(self):
        with tempfile.TemporaryDirectory() as root:
            campaign, raw = self.frozen_campaign(root)
            with patch("journal.evaluate_bayesian_certificate.declared_family",
                       side_effect=ValueError("frozen source changed")):
                with self.assertRaisesRegex(ValueError, "frozen certificate source campaign changed"):
                    self.run_mock(root)
            self.assertEqual((campaign / "test_results.json").read_bytes(), raw)
            self.assertEqual(len(self.commands), 1)
            self.assertEqual(self.commands[0][0][0], "journal.campaign")
            self.assertEqual(self.finished_indices, set())
            self.assertEqual(json.loads(Path(root, "output", "status.json").read_text())["status"], "failed")

    def test_incomplete_pilot_and_failed_test_prevent_dependent_stages(self):
        for failure in ("incomplete_pilot", "failed_evaluation"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                self.setUp()
                setattr(self, failure, True)
                with self.assertRaises(ValueError):
                    self.run_mock(root)
                self.assertFalse(any(args[0] == "journal.evaluate_bayesian_certificate" for args, env in self.commands))
                if failure == "incomplete_pilot":
                    self.assertEqual(len(self.commands), 1)

    def test_memory_gate_stops_after_measured_pilot(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root, "certificate.jsonl")
            manifest.write_text("".join(json.dumps(item) + "\n" for item in diagnostic_entries()))
            output = Path(root, "output")
            with patch("journal.run_bayesian_certificate.command", side_effect=self.fake_command), \
                    patch("journal.batch_campaign.validate_cpu_budget", return_value=8), \
                    patch("journal.run_task.verify_submission_snapshot"), \
                    patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "2000"}):
                # 1GiB coordinator +4x256MiB workers with25% headroom=2304MiB.
                with self.assertRaisesRegex(RuntimeError, "pilot pool"):
                    run(manifest, output, workers=4, threads=2)
            self.assertEqual(len(self.commands), 1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_full_source_guard_runs_before_manifest_read_or_process_execution(self):
        with tempfile.TemporaryDirectory() as root, \
                patch("journal.batch_campaign.validate_cpu_budget", return_value=8), \
                patch("journal.run_task.verify_submission_snapshot", side_effect=RuntimeError("source changed")), \
                patch("journal.run_task.read_manifest") as reader, \
                patch("journal.run_bayesian_certificate.command") as command:
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                run(Path(root, "manifest.jsonl"), Path(root, "output"))
            reader.assert_not_called()
            command.assert_not_called()
            self.assertFalse(Path(root, "output").exists())


if __name__ == "__main__":
    unittest.main()
