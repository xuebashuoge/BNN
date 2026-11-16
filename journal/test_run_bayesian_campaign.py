"""Verify staged orchestration and honest fixed-seed main comparison reporting.

All process execution is mocked. These tests do not open datasets, train, or
perform posterior Monte Carlo evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from journal.run_bayesian_campaign import STAGES, pipeline, ranking_outcome


LABELS = tuple(f"{method}-{regime}" for regime in ("Matched", "Perfect")
               for method in ("ERM", "L2", "T1", "T2"))


def ranking_rows(accuracies=None, group="moons_outage", seeds=(73101, 73102)):
    accuracies = accuracies or {}
    rows = []
    for label in LABELS:
        values = accuracies.get(label, [.8] * len(seeds))
        for seed, accuracy in zip(seeds, values):
            rows.append(dict(task_id=f"{group}_{label}_{seed}", group=group,
                             method_label=label, seed=seed,
                             test_status="complete", test_accuracy=accuracy))
    return rows


def write_rows(root, rows):
    Path(root, "test_results.json").write_text(json.dumps(rows))


def write_manifest(root, rows, name="confirm.jsonl"):
    path = Path(root, name)
    path.write_text("".join(json.dumps({key: value for key, value in row.items()
                                      if key in {"task_id", "group", "method_label"}} |
                                     {"config": {"seed": row["seed"]}}) + "\n" for row in rows))
    return path


class RankingOutcomeTests(unittest.TestCase):
    def test_all_eight_means_and_one_fixed_proposed_method_per_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            # Switching between T1 and T2 for each seed would manufacture a .9
            # proposed accuracy; each actual fixed method has mean .79.
            rows = ranking_rows({"T1-Matched": [.9, .68], "T2-Matched": [.68, .9]})
            write_rows(directory, rows)
            outcome = ranking_outcome(directory)
            self.assertFalse(outcome["target_met_in_every_condition"])
            condition = outcome["conditions"][0]
            self.assertEqual(set(condition["means"]), set(LABELS))
            self.assertAlmostEqual(condition["means"]["T1-Matched"], .79)
            self.assertAlmostEqual(condition["means"]["T2-Matched"], .79)
            self.assertAlmostEqual(condition["mean_accuracy_margin"], -.01)
            self.assertTrue(all(seeds == [73101, 73102] for seeds in condition["seeds"].values()))
            self.assertEqual({item["fixed_matched_candidate"] for item in condition["fixed_candidate_per_seed"]},
                             {condition["best_matched_proposed_by_observed_mean"]})
            self.assertFalse(condition["target_met_for_every_seed"])

    def test_perfect_competitor_and_tie_block_strict_matched_win(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = ranking_rows({"T1-Matched": [.85, .85], "T2-Matched": [.83, .83],
                                 "ERM-Perfect": [.86, .86]})
            write_rows(directory, rows)
            condition = ranking_outcome(directory)["conditions"][0]
            self.assertEqual(condition["strongest_other_method"], "ERM-Perfect")
            self.assertFalse(condition["target_met"])
            for row in rows:
                if row["method_label"] == "ERM-Perfect":
                    row["test_accuracy"] = .85
            write_rows(directory, rows)
            self.assertFalse(ranking_outcome(directory)["target_met_in_every_condition"])
            for row in rows:
                if row["method_label"] == "ERM-Perfect":
                    row["test_accuracy"] = .84
            write_rows(directory, rows)
            self.assertTrue(ranking_outcome(directory)["target_met_in_every_condition"])

    def test_missing_seed_duplicate_foreign_label_or_nonfinite_cannot_claim_win(self):
        base = ranking_rows({"T1-Matched": [.9, .9]})
        cases = {
            "empty": [],
            "missing seed": base[:-1],
            "duplicate": base + [dict(base[0])],
            "foreign label": [{**row, "method_label": "Unexpected"} if row["method_label"] == "L2-Perfect" else row
                              for row in base],
            "nonfinite": [{**row, "test_accuracy": float("nan")} if index == 0 else row
                          for index, row in enumerate(base)],
            "failed test": [{**row, "test_status": "failed"} if index == 0 else row
                            for index, row in enumerate(base)],
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in cases.items():
                with self.subTest(name=name):
                    write_rows(directory, rows)
                    with self.assertRaises(ValueError):
                        ranking_outcome(directory)

    def test_manifest_prevents_omitting_entire_seed_or_condition(self):
        rows = ranking_rows({"T1-Matched": [.9, .9]})
        rows += ranking_rows({"T1-Matched": [.9, .9]}, group="mnist_analog")
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_manifest(directory, rows)
            write_rows(directory, rows)
            self.assertTrue(ranking_outcome(directory, manifest=manifest)["target_met_in_every_condition"])
            for incomplete in ([row for row in rows if row["group"] != "mnist_analog"],
                               [row for row in rows if row["seed"] != 73102]):
                write_rows(directory, incomplete)
                with self.assertRaises(ValueError):
                    ranking_outcome(directory, manifest=manifest)


class PipelineOrchestrationTests(unittest.TestCase):
    def fake_run(self, command, log, environment):
        self.commands.append(list(command))
        module = command[command.index("-m") + 1]
        if module == "journal.build_bayesian_campaign":
            stage = command[command.index("--stage") + 1]
            manifest = Path(command[command.index("--output") + 1])
            manifest.write_text(json.dumps(dict(task_id=stage)) + "\n")
            manifest.with_suffix(".plan.json").write_text(json.dumps(dict(tasks=1)))
        elif module == "journal.campaign":
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True, exist_ok=True)
            if self.fail_stage == root.name:
                raise subprocess.CalledProcessError(1, command)
            summary = dict(completed_trials=1, failed_trials=[], failed_evaluations=[])
            if self.incomplete_evaluation == root.name:
                summary["failed_evaluations"] = [dict(task_id="failed")]
            (root / "summary.json").write_text(json.dumps(summary))
            (root / "training_results.json").write_text(json.dumps([
                dict(task_id=root.name, peak_rss_mb=128., seconds=.01)]))
            # A deliberately distinct test sentinel must never become the
            # frozen selection source, including when it exists on disk.
            (root / "test_results.json").write_text(json.dumps([dict(test_only_sentinel=True)]))

    def run_mock_pipeline(self, directory):
        self.commands = []
        with patch("journal.run_bayesian_campaign.run", side_effect=self.fake_run), \
                patch("journal.run_bayesian_campaign.subprocess.check_output", side_effect=["abc123\n", ""]), \
                patch("journal.run_bayesian_campaign.ranking_outcome", return_value={"conditions": []}), \
                patch("journal.batch_campaign.validate_cpu_budget", return_value=64):
            return pipeline(directory, workers=32, threads=2, confirm_draws=20, sweep_draws=10)

    def setUp(self):
        self.fail_stage = None
        self.incomplete_evaluation = None

    def test_stages_order_validation_only_screen_then_frozen_full_risk_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.run_mock_pipeline(directory)
            self.assertEqual(state["status"], "complete")
            campaigns = [command for command in self.commands if "journal.campaign" in command]
            self.assertEqual([Path(command[command.index("--output-root") + 1]).name
                              for command in campaigns], list(STAGES))
            for stage, command in zip(STAGES, campaigns):
                self.assertEqual(command[command.index("--evaluate-top") + 1],
                                 "0" if stage in {"pilot", "screen"} else "1")
                self.assertEqual("--full-risks" in command, stage not in {"pilot", "screen"})
                self.assertIn("--resume", command)
            builders = [command for command in self.commands if "journal.build_bayesian_campaign" in command]
            for stage, command in zip(STAGES, builders):
                self.assertEqual(command[command.index("--stage") + 1], stage)
                if stage in {"pilot", "screen"}:
                    self.assertNotIn("--from-results", command)
                else:
                    self.assertEqual(Path(command[command.index("--from-results") + 1]).name,
                                     "validation_selection_results.json")
                    self.assertEqual(Path(command[command.index("--from-manifest") + 1]).name,
                                     "screen.jsonl")
            plots = [command for command in self.commands if "journal.plot_bayesian_campaign" in command]
            self.assertEqual(len(plots), 3)
            self.assertIn("journal.bayesian_bounds", self.commands[-1])
            bounds = self.commands[-1]
            self.assertEqual(bounds.count("--campaign"), 3)
            status = json.loads(Path(directory, "pipeline_status.json").read_text())
            self.assertTrue(all(stage["status"] == "complete" for stage in status["stages"].values()))
            frozen = json.loads(Path(directory, "validation_selection_results.json").read_text())
            training = json.loads(Path(directory, "screen", "training_results.json").read_text())
            test = json.loads(Path(directory, "screen", "test_results.json").read_text())
            self.assertEqual(frozen, training)
            self.assertNotEqual(frozen, test)

    def test_training_failure_stops_before_selection_and_preserves_failure_status(self):
        self.fail_stage = "screen"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_mock_pipeline(directory)
            self.assertFalse(any("confirm" in command for command in self.commands))
            self.assertFalse(any("journal.bayesian_bounds" in command for command in self.commands))
            status = json.loads(Path(directory, "pipeline_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("CalledProcessError", status["error"])

    def test_failed_confirmation_evaluation_stops_before_figure_or_sweep(self):
        self.incomplete_evaluation = "confirm"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.run_mock_pipeline(directory)
            self.assertFalse(any("journal.plot_bayesian_campaign" in command for command in self.commands))
            self.assertFalse(any("m_sweep" in command for command in self.commands))

    def test_measured_pilot_memory_gate_stops_before_screen(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "6000"}):
            # Mock worker RSS128MiB x32, 25% pool headroom and 2GiB
            # coordinator allowance projects7168MiB, above this allocation.
            with self.assertRaisesRegex(RuntimeError, "Pilot peak RSS"):
                self.run_mock_pipeline(directory)
            self.assertFalse(any("screen" in command for command in self.commands))
            state = json.loads(Path(directory, "pipeline_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertGreater(state["pilot_resources"]["projected_pool_rss_with_headroom_mb"], 6000)


if __name__ == "__main__":
    unittest.main()
