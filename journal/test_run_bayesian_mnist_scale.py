"""Exercise isolated MNIST orchestration without training or dataset access."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from journal.build_bayesian_campaign import screen_entries
from journal.build_bayesian_mnist_scale import extra_entries, manifest_text
from journal.run_bayesian_mnist_scale import (CONTROL_FILES, SOURCE_FILES, STAGES,
    fingerprint, memory_projection, pilot_entries, pipeline, verify_sources)


def records(entries):
    return [dict(task_id=item["task_id"], config=item["config"], status="complete",
        group=item["group"], method_label=item["method_label"], family=item["family"],
        variant_id=item["variant_id"], stage=item["stage"], seed=item["config"]["seed"],
        validation_accuracy=.9 if item["config"].get("rho_lr_scale") == 0 else .8,
        validation_cross_entropy=.4, selected_epoch=1, seconds=1., peak_rss_mb=450.) for item in entries]


class MnistControlPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mnist-control-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main, self.output = self.root / "main", self.root / "mnist_control"
        (self.main / "manifests").mkdir(parents=True)
        (self.main / "screen").mkdir()
        manifest = self.main / "manifests" / "screen.jsonl"
        entries = screen_entries(datasets=("mnist",))
        manifest.write_text(manifest_text(entries))
        (self.main / "screen" / "training_results.json").write_text(json.dumps(records(entries)))
        (self.main / "pipeline_status.json").write_text(json.dumps(dict(git_sha="mainsha", stages={
            "screen": dict(manifest_sha256=fingerprint(manifest)["sha256"])})))
        # Invalid official-test JSON would fail if the coordinator tried to
        # use the main confirmation/test cohort as a selection source.
        (self.main / "confirm").mkdir()
        (self.main / "confirm" / "test_results.json").write_text("FORBIDDEN OFFICIAL TEST SENTINEL")
        self.verification = dict(main_revision="mainsha", execution_revision="controlsha",
                                 files={name: "same blob" for name in SOURCE_FILES}, control_files={})
        self.commands, self.fail_stage = [], None

    def fake_command(self, arguments, log, environment):
        self.commands.append(list(arguments))
        stage = Path(arguments[arguments.index("--output-root") + 1])
        if stage.name == self.fail_stage:
            raise subprocess.CalledProcessError(1, arguments)
        stage.mkdir(parents=True, exist_ok=True)
        manifest = Path(arguments[arguments.index("--manifest") + 1])
        entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.assertEqual(environment["JOURNAL_MANIFEST_SHA256"], fingerprint(manifest)["sha256"])
        self.assertEqual(environment["JOURNAL_GIT_SHA"], "controlsha")
        (stage / "training_results.json").write_text(json.dumps(records(entries)))
        (stage / "summary.json").write_text(json.dumps(dict(training_trials=len(entries),
            completed_trials=len(entries), failed_trials=[], failed_evaluations=[])))

    def fake_validate(self, stage, manifest, revision, name):
        self.assertEqual(revision, "controlsha")
        return json.loads((stage / "training_results.json").read_text())

    def run_pipeline(self, memory="16384", submitted="controlsha"):
        with patch("journal.run_bayesian_mnist_scale.verify_sources", return_value=self.verification), \
             patch("journal.batch_campaign.validate_cpu_budget", return_value=32), \
             patch("journal.run_bayesian_mnist_scale.command", side_effect=self.fake_command), \
             patch("journal.run_bayesian_mnist_scale.validate_stage", side_effect=self.fake_validate), \
             patch.dict(os.environ, {"JOURNAL_GIT_SHA": submitted, "SLURM_MEM_PER_NODE": memory}):
            return pipeline(self.main, self.output, "mainsha")

    def test_exact_stage_order_no_test_screen_and_full_risk_final_draws(self):
        before = {str(path.relative_to(self.main)): path.read_bytes() for path in self.main.rglob("*") if path.is_file()}
        state = self.run_pipeline()
        self.assertEqual(state["status"], "complete")
        self.assertEqual([Path(cmd[cmd.index("--output-root")+1]).name for cmd in self.commands], list(STAGES))
        for stage, cmd in zip(STAGES, self.commands):
            self.assertEqual(cmd[cmd.index("--evaluate-top")+1], "0" if stage in {"pilot", "screen"} else "1")
            self.assertEqual("--full-risks" in cmd, stage not in {"pilot", "screen"})
            self.assertEqual(cmd[cmd.index("--eval-draws")+1], "20" if stage == "confirm" else "10")
        protocol = json.loads((self.output / "selected" / "selection_protocol.json").read_text())
        self.assertEqual(protocol["combined_trials"], 1024)
        self.assertEqual(protocol["candidates_per_method_regime"], 32)
        self.assertEqual(before, {str(path.relative_to(self.main)): path.read_bytes() for path in self.main.rglob("*") if path.is_file()})

    def test_resume_reuses_frozen_original_even_when_live_main_advances(self):
        self.run_pipeline()
        frozen = self.output / "inputs" / "original" / "original_training_results.json"
        before = frozen.read_bytes()
        # The live main file can advance or change after the historical input
        # was captured. Resume must not refresh selection's frozen source.
        (self.main / "screen" / "training_results.json").write_text("LIVE MAIN HAS ADVANCED")
        self.commands.clear()
        state = self.run_pipeline()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(self.commands, [])
        self.assertEqual(frozen.read_bytes(), before)

    def test_memory_projection_stops_before_added_screen(self):
        with self.assertRaisesRegex(ValueError, "peak RSS"):
            self.run_pipeline(memory="6000")
        self.assertEqual(len(self.commands), 1)
        state = json.loads((self.output / "control_status.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_stage"], "pilot")

    def test_failure_stops_before_selection_or_any_official_test(self):
        self.fail_stage = "screen"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.assertEqual(len(self.commands), 2)
        self.assertFalse((self.output / "selected").exists())
        self.assertEqual(json.loads((self.output / "control_status.json").read_text())["status"], "failed")

    def test_submitted_revision_mismatch_fails_before_any_owned_output(self):
        with self.assertRaisesRegex(ValueError, "Submitted"):
            self.run_pipeline(submitted="wrong")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.commands, [])

    def test_pilot_has_all_sixteen_full_shapes_with_positive_fixed_variance(self):
        entries = pilot_entries(extra_entries(screen_entries(datasets=("mnist",))))
        self.assertEqual(len(entries), 16)
        self.assertEqual(len({(entry["group"], entry["method_label"]) for entry in entries}), 16)
        for entry in entries:
            cfg = entry["config"]
            self.assertEqual((cfg["train_size"], cfg["val_size"], cfg["epochs"], cfg["seed"]), (55000, 5000, 1, 76200))
            self.assertEqual((cfg["predictor"], cfg["posterior_std"], cfg["rho_lr_scale"]), ("gaussian", .001, 0.))
            self.assertFalse(entry["scientific_result"])


class MnistControlSourceTests(unittest.TestCase):
    def test_seven_main_source_blobs_must_match_committed_and_working_files(self):
        with tempfile.TemporaryDirectory(prefix="mnist-control-git-") as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
            git("init", "--quiet")
            for name in (*SOURCE_FILES, *CONTROL_FILES):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixed source\n")
            git("add", "journal")
            git("-c", "user.name=Control Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "source")
            source = git("rev-parse", "HEAD")
            self.assertEqual(len(verify_sources(source, root)["files"]), 7)
            (root / SOURCE_FILES[0]).write_text("# changed training source\n")
            git("add", SOURCE_FILES[0])
            git("-c", "user.name=Control Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "changed")
            with self.assertRaisesRegex(ValueError, "differs from running main"):
                verify_sources(source, root)


if __name__ == "__main__":
    unittest.main()
