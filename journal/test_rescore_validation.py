"""Synthetic 32-row metadata and mocked evaluate subprocesses; no inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from journal import rescore_validation as rescore
from journal.run_experiment import normalize_config


class ValidationRescoreTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        entries, rows = [], []
        for label, method in rescore.METHODS.items():
            for variant in range(4):
                for seed in (7101, 7102):
                    task = f"{method}_v{variant}_s{seed}"
                    config = normalize_config(dict(
                        dataset="cifar10", architecture="cnn_mid", method=method,
                        training_loss="bounded_ce_smooth", selection_metric="accuracy",
                        train_size=45000, val_size=5000, epochs=100, width=128, m=100,
                        threads=2, device="cpu", seed=seed, channel_seed=seed+10000,
                        eval_seed=seed+20000, lr=(variant+1)*.001,
                        cpu_flush_denormal=True,
                        train_channel=dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.),
                        deployment_channel=dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)))
                    entry = dict(task_id=task, group="fixed", method_label=label, family=(
                        "baseline" if method in {"erm", "l2"} else "proposed"),
                        variant_id=f"v{variant}", stage="search", config=config)
                    entries.append(entry)
                    checkpoint = root / task
                    checkpoint.mkdir()
                    history = [dict(epoch=epoch, validation=dict(
                        accuracy=.4+variant*.01+epoch*.00001,
                        cross_entropy=1.5-variant*.01-epoch*.00001)) for epoch in range(1, 101)]
                    metadata = dict(config=config, selected_epoch=100, total_seconds=1.,
                                    peak_rss_mb=1234., git_sha="synthetic",
                                    split=dict(train_indices_hash="train", validation_indices_hash="validation"),
                                    training_channel_bank_hash=f"bank_{seed}")
                    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
                    (checkpoint / "history.json").write_text(json.dumps(history))
                    # completed_trial verifies existence only; the mock never loads it.
                    (checkpoint / "weights.pt").write_bytes(b"synthetic checkpoint; not a model")
                    forbidden = checkpoint / "test_deployment"
                    forbidden.mkdir()
                    (forbidden / "evaluation.json").write_text("TEST REPORT MUST NEVER BE READ")
                    selected = history[-1]["validation"]
                    rows.append(dict(task_id=task, group="fixed", method_label=label,
                                     family=entry["family"], variant_id=entry["variant_id"],
                                     stage="search", seed=seed, config=config,
                                     checkpoint=str(checkpoint.resolve()), status="complete", resumed=True,
                                     selected_epoch=100, validation_accuracy=selected["accuracy"],
                                     validation_cross_entropy=selected["cross_entropy"],
                                     seconds=1., peak_rss_mb=1234., git_sha="synthetic"))
        manifest = root / "aggregate32.jsonl"
        manifest.write_text("".join(json.dumps(entry)+"\n" for entry in entries))
        (root / "training_results.json").write_text(json.dumps(rows))
        return root, manifest, rows

    def fake_evaluate(self, command, **kwargs):
        self.assertEqual(command[command.index("--mode")+1], "evaluate")
        self.assertEqual(command[command.index("--split")+1], "validation")
        self.assertEqual(command[command.index("--risk-scope")+1], "deployment")
        checkpoint = Path(command[command.index("--checkpoint")+1])
        output = Path(command[command.index("--output")+1])
        override = json.loads(Path(command[command.index("--config")+1]).read_text())
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        config = {**metadata["config"], **override}
        self.assertEqual(override["eval_seed"], metadata["config"]["eval_seed"]+rescore.SEED_OFFSET)
        self.assertEqual(override["eval_draws"], 100)
        self.assertEqual(override["threads"], 2)
        self.assertEqual(kwargs["env"]["OMP_PROC_BIND"], "false")
        variant = int(checkpoint.name.split("_v")[1].split("_")[0])
        accuracy = (.55, .65, .75, .6)[variant]
        replicate = dict(accuracy=accuracy, cross_entropy=1.-accuracy,
                         bounded_loss=.2, feature_power_per_coordinate=1.)
        report = dict(checkpoint=str(checkpoint.resolve()), split="validation", risk_scope="deployment",
                      evaluation_config=config, selected_epoch=100,
                      heldout_gibbs_mc=dict(**replicate, examples=5000, draws=100,
                          channel_source="deployment", posterior_rule=(
                              "one weight draw fixed across entire loader per replicate"),
                          bounded_loss_definition=(
                              "-log(epsilon+(1-epsilon)*p_y)/(-log(epsilon)); epsilon=0.01"),
                          replicate_statistics=[replicate.copy() for _ in range(100)]))
        (output / "evaluation.json").write_text(json.dumps(report))
        return mock.Mock(returncode=0)

    def run_mocked(self, manifest, root):
        with mock.patch.object(rescore, "verify_submission_snapshot") as verify, \
                mock.patch.object(rescore.subprocess, "run", side_effect=self.fake_evaluate) as process, \
                mock.patch("builtins.print"):
            result = rescore.rescore_search(manifest, root)
        self.assertEqual(verify.call_count, 2)
        return result, process.call_count

    def test_complete_search_rescores_validation_preserves_sources_and_uses_fresh_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, original = self.fixture(directory)
            protected = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            summary, calls = self.run_mocked(manifest, root)
            self.assertEqual(calls, 32)
            self.assertEqual(summary["training_trials"], 32)
            self.assertEqual(summary["completed_trials"], 32)
            self.assertEqual(summary["rescored_trials"], 32)
            self.assertFalse(summary["selection_uses_test"])
            self.assertTrue(summary["no_training_fallback"])
            self.assertEqual(summary["validation_draws"], 100)
            self.assertFalse(summary["failed_trials"] or summary["failed_evaluations"])
            self.assertEqual(len(summary["selected_variants"]), 4)
            self.assertTrue(all(choice["variant_id"] == "v2" and choice["complete_seed_set"]
                                and choice["seeds"] == [7101, 7102]
                                for choice in summary["selected_variants"]))
            rescored = rescore.read_json(root / "validation_rescored_training_results.json")
            self.assertEqual(len(rescored), 32)
            for before, after in zip(original, rescored):
                self.assertEqual(after["original_validation_accuracy"], before["validation_accuracy"])
                self.assertEqual(after["original_validation_cross_entropy"], before["validation_cross_entropy"])
                for key, value in before.items():
                    if key not in {"validation_accuracy", "validation_cross_entropy"}:
                        self.assertEqual(after[key], value)
            for path, contents in protected.items():
                self.assertEqual(path.read_bytes(), contents)
            again, calls = self.run_mocked(manifest, root)
            self.assertEqual(calls, 0)
            self.assertEqual(again["selected_variants"], summary["selected_variants"])

    def test_missing_row_or_checkpoint_rejects_before_any_subprocess_without_training_fallback(self):
        for missing in ("row", "checkpoint"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root, manifest, rows = self.fixture(directory)
                if missing == "row":
                    (root / "training_results.json").write_text(json.dumps(rows[:-1]))
                else:
                    (Path(rows[-1]["checkpoint"]) / "weights.pt").unlink()
                with mock.patch.object(rescore, "verify_submission_snapshot"), \
                        mock.patch.object(rescore.subprocess, "run") as process, \
                        self.assertRaises(ValueError):
                    rescore.rescore_search(manifest, root)
                process.assert_not_called()
                self.assertFalse((root / "validation_selection.json").exists())
                self.assertFalse(any(root.glob("*/validation_rescore")))

    def test_mismatched_cache_never_falls_back_to_original_scores_or_retrains(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, rows = self.fixture(directory)
            self.run_mocked(manifest, root)
            path = Path(rows[-1]["checkpoint"]) / "validation_rescore/evaluation.json"
            report = rescore.read_json(path)
            report["evaluation_config"]["eval_seed"] += 1
            path.write_text(json.dumps(report))
            stale = path.read_bytes()
            summary, calls = self.run_mocked(manifest, root)
            self.assertEqual(calls, 0)
            self.assertEqual(summary["rescored_trials"], 31)
            self.assertEqual(len(summary["failed_evaluations"]), 1)
            self.assertEqual(summary["selected_variants"], [])
            self.assertEqual(summary["variant_validation_summaries"], [])
            self.assertEqual(path.read_bytes(), stale)

    def test_child_failure_never_manufactures_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, _ = self.fixture(directory)
            with mock.patch.object(rescore, "verify_submission_snapshot"), \
                    mock.patch.object(rescore.subprocess, "run", side_effect=RuntimeError("mock inference failed")) as process, \
                    mock.patch("builtins.print"):
                summary = rescore.rescore_search(manifest, root)
            self.assertEqual(process.call_count, 32)
            self.assertEqual(summary["rescored_trials"], 0)
            self.assertEqual(len(summary["failed_evaluations"]), 32)
            self.assertEqual(summary["selected_variants"], [])
            self.assertTrue(all(call.args[0][call.args[0].index("--mode")+1] == "evaluate"
                                for call in process.call_args_list))

    def test_actual_mixed_training_threads_are_preserved_with_separate_inference_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, rows = self.fixture(directory)
            entries = [json.loads(line) for line in manifest.read_text().splitlines()]
            # Four declared control reruns used one training thread. All other
            # anchors/proposals retain their recorded two-thread execution.
            changed = {row["task_id"] for row in rows if row["method_label"] in
                       {"ERM-Matched", "L2-Matched"} and row["variant_id"] == "v2"}
            self.assertEqual(len(changed), 4)
            for entry, row in zip(entries, rows):
                if row["task_id"] in changed:
                    entry["config"]["threads"] = row["config"]["threads"] = 1
                    metadata_path = Path(row["checkpoint"]) / "metadata.json"
                    metadata = rescore.read_json(metadata_path)
                    metadata["config"]["threads"] = 1
                    metadata_path.write_text(json.dumps(metadata))
            manifest.write_text("".join(json.dumps(entry)+"\n" for entry in entries))
            (root / "training_results.json").write_text(json.dumps(rows))
            summary, calls = self.run_mocked(manifest, root)
            self.assertEqual(calls, 32)
            self.assertFalse(summary["failed_evaluations"])
            self.assertEqual(summary["inference_threads"], 2)
            for row in rescore.read_json(root / "validation_rescored_training_results.json"):
                actual = 1 if row["task_id"] in changed else 2
                self.assertEqual(row["config"]["threads"], actual)
                self.assertEqual(summary["training_threads_by_task"][row["task_id"]], actual)
                report = rescore.read_json(row["validation_rescore_report"])
                self.assertEqual(report["evaluation_config"]["threads"], 2)


if __name__ == "__main__":
    unittest.main()
