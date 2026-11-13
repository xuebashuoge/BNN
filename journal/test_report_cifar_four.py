"""Synthetic saved-report audits only: no Torch, weights, training or inference."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

from journal import report_cifar_four as report


class FourMethodReportTests(unittest.TestCase):
    def fixture(self, directory, scores=None):
        root = Path(directory) / "campaign"
        root.mkdir()
        scores = scores or {"ERM-Matched": [.6]*3, "L2-Matched": [.59]*3,
                            "T1-Matched": [.7, .6, .56], "T2-Matched": [.59]*3}
        rows, entries, choices = [], [], []
        channel = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)
        for label, algorithm in report.METHODS.items():
            method_rows = []
            for index, seed in enumerate(report.SEEDS):
                task = f"{algorithm}_s{seed}"
                family = "baseline" if algorithm in {"erm", "l2"} else "proposed"
                config = dict(dataset="cifar10", architecture="cnn_spatial3", train_size=45000,
                              val_size=5000, epochs=25, batch_size=128, width=128, m=100,
                              data_seed=20261111, deployment_channel=channel.copy(), train_channel=channel.copy(),
                              training_loss="bounded_ce_smooth", loss_epsilon=.01, clip_ce=10.,
                              prior_std=1., delta=.05, selection_metric="accuracy", validation_seed_policy="fixed",
                              val_draws=2, eval_draws=100, device="cpu", threads=1, cpu_flush_denormal=True,
                              method=algorithm, seed=seed, channel_seed=seed+10000, eval_seed=seed+20000,
                              lr=.003, l2=1e-6, posterior_std=.003, rho_lr_scale=.1,
                              eta_channel=.001, eta_data=.01 if algorithm == "proposed" else .002)
                checkpoint = root / task
                checkpoint.mkdir()
                # The report must not read this or invoke a model loader.
                (checkpoint / "weights.pt").write_text("DO NOT LOAD SYNTHETIC WEIGHTS")
                split = dict(train_size=45000, validation_size=5000, data_seed=20261111,
                             train_indices_hash="fixed-training-hash", validation_indices_hash="fixed-validation-hash",
                             transform="fixed pixel/127.5-1")
                network = dict(transmitted_shape=[8, 4, 4], real_channel_uses=128,
                               transmitter_parameter_count=24104, receiver_parameter_count=137162,
                               description="three-convolution spatial transmitter and convolutional receiver")
                metadata = dict(config=copy.deepcopy(config), git_sha="frozen-code", git_dirty=False,
                                selected_epoch=25, parameter_count=161266, network_split=network, split=split,
                                training_channel_bank_hash=f"paired-bank-{seed}",
                                cpu_numerics=dict(flush_denormal_requested=True, flush_denormal_supported=True),
                                cpu_threading=dict(torch_intraop_threads=1))
                history = [dict(epoch=epoch, validation=dict(accuracy=.4+epoch*.001,
                                                            cross_entropy=1.-epoch*.001)) for epoch in range(1, 26)]
                accuracy = scores[label][index]
                metrics = dict(accuracy=accuracy, bounded_loss=1.-accuracy,
                               cross_entropy=2.-accuracy, feature_power_per_coordinate=3.+index)
                evaluation = dict(checkpoint=str(checkpoint), split="test", risk_scope="deployment",
                                  evaluation_config={**config, "eval_seed": config["eval_seed"]+1_000_003},
                                  selected_epoch=25, heldout_gibbs_mc=dict(
                                      **metrics, examples=10000, draws=100, channel_source="deployment",
                                      posterior_rule="one weight draw fixed across entire loader per replicate",
                                      bounded_loss_definition="-log(epsilon+(1-epsilon)*p_y)/(-log(epsilon)); epsilon=0.01",
                                      replicate_statistics=[metrics.copy() for _ in range(100)]))
                (checkpoint / "metadata.json").write_text(json.dumps(metadata))
                (checkpoint / "history.json").write_text(json.dumps(history))
                (checkpoint / "test_deployment").mkdir()
                (checkpoint / "test_deployment/evaluation.json").write_text(json.dumps(evaluation))
                row = dict(task_id=task, method_label=label, group="frozen-group", variant_id=algorithm,
                           family=family, seed=seed, config=config, selected_epoch=25, checkpoint=str(checkpoint),
                           git_sha="frozen-code", status="complete", test_status="complete",
                           test_accuracy=accuracy, test_cross_entropy=metrics["cross_entropy"])
                rows.append(row)
                method_rows.append(row)
                entries.append({key: copy.deepcopy(row[key]) for key in (
                    "task_id", "method_label", "group", "variant_id", "family", "config")})
                entries[-1]["selection_uses_test"] = False
            choices.append(dict(method_label=label, group="frozen-group", variant_id=algorithm,
                                rank=1, complete_seed_set=True, seeds=report.SEEDS,
                                task_ids=[row["task_id"] for row in method_rows], config=method_rows[0]["config"]))
        summary = dict(training_trials=12, completed_trials=12, failed_trials=[], failed_evaluations=[],
                       selected_variants=choices, selection_uses_test=False)
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "test_results.json").write_text(json.dumps(rows))
        freeze = dict(status="selected", selection_uses_test=False, selected_group="frozen-group",
                      selected_architecture="cnn_spatial3", confirmation_seeds=report.SEEDS,
                      confirmation_entries=entries, groups=[dict(group="frozen-group", best_proposed_method="T2-Matched")],
                      selection_rule="validation-only frozen discovery", interpretation="fresh seed confirmation on reused data")
        selection = Path(directory) / "freeze.json"
        selection.write_text(json.dumps(freeze))
        return root, selection

    def test_recomputes_means_sd_preserves_primary_and_writes_all_four_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root, selection = self.fixture(directory)
            original_read = Path.read_bytes

            def guarded_read(path):
                if path.name == "weights.pt":
                    raise AssertionError("Weights must never be read")
                return original_read(path)

            with mock.patch.object(Path, "read_bytes", guarded_read):
                result = report.load_report(root, selection)
            self.assertEqual(len(result["methods"]), 4)
            self.assertTrue(result["goal_met"])
            self.assertEqual(result["frozen_primary_from_validation"], "T2-Matched")
            self.assertFalse(result["frozen_primary_comparison"]["goal_met"])
            self.assertEqual(result["observed_best_proposed_by_test_mean"], "T1-Matched")
            self.assertFalse(result["fixed_algorithm_comparisons"]["T1-Matched"]["all_seeds_strictly_beat_both_baselines"])
            t1 = next(row for row in result["methods"] if row["method"] == "T1-Matched")
            self.assertAlmostEqual(t1["accuracy"]["mean"], statistics.fmean([.7, .6, .56]))
            self.assertAlmostEqual(t1["accuracy"]["sd"], statistics.stdev([.7, .6, .56]))
            self.assertFalse(result["selection_uses_test"])
            self.assertTrue(result["hyperconfigs_locked"] and result["dataset_reused_from_prior_study"])
            self.assertEqual(len(result["source_hashes"]), 39)
            for source, details in result["source_hashes"].items():
                raw = Path(source).read_bytes()
                self.assertEqual(details, dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            prefix = Path(directory) / "reports/confirmation"
            report.write_report(result, prefix)
            with Path(str(prefix)+".csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual([row["method"] for row in csv_rows], list(report.METHODS))
            self.assertNotIn("replicate_statistics", Path(str(prefix)+".json").read_text())

    def test_switching_proposed_objective_between_seeds_cannot_manufacture_win(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = {"ERM-Matched": [.6]*3, "L2-Matched": [.59]*3,
                      "T1-Matched": [.75, .5, .5], "T2-Matched": [.5, .75, .5]}
            root, selection = self.fixture(directory, scores)
            result = report.load_report(root, selection)
            self.assertFalse(result["goal_met"])
            self.assertTrue(all(not value["goal_met"] for value in result["fixed_algorithm_comparisons"].values()))
            self.assertGreater(statistics.fmean(max(a, b) for a, b in zip(scores["T1-Matched"], scores["T2-Matched"])), .6)

    def test_tied_mean_is_not_a_strict_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root, selection = self.fixture(directory, {label: [.6]*3 for label in report.METHODS})
            self.assertFalse(report.load_report(root, selection)["goal_met"])

    def test_incomplete_mismatched_or_corrupted_sources_are_rejected(self):
        for problem in ("missing_row", "failed", "split", "draws", "replicate", "history", "bank", "network", "freeze", "unlocked_config"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root, selection = self.fixture(directory)
                rows = json.loads((root / "test_results.json").read_text())
                first = root / rows[0]["task_id"]
                if problem == "missing_row":
                    (root / "test_results.json").write_text(json.dumps(rows[:-1]))
                elif problem == "failed":
                    s = json.loads((root / "summary.json").read_text())
                    s["failed_evaluations"] = [dict(error="synthetic")]
                    (root / "summary.json").write_text(json.dumps(s))
                elif problem in {"split", "draws", "replicate"}:
                    path = first / "test_deployment/evaluation.json"
                    e = json.loads(path.read_text())
                    if problem == "split":
                        e["split"] = "validation"
                    elif problem == "draws":
                        e["heldout_gibbs_mc"]["draws"] = 10
                    else:
                        e["heldout_gibbs_mc"]["replicate_statistics"][0]["accuracy"] += .01
                    path.write_text(json.dumps(e))
                elif problem == "history":
                    path = first / "history.json"
                    path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
                elif problem in {"bank", "network"}:
                    path = first / "metadata.json"
                    m = json.loads(path.read_text())
                    if problem == "bank":
                        m["training_channel_bank_hash"] = "different-bank"
                    else:
                        m["network_split"]["real_channel_uses"] = 2048
                    path.write_text(json.dumps(m))
                else:
                    f = json.loads(selection.read_text())
                    if problem == "freeze":
                        f["selection_uses_test"] = True
                    else:
                        f["confirmation_entries"][0]["config"]["lr"] = .123
                    selection.write_text(json.dumps(f))
                with self.assertRaises((ValueError, OSError)):
                    report.load_report(root, selection)
                self.assertFalse((root / "confirmation.json").exists())


if __name__ == "__main__":
    unittest.main()
