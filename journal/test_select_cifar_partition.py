"""Synthetic metadata checks for partition selection; no models or datasets."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = REPO_ROOT / "journal/configs/cifar_partition25.jsonl"
EVAL_SOURCE = REPO_ROOT / "journal/configs/cifar_partition_validation100.jsonl"
LOSS_DEFINITION = "-log(epsilon+(1-epsilon)*p_y)/(-log(epsilon)); epsilon=0.01"
POSTERIOR_RULE = "one weight draw fixed across entire loader per replicate"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_manifest(path, entries):
    path.write_text("".join(json.dumps(entry, allow_nan=False) + "\n" for entry in entries))


class PartitionFixture:
    """Clone the actual control grid, replacing artifacts by small JSON files."""

    def __init__(self, directory, scores=None, evaluation_threads=1):
        self.root = Path(directory)
        self.train_root = self.root / "training"
        self.scores_root = self.root / "validation100"
        self.train_manifest = self.root / "train.jsonl"
        self.eval_manifest = self.root / "evaluate.jsonl"
        self.output = self.root / "selection.json"
        self.confirm_manifest = self.root / "confirm.jsonl"
        self.entries = [json.loads(line) for line in TRAIN_SOURCE.read_text().splitlines() if line.strip()]
        sources = [json.loads(line) for line in EVAL_SOURCE.read_text().splitlines() if line.strip()]
        by_source = {entry["source_task_id"]: entry for entry in sources}
        self.evaluations = []
        self.report_paths = []
        self.metadata_paths = []
        self.history_paths = []
        self.scores = scores or {
            "cnn_mid_relu": (.70, .75),
            "cnn_rxconv": (.78, .80),
            "cnn_spatial3": (.83, .82),
        }
        method_indices = {}
        for entry in self.entries:
            config = entry["config"]
            architecture = config["architecture"]
            baseline, proposed = self.scores[architecture]
            label = entry["method_label"]
            key = (architecture, label)
            index = method_indices.get(key, 0)
            method_indices[key] = index + 1
            accuracies = {
                "ERM-Matched": [baseline - .02, baseline - .01],
                "L2-Matched": [baseline - .01, baseline - .005, baseline],
                "T1-Matched": [proposed - .01, proposed, proposed - .02],
                "T2-Matched": [proposed - .004],
            }
            accuracy = accuracies[label][index]
            cross_entropy, bounded_loss = 1.5 - accuracy, .8 - .5 * accuracy
            checkpoint = self.train_root / entry["task_id"]
            metadata_path = checkpoint / "metadata.json"
            history_path = checkpoint / "history.json"
            split = dict(train_indices_hash=hashlib.sha256(b"training indices").hexdigest(),
                         validation_indices_hash=hashlib.sha256(b"validation indices").hexdigest(),
                         train_size=45000, validation_size=5000)
            metadata = dict(config=deepcopy(config),
                            config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                            selected_epoch=25, total_seconds=1., peak_rss_mb=1234., git_sha="synthetic",
                            split=split, training_channel_bank_hash=hashlib.sha256(b"common bank").hexdigest(),
                            selected_validation_accuracy=accuracy,
                            selected_validation_cross_entropy=cross_entropy,
                            selected_validation_bounded_loss=bounded_loss)
            history = [dict(epoch=epoch, epoch_seconds=.01,
                            validation=dict(accuracy=accuracy - (25 - epoch) * .001,
                                            cross_entropy=cross_entropy + (25 - epoch) * .001,
                                            bounded_loss=bounded_loss + (25 - epoch) * .001))
                       for epoch in range(1, 26)]
            write_json(metadata_path, metadata)
            write_json(history_path, history)
            (checkpoint / "weights.pt").write_bytes(b"synthetic sentinel; never deserialize")
            evaluation = deepcopy(by_source[entry["task_id"]])
            evaluation.update(checkpoint=str(checkpoint), split="validation", risk_scope="deployment")
            evaluation["config"]["threads"] = evaluation_threads
            self.evaluations.append(evaluation)
            metric = dict(accuracy=accuracy, cross_entropy=cross_entropy,
                          bounded_loss=bounded_loss, feature_power_per_coordinate=3.)
            report = dict(checkpoint=str(checkpoint.resolve()), split="validation", risk_scope="deployment",
                          evaluation_config={**deepcopy(config), **evaluation["config"]}, selected_epoch=25,
                          heldout_gibbs_mc={**metric, "draws": 100, "examples": 5000,
                                           "channel_source": "deployment", "posterior_rule": POSTERIOR_RULE,
                                           "bounded_loss_definition": LOSS_DEFINITION,
                                           "replicate_statistics": [deepcopy(metric) for _ in range(100)]})
            report_path = self.scores_root / evaluation["task_id"] / "evaluation.json"
            write_json(report_path, report)
            self.report_paths.append(report_path)
            self.metadata_paths.append(metadata_path)
            self.history_paths.append(history_path)
        self.save_manifests()

    def save_manifests(self):
        write_manifest(self.train_manifest, self.entries)
        write_manifest(self.eval_manifest, self.evaluations)

    def select(self, confirmation=True):
        from journal.select_cifar_partition import select_partition
        with ExitStack() as guards:
            for name in ("torch.load", "journal.run_experiment.load_data",
                         "journal.run_experiment.train", "journal.run_experiment.evaluate"):
                guards.enter_context(patch(name, side_effect=AssertionError(f"Forbidden execution: {name}")))
            return select_partition(self.train_manifest, self.eval_manifest, self.train_root,
                                    self.scores_root, self.output,
                                    self.confirm_manifest if confirmation else None)


class CifarPartitionSelectionTests(unittest.TestCase):
    def test_selects_highest_absolute_eligible_accuracy_and_retains_losing_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory)
            summary = fixture.select()
            self.assertEqual(summary["status"], "selected")
            self.assertEqual(summary["selected_architecture"], "cnn_rxconv")
            group = next(entry["group"] for entry in fixture.entries
                         if entry["config"]["architecture"] == "cnn_rxconv")
            self.assertEqual(summary["selected_group"], group)
            self.assertEqual(len(summary["rows"]), 27)
            self.assertEqual(len(summary["selected_variants"]), 12)
            groups = {item["architecture"]: item for item in summary["groups"]}
            self.assertEqual(set(groups), {"cnn_mid_relu", "cnn_rxconv", "cnn_spatial3"})
            self.assertTrue(groups["cnn_mid_relu"]["eligible"])
            self.assertTrue(groups["cnn_rxconv"]["eligible"])
            self.assertFalse(groups["cnn_spatial3"]["eligible"])
            self.assertGreater(groups["cnn_mid_relu"]["margin"], groups["cnn_rxconv"]["margin"])
            self.assertLess(groups["cnn_spatial3"]["margin"], 0.)
            for item in groups.values():
                self.assertEqual(len(item["selected_variants"]), 4)
                self.assertEqual(item["best_baseline_method"], "L2-Matched")
                self.assertEqual(item["best_proposed_method"], "T1-Matched")
                expected_base, expected_prop = fixture.scores[item["architecture"]]
                self.assertAlmostEqual(item["best_baseline_accuracy"], expected_base)
                self.assertAlmostEqual(item["best_proposed_accuracy"], expected_prop)
            self.assertEqual(json.loads(fixture.output.read_text())["selected_group"], group)

    def test_confirmation_resets_all_three_seed_streams_and_keeps_selected_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory)
            summary = fixture.select()
            confirmation = [json.loads(line) for line in fixture.confirm_manifest.read_text().splitlines()]
            self.assertEqual(len(confirmation), 12)
            self.assertEqual(len({entry["task_id"] for entry in confirmation}), 12)
            self.assertEqual({entry["method_label"] for entry in confirmation},
                             {"ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched"})
            for label in {entry["method_label"] for entry in confirmation}:
                rows = [entry for entry in confirmation if entry["method_label"] == label]
                self.assertEqual({entry["config"]["seed"] for entry in rows}, {8101, 8102, 8103})
            for entry in confirmation:
                config = entry["config"]
                self.assertEqual(entry["stage"], "partition_confirm")
                self.assertEqual(config["architecture"], summary["selected_architecture"])
                self.assertEqual((config["width"], config["epochs"], config["threads"]), (128, 25, 1))
                self.assertEqual((config["eval_draws"], config["val_draws"]), (100, 2))
                self.assertEqual(config["channel_seed"], config["seed"] + 10000)
                self.assertEqual(config["eval_seed"], config["seed"] + 20000)
                self.assertTrue(config["cpu_flush_denormal"])
                controls = [source for source in fixture.entries
                            if source["config"]["architecture"] == summary["selected_architecture"]
                            and source["method_label"] == entry["method_label"]]
                winner_index = {"ERM-Matched": 1, "L2-Matched": 2,
                                "T1-Matched": 1, "T2-Matched": 0}[entry["method_label"]]
                expected = {**controls[winner_index]["config"], "seed": config["seed"],
                            "channel_seed": config["seed"] + 10000,
                            "eval_seed": config["seed"] + 20000, "eval_draws": 100,
                            "val_draws": 2, "threads": 1}
                for key, value in expected.items():
                    self.assertEqual(config[key], value, f"Frozen selected setting changed: {key}")

    def test_no_eligible_group_publishes_complete_summary_without_confirmation(self):
        scores = {arch: (.75, .754) for arch in ("cnn_mid_relu", "cnn_rxconv", "cnn_spatial3")}
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory, scores)
            summary = fixture.select()
            self.assertEqual(summary["status"], "no_eligible_group")
            self.assertIsNone(summary["selected_group"])
            self.assertIsNone(summary["selected_architecture"])
            self.assertEqual(len(summary["rows"]), 27)
            self.assertEqual(len(summary["selected_variants"]), 12)
            self.assertTrue(all(not group["eligible"] for group in summary["groups"]))
            self.assertTrue(fixture.output.is_file())
            self.assertFalse(fixture.confirm_manifest.exists())

    def test_exact_half_percentage_point_margin_is_eligible(self):
        scores = {arch: (.50, .50) for arch in ("cnn_mid_relu", "cnn_rxconv", "cnn_spatial3")}
        scores["cnn_mid_relu"] = (.50, .505)
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory, scores)
            summary = fixture.select(confirmation=False)
            self.assertEqual(summary["status"], "selected")
            self.assertEqual(summary["selected_architecture"], "cnn_mid_relu")

    def test_evaluation_thread_override_does_not_change_the_trained_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory, evaluation_threads=2)
            summary = fixture.select(confirmation=False)
            self.assertEqual(summary["status"], "selected")
            self.assertEqual(summary["selected_architecture"], "cnn_rxconv")

    def test_missing_optional_weights_still_allows_metadata_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory)
            for path in fixture.train_root.glob("*/weights.pt"):
                path.unlink()
            self.assertEqual(fixture.select(confirmation=False)["status"], "selected")

    def test_remote_checkpoint_paths_relocate_by_the_declared_source_task(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PartitionFixture(directory)
            for evaluation, path in zip(fixture.evaluations, fixture.report_paths):
                declared = "results/journal_cifar_search/partition25/" + evaluation["source_task_id"]
                evaluation["checkpoint"] = declared
                report = json.loads(path.read_text())
                report["checkpoint"] = "/remote/spartan/" + declared
                write_json(path, report)
            fixture.save_manifests()
            self.assertEqual(fixture.select(confirmation=False)["status"], "selected")

    def test_every_run_must_be_complete_and_match_declared_validation_protocol(self):
        def mutate_json(path, change):
            value = json.loads(path.read_text())
            change(value)
            write_json(path, value)

        cases = {
            "missing_report": lambda f: f.report_paths[-1].unlink(),
            "incomplete_history": lambda f: mutate_json(f.history_paths[-1], lambda x: x.pop()),
            "invalid_completion_time": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x.update(total_seconds=0.)),
            "report_test_split": lambda f: mutate_json(f.report_paths[-1], lambda x: x.update(split="test")),
            "report_full_scope": lambda f: mutate_json(f.report_paths[-1], lambda x: x.update(risk_scope="full")),
            "selected_epoch_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x.update(selected_epoch=24)),
            "selected_epoch_out_of_range": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x.update(selected_epoch=26)),
            "training_config_mismatch": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x["config"].update(lr=.009)),
            "evaluation_config_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["evaluation_config"].update(lr=.009)),
            "training_seed_mismatch": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x["config"].update(seed=7102)),
            "evaluation_seed_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["evaluation_config"].update(eval_seed=2027105)),
            "draw_count_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["heldout_gibbs_mc"].update(draws=99)),
            "replicate_count_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["heldout_gibbs_mc"]["replicate_statistics"].pop()),
            "examples_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["heldout_gibbs_mc"].update(examples=10000)),
            "posterior_rule_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["heldout_gibbs_mc"].update(posterior_rule="per minibatch")),
            "loss_definition_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x["heldout_gibbs_mc"].update(bounded_loss_definition="raw CE")),
            "checkpoint_mismatch": lambda f: mutate_json(f.report_paths[-1], lambda x: x.update(checkpoint=str(f.train_root / "other_task"))),
            "validation_split_hash_mismatch": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x["split"].update(validation_indices_hash="different")),
            "bank_hash_mismatch": lambda f: mutate_json(f.metadata_paths[-1], lambda x: x.update(training_channel_bank_hash="different")),
        }
        for name, change in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                fixture = PartitionFixture(directory)
                change(fixture)
                with self.assertRaises((ValueError, OSError, RuntimeError)):
                    fixture.select()
                self.assertFalse(fixture.output.exists(), "Invalid inputs must not publish a summary")
                self.assertFalse(fixture.confirm_manifest.exists(), "Invalid inputs must not publish confirmation jobs")

    def test_incomplete_or_nonvalidation_evaluation_manifest_is_rejected(self):
        for name in ("missing_entry", "test_split", "full_scope", "wrong_seed", "wrong_draws"):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                fixture = PartitionFixture(directory)
                if name == "missing_entry":
                    fixture.evaluations.pop()
                elif name == "test_split":
                    fixture.evaluations[-1]["split"] = "test"
                elif name == "full_scope":
                    fixture.evaluations[-1]["risk_scope"] = "full"
                elif name == "wrong_seed":
                    fixture.evaluations[-1]["config"]["eval_seed"] += 1
                else:
                    fixture.evaluations[-1]["config"]["eval_draws"] = 99
                fixture.save_manifests()
                with self.assertRaises((ValueError, OSError, RuntimeError)):
                    fixture.select()
                self.assertFalse(fixture.output.exists())
                self.assertFalse(fixture.confirm_manifest.exists())


if __name__ == "__main__":
    unittest.main()
