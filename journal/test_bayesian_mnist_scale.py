"""Guard the fair MNIST scale-control extension and validation-only selection."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from journal.build_bayesian_campaign import SEEDS, screen_entries
from journal.build_bayesian_mnist_scale import (
    build_screen, build_selected, combine_and_select, extra_entries, freeze_original,
    manifest_text, read_training_records,
)
from journal.campaign import validate_entries


def validation_rows(entries, score=.8):
    return [dict(task_id=item["task_id"], config=copy.deepcopy(item["config"]), status="complete",
                 group=item["group"], method_label=item["method_label"], family=item["family"],
                 variant_id=item["variant_id"], seed=item["config"]["seed"],
                 validation_accuracy=score, validation_cross_entropy=.4, selected_epoch=30,
                 test_accuracy=1., resumed=False) for item in entries]


class MnistScaleControlTests(unittest.TestCase):
    def setUp(self):
        self.original = screen_entries(datasets=("mnist",))
        self.extra = extra_entries(self.original)

    def test_append_changes_only_rho_lr_and_retains_positive_gaussian_scale(self):
        before = copy.deepcopy(self.original)
        self.assertEqual(len(self.original), 512)
        self.assertEqual(len(self.extra), 512)
        self.assertFalse({item["task_id"] for item in self.original} & {item["task_id"] for item in self.extra})
        validate_entries(self.extra)
        counts = Counter((item["group"], item["method_label"]) for item in self.extra)
        self.assertEqual(set(counts.values()), {32})
        for source, fixed in zip(self.original, self.extra):
            old, new = source["config"], fixed["config"]
            self.assertEqual({key for key in old if old[key] != new[key]}, {"rho_lr_scale"})
            self.assertEqual(old["rho_lr_scale"], .1)
            self.assertEqual(new["rho_lr_scale"], 0.)
            self.assertEqual(new["predictor"], "gaussian")
            self.assertIn(new["posterior_std"], (.001, .003))
            self.assertEqual(fixed["appended_from_task"], source["task_id"])
            self.assertEqual(fixed["stage"], "bayes_screen")
        self.assertEqual(self.original, before)

    def test_complete_equal_budget_keeps_original_when_it_wins_on_validation(self):
        combined, rows, selected = combine_and_select(
            self.original, validation_rows(self.original, .9), self.extra, validation_rows(self.extra, .8))
        self.assertEqual(combined[:512], self.original)
        self.assertEqual(len(rows), 1024)
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(item["config"]["rho_lr_scale"] == .1 for item in selected))
        self.assertTrue(all(item["seeds"] == list(SEEDS["screen"]) for item in selected))
        self.assertTrue(all("test_accuracy" not in row and "resumed" not in row for row in rows))
        # Even inverted test-only sentinels cannot affect validation selection.
        original = validation_rows(self.original, .8)
        extra = validation_rows(self.extra, .9)
        for row in original:
            row["test_accuracy"] = 1.
        for row in extra:
            row["test_accuracy"] = 0.
        _, _, selected = combine_and_select(self.original, original, self.extra, extra)
        self.assertTrue(all(item["config"]["rho_lr_scale"] == 0. for item in selected))

    def test_partial_failed_foreign_or_changed_config_cannot_enter_selection(self):
        original, extra = validation_rows(self.original), validation_rows(self.extra)
        with self.assertRaisesRegex(ValueError, "every declared"):
            combine_and_select(self.original, original, self.extra, extra[:-1])
        failed = copy.deepcopy(extra)
        failed[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            combine_and_select(self.original, original, self.extra, failed)
        wrong = copy.deepcopy(extra)
        wrong[0]["config"]["posterior_std"] = .2
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            combine_and_select(self.original, original, self.extra, wrong)
        changed = copy.deepcopy(self.extra)
        changed[0]["config"]["rho_lr_scale"] = .1
        with self.assertRaisesRegex(ValueError, "fixed-scale configuration"):
            combine_and_select(self.original, original, changed, extra)

    def test_nonfinite_validation_and_favorable_seed_subsets_are_rejected(self):
        original, extra = validation_rows(self.original), validation_rows(self.extra)
        extra[0]["validation_accuracy"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid validation"):
            combine_and_select(self.original, original, self.extra, extra)
        only_favorable = [row for row in validation_rows(self.extra) if row["seed"] == 72101]
        with self.assertRaisesRegex(ValueError, "every declared"):
            combine_and_select(self.original, original, self.extra, only_favorable)

    def test_snapshot_and_final_manifests_are_separate_immutable_and_complete(self):
        with tempfile.TemporaryDirectory(prefix="mnist-scale-control-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            original_manifest = source / "screen.jsonl"
            original_results = source / "training_results.json"
            original_manifest.write_text(manifest_text(self.original))
            original_results.write_text(json.dumps(validation_rows(self.original, .8)))
            source_bytes = original_manifest.read_bytes(), original_results.read_bytes()
            snapshot = root / "frozen_original"
            freeze_original(original_manifest, original_results, snapshot)
            fixed_manifest = root / "new" / "screen.jsonl"
            build_screen(snapshot / "original_manifest.jsonl", fixed_manifest)
            extra_results = root / "new" / "training_results.json"
            extra_results.write_text(json.dumps(validation_rows(self.extra, .9)))
            output = root / "selected"
            protocol = build_selected(snapshot / "original_manifest.jsonl", snapshot / "original_training_results.json",
                                      fixed_manifest, extra_results, output)
            self.assertEqual(protocol["combined_trials"], 1024)
            self.assertEqual(protocol["candidates_per_method_regime"], 32)
            for stage, count in (("confirm", 80), ("m_sweep", 264), ("p_train_sweep", 168)):
                entries = [json.loads(line) for line in (output / f"{stage}.jsonl").read_text().splitlines()]
                self.assertEqual(len(entries), count)
                self.assertEqual({item["config"]["seed"] for item in entries}, set(SEEDS[stage]))
                self.assertTrue(all(item["config"]["dataset"] == "mnist" for item in entries))
                self.assertTrue(all(item["config"]["rho_lr_scale"] == 0. for item in entries))
            self.assertEqual(source_bytes, (original_manifest.read_bytes(), original_results.read_bytes()))
            self.assertEqual(protocol, build_selected(snapshot / "original_manifest.jsonl", snapshot / "original_training_results.json",
                                      fixed_manifest, extra_results, output))
            changed = validation_rows(self.extra, .95)
            extra_results.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Immutable"):
                build_selected(snapshot / "original_manifest.jsonl", snapshot / "original_training_results.json",
                               fixed_manifest, extra_results, output)

    def test_explicit_test_result_path_is_rejected_before_it_is_opened(self):
        # The nonexistent path proves the forbidden source is rejected before
        # filesystem reading, rather than accidentally attempting test access.
        with self.assertRaisesRegex(ValueError, "training/validation"):
            read_training_records("/does/not/exist/test_results.json")


if __name__ == "__main__":
    unittest.main()
