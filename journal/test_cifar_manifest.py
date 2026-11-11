"""Confirmation coverage checks with synthetic metadata; no training or data."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from journal.build_cifar_campaign import (METHODS, confirm_entries,
                                          screen_entries, validate_confirmation_summary)


def complete_summary():
    entries = screen_entries([4101, 4102], "severe", "cnn", 128)
    selected = {}
    for entry in entries:
        label = entry["method_label"]
        variant = selected.setdefault(label, dict(
            rank=1, config=entry["config"], variant_id=entry["variant_id"],
            method_label=label, group=entry["group"], complete_seed_set=True,
            seeds=[], task_ids=[]))
        if entry["variant_id"] == variant["variant_id"]:
            variant["seeds"].append(entry["config"]["seed"])
            variant["task_ids"].append(entry["task_id"])
    return dict(training_trials=len(entries), completed_trials=len(entries),
                failed_trials=[], failed_evaluations=[],
                selected_variants=list(selected.values()))


class CifarConfirmationTests(unittest.TestCase):
    def test_complete_seven_method_screen_builds_21_distinct_cases(self):
        summary = complete_summary()
        before = copy.deepcopy(summary)
        self.assertEqual(len(validate_confirmation_summary(summary)), 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary))
            entries = confirm_entries(path, [5101, 5102, 5103])
        self.assertEqual(summary, before)
        self.assertEqual(len(entries), 21)
        self.assertEqual(len({item["task_id"] for item in entries}), 21)
        self.assertEqual({item["method_label"] for item in entries}, {label for _, label in METHODS})
        self.assertTrue(all(item["config"]["train_size"] == 45000 and
                            item["config"]["val_size"] == 5000 and
                            item["config"]["epochs"] == 100 for item in entries))

    def test_incomplete_or_failed_screen_is_rejected(self):
        for mutation in ("count", "train_failure", "eval_failure", "pre_evaluation"):
            summary = complete_summary()
            if mutation == "count":
                summary["completed_trials"] -= 1
            elif mutation == "train_failure":
                summary["failed_trials"] = [{"status": "failed"}]
            elif mutation == "eval_failure":
                summary["failed_evaluations"] = [{"test_status": "failed"}]
            else:
                del summary["failed_evaluations"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_summary(summary)

    def test_missing_duplicate_or_inconsistent_method_is_rejected(self):
        for mutation in ("missing", "duplicate", "wrong_algorithm", "wrong_law"):
            summary = complete_summary()
            if mutation == "missing":
                summary["selected_variants"].pop()
            elif mutation == "duplicate":
                summary["selected_variants"].append(copy.deepcopy(summary["selected_variants"][0]))
            elif mutation == "wrong_algorithm":
                summary["selected_variants"][0]["config"]["method"] = "joint"
            else:
                summary["selected_variants"][0]["config"]["train_channel"] = {"kind": "perfect"}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_summary(summary)

    def test_missing_unpaired_or_partial_seed_set_is_rejected(self):
        for mutation in ("missing_seed", "unpaired", "partial", "duplicate_task"):
            summary = complete_summary()
            variant = summary["selected_variants"][0]
            if mutation == "missing_seed":
                variant["seeds"] = [4101]
                variant["task_ids"] = variant["task_ids"][:1]
            elif mutation == "unpaired":
                variant["seeds"] = [4101, 9999]
            elif mutation == "partial":
                variant["complete_seed_set"] = False
            else:
                variant["task_ids"][1] = variant["task_ids"][0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_summary(summary)


if __name__ == "__main__":
    unittest.main()
