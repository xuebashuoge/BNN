"""The balanced CIFAR refinement uses only paired validation records before confirmation."""
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

from journal.diagonal_fading_comparison import (
    CONFIRM_SEEDS, METHODS, SCREEN_SEEDS, balanced_cifar_entries, cifar_screen_bank_size, confirmation_entries,
    prepare_confirmation, screening_entries, validation_rows,
)
from journal.full_fading_comparison import report_file, write_manifest


class CifarRefinementTests(unittest.TestCase):
    def test_bank_size_comes_from_a_complete_single_size_screen(self):
        for m in (200, 500, 1000, 2000):
            entries = screening_entries(("cifar10",), m, .35)
            self.assertEqual(cifar_screen_bank_size(entries), m)
        mixed = entries[:-1] + screening_entries(("cifar10",), 200, .35)[-1:]
        for wrong in ([], entries[:-1], mixed):
            with self.assertRaises(ValueError):
                cifar_screen_bank_size(wrong)

    def test_shared_grid_and_freeze_across_validation_rounds(self):
        for include_upper in (False, True):
            with self.subTest(upper=include_upper):
                self.check_confirmation(include_upper)

    def check_confirmation(self, include_upper):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = list(METHODS)
            stages = ["screen", "refine", "cifar_balanced"]
            if include_upper:
                stages.append("cifar_upper")
            targets = {label: stages[i % len(stages)] for i, label in enumerate(labels)}
            for stage in stages:
                if stage in {"cifar_balanced", "cifar_upper"}:
                    original = screening_entries(("cifar10",), 2000, .35)
                    entries = balanced_cifar_entries(original, validation_rows(root / "screen", original), stage)
                    self.assertEqual(len(entries), 56)
                    for label in METHODS:
                        choices = [item for item in entries if item["method_label"] == label]
                        rates = (.0045, .006) if stage == "cifar_upper" else (.0015, .002)
                        expected = {(rate, scale, seed) for rate in rates
                                    for scale in (.001, .003) for seed in SCREEN_SEEDS}
                        self.assertEqual({(item["config"]["lr"], item["config"]["posterior_std"],
                                           item["config"]["seed"]) for item in choices}, expected)
                        for item in choices:
                            source = next(x for x in original if x["method_label"] == label
                                          and x["variant_id"] == item["penalty_from_validation_variant"])
                            for key in ("l2", "eta_channel", "eta_data", "m", "train_size", "epochs"):
                                self.assertEqual(item["config"][key], source["config"][key])
                else:
                    entries = screening_entries(("cifar10",), 2000, .35, stage)
                write_manifest(root / (stage + ".jsonl"), entries)
                for i, item in enumerate(entries):
                    row = dict(item, status="complete", seed=item["config"]["seed"], seconds=1,
                               validation_accuracy=(.7 if targets[item["method_label"]] == stage else .5)
                               + .001 * (i % 8 // 2), validation_cross_entropy=1.)
                    path = report_file(root / stage, item)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(row))

            output = root / "confirm-balanced.jsonl"
            args = Namespace(screen_manifest=root / "screen.jsonl", screen_root=root / "screen",
                             refinement_manifest=root / "refine.jsonl", refinement_root=root / "refine",
                             balanced_manifest=root / "cifar_balanced.jsonl", balanced_root=root / "cifar_balanced",
                             upper_manifest=root / "cifar_upper.jsonl" if include_upper else None,
                             upper_root=root / "cifar_upper" if include_upper else None,
                             eval_draws=20, output=output)
            self.assertEqual(prepare_confirmation(args)["tasks"], 35 if include_upper else 21)
            selection = json.loads(output.with_suffix(".selection.json").read_text())
            self.assertEqual(len(selection["validation_trials"]), 28 * len(stages))
            for choice in selection["selected_variants"]:
                self.assertTrue(all(name.startswith("diagonal_fading_accuracy_" + targets[choice["method_label"]] + "_")
                                    for name in choice["task_ids"]))
            confirmed = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(confirmed, confirmation_entries(selection))
            confirmed_stage = "cifar_upper_confirm" if include_upper else "cifar_balanced_confirm"
            expected_seeds = set(CONFIRM_SEEDS[confirmed_stage])
            for label in METHODS:
                self.assertEqual({x["config"]["seed"] for x in confirmed if x["method_label"] == label},
                                 expected_seeds)
            previous = set(SCREEN_SEEDS)
            for name, seeds in CONFIRM_SEEDS.items():
                if name != confirmed_stage:
                    previous.update(seeds)
            self.assertTrue(expected_seeds.isdisjoint(previous))

            last = report_file(root / stages[-1], entries[-1])
            invalid = json.loads(last.read_text())
            invalid["test_accuracy"] = .9
            last.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "validation-only"):
                prepare_confirmation(args)


if __name__ == "__main__":
    unittest.main()
