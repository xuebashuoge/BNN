"""Validation-only search bookkeeping checks; no datasets or training."""
import json
from pathlib import Path
import unittest

from journal.build_cifar_search import confirmation_entries, search_entries
from journal.run_experiment import normalize_config


class CifarSearchPlanTests(unittest.TestCase):
    def original(self):
        return [json.loads(line) for line in Path("journal/configs/cifar_mid_confirm.jsonl").read_text().splitlines()]

    def test_paired_search_preserves_physical_and_data_conditions(self):
        new, cached = search_entries(self.original())
        self.assertEqual((len(new), len(cached)), (24, 8))
        grouped = {}
        for entry in new + cached:
            config = normalize_config(entry["config"])
            self.assertEqual((config["train_size"], config["val_size"], config["epochs"]), (45000, 5000, 100))
            self.assertEqual((config["architecture"], config["width"], config["m"]), ("cnn_mid", 128, 100))
            self.assertEqual(config["train_channel"], config["deployment_channel"])
            self.assertEqual((config["prior_std"], config["training_loss"], config["loss_epsilon"]),
                             (1., "bounded_ce_smooth", .01))
            self.assertGreater(config["posterior_std"], 0)
            grouped.setdefault(entry["variant_id"], set()).add(config["seed"])
        self.assertTrue(all(seeds == {7101, 7102} for seeds in grouped.values()))
        self.assertEqual(len({entry["task_id"] for entry in new + cached}), 32)

    def test_confirmation_uses_validation_choices_and_fresh_seeds(self):
        _, cached = search_entries(self.original())
        choices = []
        for label in ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched"):
            entry = next(item for item in cached if item["method_label"] == label)
            choices.append(dict(rank=1, group=entry["group"], method_label=label,
                                config=entry["config"], variant_id=entry["variant_id"],
                                complete_seed_set=True, seeds=[7101, 7102]))
        summary = dict(training_trials=32, completed_trials=32, failed_trials=[],
                       selected_variants=choices, test_comparisons=[{"misleading_test_winner": "ignored"}])
        entries = confirmation_entries(summary, [8101, 8102, 8103])
        self.assertEqual(len(entries), 12)
        self.assertEqual({entry["config"]["seed"] for entry in entries}, {8101, 8102, 8103})
        for entry in entries:
            source = next(item for item in choices if item["method_label"] == entry["method_label"])
            self.assertEqual(entry["selected_from_validation_variant"], source["variant_id"])
        choices[0]["complete_seed_set"] = False
        with self.assertRaises(ValueError):
            confirmation_entries(summary, [8101, 8102, 8103])


if __name__ == "__main__":
    unittest.main()
