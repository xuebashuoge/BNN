"""Guard coverage, validation-only selection and controlled sweep semantics."""
import copy
from collections import Counter, defaultdict
import unittest

from journal.build_bayesian_campaign import (
    CHANNELS, DATASETS, METHODS, M_GRID, P_GRID, SEEDS, screen_entries,
    select_validation, selected_entries,
)
from journal.campaign import validate_entries


def fake_results(entries):
    """Only validation scores are provided; selection cannot require test data."""
    return [dict(task_id=item["task_id"], config=item["config"], status="complete",
                 group=item["group"], method_label=item["method_label"],
                 family=item["family"], variant_id=item["variant_id"],
                 seed=item["config"]["seed"], validation_accuracy=.7 + .01 * (index % 4),
                 validation_cross_entropy=.5) for index, item in enumerate(entries)]


class BayesianCampaignTests(unittest.TestCase):
    def test_full_matrix_has_equal_budgets_and_gaussian_predictors(self):
        entries = screen_entries()
        self.assertEqual(len(entries), 4 * 2 * 2 * 4 * 8 * 2)
        validate_entries(entries)
        budgets = Counter((item["group"], item["method_label"]) for item in entries)
        self.assertEqual(len(budgets), 64)
        self.assertEqual(set(budgets.values()), {16})
        self.assertTrue(all(item["config"]["predictor"] == "gaussian" for item in entries))
        self.assertEqual({item["config"]["dataset"] for item in entries}, set(DATASETS))
        for item in entries:
            config = item["config"]
            self.assertEqual(config["deployment_channel"], CHANNELS[item["channel_name"]])
            self.assertEqual(config["train_channel"], {"kind": "perfect"} if
                             item["training_regime"] == "perfect" else config["deployment_channel"])
            self.assertEqual(item["family"], "proposed" if config["method"] in {"proposed", "joint"} else "baseline")

    def test_complete_paired_validation_selects_one_per_label_without_test(self):
        entries = screen_entries(datasets=("moons",), channels=("outage",))
        rows = fake_results(entries)
        selected = select_validation(entries, rows)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(item["seeds"] == list(SEEDS["screen"]) for item in selected))
        for seed in SEEDS["screen"]:
            with self.assertRaisesRegex(ValueError, "disjoint"):
                selected_entries(selected, (seed,), "confirm")
        confirmation = selected_entries(selected, SEEDS["confirm"], "confirm")
        self.assertEqual(len(confirmation), 8 * 5)
        self.assertEqual({item["config"]["seed"] for item in confirmation}, set(SEEDS["confirm"]))
        # Changing even extreme test scores cannot affect the selected variant.
        polluted = [{**row, "test_accuracy": 1. if index % 2 else 0.}
                    for index, row in enumerate(rows)]
        self.assertEqual([item["variant_id"] for item in selected],
                         [item["variant_id"] for item in select_validation(entries, polluted)])

    def test_incomplete_foreign_failed_or_unequal_screens_are_rejected(self):
        entries = screen_entries(datasets=("moons",), channels=("outage",))
        rows = fake_results(entries)
        with self.assertRaisesRegex(ValueError, "every screening"):
            select_validation(entries, rows[:-1])
        failed = copy.deepcopy(rows)
        failed[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_validation(entries, failed)
        foreign = copy.deepcopy(rows)
        foreign[0]["config"]["posterior_std"] = .2
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            select_validation(entries, foreign)
        # Whole variant deletion still cannot shrink one method's tuning budget.
        removed_variant = entries[0]["variant_id"]
        short = [item for item in entries if item["variant_id"] != removed_variant]
        with self.assertRaisesRegex(ValueError, "Unequal"):
            select_validation(short, fake_results(short))

    def test_launcher_execution_resources_can_differ_without_changing_selection(self):
        entries = screen_entries(datasets=("moons",), channels=("outage",))
        rows = fake_results(entries)
        expected = select_validation(entries, rows)
        overridden = copy.deepcopy(rows)
        for row in overridden:
            row["config"].update(threads=1, device="cpu")
        selected = select_validation(entries, overridden)
        self.assertEqual([item["variant_id"] for item in selected],
                         [item["variant_id"] for item in expected])

    def test_dense_sweeps_freeze_hyperparameters_and_training_budget(self):
        entries = screen_entries(datasets=("mnist",), channels=("outage",))
        selected = select_validation(entries, fake_results(entries))
        m_entries = selected_entries(selected, SEEDS["m_sweep"], "m_sweep")
        self.assertEqual(len(m_entries), 4 * len(M_GRID) * 3)
        self.assertGreater(sum(value < 100 for value in M_GRID), 7)
        validate_entries(m_entries)
        curves = defaultdict(list)
        for item in m_entries:
            curves[item["method_label"]].append(item)
            self.assertEqual(item["config"]["train_size"], 10000)
            self.assertEqual(item["config"]["epochs"], 30)
            self.assertEqual(item["sweep_value"], item["config"]["m"])
        for curve in curves.values():
            configs = [{key: value for key, value in item["config"].items()
                        if key not in {"seed", "channel_seed", "eval_seed", "m"}} for item in curve]
            self.assertTrue(all(config == configs[0] for config in configs))
        p_entries = selected_entries(selected, SEEDS["p_train_sweep"], "p_train_sweep")
        self.assertEqual(len(p_entries), 4 * len(P_GRID) * 3)
        validate_entries(p_entries)
        self.assertEqual({item["sweep_value"] for item in p_entries}, set(P_GRID))
        for item in p_entries:
            self.assertEqual(item["config"]["deployment_channel"], CHANNELS["outage"])
            if item["sweep_value"] == 0:
                self.assertEqual(item["config"]["train_channel"], {"kind": "perfect"})
            if item["sweep_value"] == .5:
                self.assertEqual(item["training_regime"], "matched")

    def test_pilot_covers_full_shapes_but_is_not_a_scientific_result(self):
        entries = screen_entries(SEEDS["pilot"], pilot=True)
        self.assertEqual(len(entries), 64)
        self.assertTrue(all(item["config"]["epochs"] == 1 for item in entries))
        self.assertTrue(all(item["scientific_result"] is False for item in entries))
        self.assertEqual({item["config"]["architecture"] for item in entries
                          if item["config"]["dataset"] == "cifar10"}, {"cnn_spatial3"})


if __name__ == "__main__":
    unittest.main()
