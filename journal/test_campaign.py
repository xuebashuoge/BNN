"""Checks that search summaries cannot manufacture a complete accuracy win."""

import unittest

from journal.campaign import select_variants, test_comparisons


def row(method, family, seed, accuracy, variant=None):
    variant = variant or method
    return dict(task_id=f"{method}_{variant}_{seed}", group="condition", method_label=method,
                family=family, variant_id=variant, seed=seed, test_accuracy=accuracy,
                test_cross_entropy=1-accuracy, test_status="complete", config={})


class CampaignSummaryTests(unittest.TestCase):
    def setUp(self):
        self.labels = {"condition": {"ERM", "PAC", "T1"}}
        self.seeds = {"condition": {label: {1, 2} for label in self.labels["condition"]}}

    def test_missing_method_or_seed_blocks_complete_win(self):
        rows = [row("ERM", "baseline", seed, .8) for seed in (1, 2)]
        rows += [row("T1", "proposed", seed, .9) for seed in (1, 2)]
        result = test_comparisons(rows, self.labels, self.seeds)[0]
        self.assertEqual(result["missing_methods"], ["PAC"])
        self.assertFalse(result["goal_met_with_complete_method_coverage"])
        rows.append(row("PAC", "baseline", 1, .81))
        result = test_comparisons(rows, self.labels, self.seeds)[0]
        self.assertEqual(result["missing_seeds_by_method"], {"PAC": [2]})
        self.assertFalse(result["goal_met_for_all_evaluated_seeds"])

    def test_tie_and_one_bad_seed_are_distinguished(self):
        rows = [row(label, "baseline", seed, .8) for label in ("ERM", "PAC") for seed in (1, 2)]
        rows += [row("T1", "proposed", 1, .79), row("T1", "proposed", 2, .85)]
        result = test_comparisons(rows, self.labels, self.seeds)[0]
        self.assertTrue(result["goal_met_with_complete_method_coverage"])
        self.assertFalse(result["goal_met_for_all_evaluated_seeds"])
        rows[-1]["test_accuracy"] = .81
        result = test_comparisons(rows, self.labels, self.seeds)[0]
        self.assertTrue(result["proposed_ties_highest_mean"])
        self.assertFalse(result["proposed_strictly_highest_mean"])

    def test_variant_selection_uses_all_seeds_and_requires_completion(self):
        entries, results = [], []
        for variant, accuracies in (("steady", (.85, .85)), ("lucky", (.99, .6)), ("partial", (.99, .99))):
            for seed, accuracy in enumerate(accuracies, 1):
                item = row("T1", "proposed", seed, accuracy, variant)
                item["config"] = {"seed": seed}
                entries.append(item)
                if variant == "partial" and seed == 2:
                    continue
                results.append({**item, "status": "complete", "validation_accuracy": accuracy,
                                "validation_cross_entropy": 1-accuracy})
        _, selected = select_variants(results, entries, 1)
        self.assertEqual(selected[0]["variant_id"], "steady")
        self.assertEqual(selected[0]["seeds"], [1, 2])


if __name__ == "__main__":
    unittest.main()
