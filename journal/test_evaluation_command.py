"""Evaluation scope is an explicit launch choice; no datasets or execution."""
import unittest
from journal.run_task import experiment_command


class EvaluationCommandTests(unittest.TestCase):
    def test_validation_deployment_scope_is_forwarded(self):
        command = experiment_command(dict(mode="evaluate", checkpoint="frozen", split="validation",
                                          risk_scope="deployment"), "config.json", "scores")
        self.assertEqual(command[command.index("--mode") + 1], "evaluate")
        self.assertEqual(command[command.index("--split") + 1], "validation")
        self.assertEqual(command[command.index("--risk-scope") + 1], "deployment")

    def test_old_evaluation_default_remains_full(self):
        command = experiment_command(dict(mode="evaluate", checkpoint="frozen"), "config.json", "scores")
        self.assertEqual(command[command.index("--risk-scope") + 1], "full")

    def test_training_does_not_add_evaluation_or_checkpoint_flags(self):
        command = experiment_command(dict(mode="train"), "config.json", "weights")
        self.assertNotIn("--risk-scope", command)
        self.assertNotIn("--checkpoint", command)

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment_command(dict(mode="evaluate", checkpoint="frozen", risk_scope="typo"), "c", "o")


if __name__ == "__main__":
    unittest.main()
