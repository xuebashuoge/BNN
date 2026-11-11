"""CPU numerical-policy checks with mocks; no training or dataset access."""

import ast
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from journal import run_experiment as experiment


class CpuNumericsTests(unittest.TestCase):
    def test_default_and_strict_boolean(self):
        self.assertIs(experiment.normalize_config({})["cpu_flush_denormal"], False)
        self.assertIs(experiment.normalize_config({"cpu_flush_denormal": True})["cpu_flush_denormal"], True)
        for invalid in (0, 1, "false", "true", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                experiment.normalize_config({"cpu_flush_denormal": invalid})

    def test_cpu_applies_requested_policy_and_records_support(self):
        for requested in (False, True):
            config = experiment.normalize_config({"cpu_flush_denormal": requested})
            with self.subTest(requested=requested), \
                    patch.object(experiment, "_CPU_FLUSH_DENORMAL_SUPPORTED", None), \
                    patch.object(experiment.torch, "set_flush_denormal", return_value=True) as setter, \
                    patch.object(experiment.platform, "platform", return_value="synthetic-host"), \
                    patch.object(experiment.subprocess, "check_output", side_effect=["synthetic-sha\n", ""]):
                experiment.check_device(config)
                setter.assert_called_once_with(requested)
                policy = experiment.provenance(config, {})["cpu_numerics"]
                self.assertEqual(policy, {"flush_denormal_requested": requested,
                                          "flush_denormal_supported": True})

    def test_unsupported_opt_in_fails_but_legacy_default_remains_usable(self):
        with patch.object(experiment, "_CPU_FLUSH_DENORMAL_SUPPORTED", None), \
                patch.object(experiment.torch, "set_flush_denormal", return_value=False):
            experiment.check_device(experiment.normalize_config({}))
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                experiment.check_device(experiment.normalize_config({"cpu_flush_denormal": True}))

    def test_cuda_does_not_change_cpu_control_state(self):
        config = experiment.normalize_config({"device": "cuda", "cpu_flush_denormal": True})
        with patch.object(experiment.torch, "set_flush_denormal") as setter, \
                patch.object(experiment.torch.cuda, "is_available", return_value=True), \
                patch.object(experiment.torch.backends, "cudnn", SimpleNamespace(
                    benchmark=False, deterministic=True, allow_tf32=True)), \
                patch.object(experiment.platform, "platform", return_value="synthetic-host"), \
                patch.object(experiment.subprocess, "check_output", side_effect=["synthetic-sha\n", ""]):
            experiment.check_device(config)
            setter.assert_not_called()
            self.assertIsNone(experiment.provenance(config, {})["cpu_numerics"])

    def test_policy_precedes_intraop_worker_creation_in_both_entry_points(self):
        # Parse the entry points rather than invoking a train/evaluate path.
        for function in (experiment.main, experiment.evaluate_checkpoint):
            tree = ast.parse(inspect.getsource(function))
            calls = [(node.lineno, node.func.id if isinstance(node.func, ast.Name)
                      else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                     for node in ast.walk(tree) if isinstance(node, ast.Call)]
            policy = min(line for line, name in calls if name == "check_device")
            workers = min(line for line, name in calls if name == "set_num_threads")
            with self.subTest(entry_point=function.__name__):
                self.assertLess(policy, workers)


if __name__ == "__main__":
    unittest.main()
