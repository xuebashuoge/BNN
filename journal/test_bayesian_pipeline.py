"""Resume invariants for immutable manifests and validation-only selection."""
import json
from pathlib import Path
import tempfile
import unittest

from journal.run_bayesian_campaign import freeze_validation_results, install_immutable_manifest


class BayesianPipelineTests(unittest.TestCase):
    def test_manifest_and_plan_are_immutable_as_a_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, target = root / "candidate.jsonl", root / "manifest.jsonl"

            def build(manifest=b"declared trials\n", plan=b'{"tasks": 1}\n'):
                candidate.write_bytes(manifest)
                candidate.with_suffix(".plan.json").write_bytes(plan)

            build()
            install_immutable_manifest(candidate, target)
            self.assertEqual(target.read_bytes(), b"declared trials\n")
            build()
            install_immutable_manifest(candidate, target)
            self.assertFalse(candidate.exists())
            for manifest, plan in ((b"different trials\n", b'{"tasks": 1}\n'),
                                   (b"declared trials\n", b'{"tasks": 2}\n')):
                build(manifest, plan)
                with self.assertRaisesRegex(ValueError, "Immutable campaign"):
                    install_immutable_manifest(candidate, target)
                self.assertEqual(target.read_bytes(), b"declared trials\n")
                self.assertEqual(target.with_suffix(".plan.json").read_bytes(), b'{"tasks": 1}\n')

    def test_frozen_validation_records_tolerate_resume_order_but_reject_score_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "results.json", root / "selection.json"
            rows = [{"task_id": "b", "validation_accuracy": .7, "resumed": False},
                    {"task_id": "a", "validation_accuracy": .8, "resumed": False}]
            source.write_text(json.dumps(rows))
            freeze_validation_results(source, destination)
            frozen = destination.read_bytes()
            rows = [{**row, "resumed": True} for row in reversed(rows)]
            source.write_text(json.dumps(rows))
            freeze_validation_results(source, destination)
            self.assertEqual(destination.read_bytes(), frozen)
            rows[0]["validation_accuracy"] = 1.
            source.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "selection results changed"):
                freeze_validation_results(source, destination)
            self.assertEqual(destination.read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
