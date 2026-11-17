"""Checkpoint staging preserves bytes and rejects partial or changed training."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from journal.build_campaign import make_entry
from journal.run_experiment import digest, make_models, normalize_config
from journal.stage_bayesian_checkpoints import (CHECKPOINT_FILES, TRAINING_FILES, fingerprint,
                                               stage_checkpoints, verify_training_sources)


class StageBayesianCheckpointsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.git("init", "--quiet")
        for relative in TRAINING_FILES:
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixed training source\n")
        self.git("add", "journal")
        self.git("-c", "user.name=Staging Test", "-c", "user.email=test@example.invalid",
                 "commit", "--quiet", "-m", "training source")
        self.source_revision = self.git("rev-parse", "HEAD")
        self.source = self.root / "old"
        (self.source / "pilot").mkdir(parents=True)
        self.output = self.root / "new"
        self.manifest = self.root / "pilot.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *arguments):
        return subprocess.check_output(["git", *arguments], cwd=self.repo, text=True,
                                       stderr=subprocess.PIPE).strip()

    def checkpoint(self, seed=1, **overrides):
        config = normalize_config({"dataset": "moons", "method": "erm", "predictor": "gaussian",
                                   "width": 4, "train_size": 8, "val_size": 4,
                                   "epochs": 2, "m": 3, **overrides})
        entry = make_entry(config, seed, "moons_analog", "ERM-Matched", "bayes_pilot")
        config = normalize_config(entry["config"])
        path = self.source / "pilot" / entry["task_id"]
        path.mkdir()
        model, bank = make_models(config, 2, 2)
        metadata = dict(config=config, git_sha=self.source_revision, git_dirty=False,
                        total_seconds=2., peak_rss_mb=3., selected_epoch=1,
                        training_channel_bank_hash=digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()}))
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "task_config.json").write_text(json.dumps(config))
        (path / "manifest_entry.json").write_text(json.dumps(entry))
        (path / "history.json").write_text(json.dumps([
            dict(epoch=epoch, validation=dict(accuracy=.6 + epoch / 100, cross_entropy=.7))
            for epoch in (1, 2)]))
        (path / "train.log").write_bytes(b"original training log\n")
        torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), path / "weights.pt")
        return entry, path

    def stage(self, entries):
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        return stage_checkpoints(self.source, self.output, self.source_revision,
                                 self.manifest, "pilot", self.repo)

    def test_complete_checkpoint_bytes_and_provenance_are_preserved(self):
        entry, source = self.checkpoint()
        # Evaluation output is deliberately excluded from the staging allowlist.
        (source / "test_full").mkdir()
        (source / "test_full" / "evaluation.json").write_text('{"test": "never reuse"}')
        source_before = {str(path.relative_to(source)): path.read_bytes()
                         for path in source.rglob("*") if path.is_file()}
        result = self.stage([entry])
        target = self.output / "pilot" / entry["task_id"]
        self.assertEqual(result["counts"], {"staged_complete": 1})
        self.assertEqual({item.name for item in target.iterdir()}, {*CHECKPOINT_FILES, "staging_provenance.json"})
        for filename in CHECKPOINT_FILES:
            self.assertEqual((target / filename).read_bytes(), source_before[filename])
        self.assertEqual(source_before, {str(path.relative_to(source)): path.read_bytes()
                                         for path in source.rglob("*") if path.is_file()})
        provenance = json.loads((target / "staging_provenance.json").read_text())
        self.assertEqual(provenance["source_revision"], self.source_revision)
        self.assertEqual(provenance["source_fingerprints"]["weights.pt"], fingerprint(source / "weights.pt"))
        self.assertEqual(self.stage([entry])["counts"], {"already_staged_identical": 1})

    def test_partial_corrupt_and_wrong_revision_checkpoints_are_not_copied(self):
        complete, _ = self.checkpoint(seed=1)
        partial, partial_path = self.checkpoint(seed=2)
        history = json.loads((partial_path / "history.json").read_text())[:1]
        (partial_path / "history.json").write_text(json.dumps(history))
        corrupt, corrupt_path = self.checkpoint(seed=3)
        (corrupt_path / "weights.pt").write_bytes(b"broken pickle")
        foreign, foreign_path = self.checkpoint(seed=4)
        metadata = json.loads((foreign_path / "metadata.json").read_text())
        metadata["git_sha"] = "a" * 40
        (foreign_path / "metadata.json").write_text(json.dumps(metadata))
        original = {str(path.relative_to(self.source)): path.read_bytes()
                    for path in self.source.rglob("*") if path.is_file()}
        result = self.stage([complete, partial, corrupt, foreign])
        self.assertEqual(result["counts"].get("staged_complete"), 1)
        self.assertEqual(len(result["records"]), 4)
        for entry in (partial, corrupt, foreign):
            self.assertFalse((self.output / "pilot" / entry["task_id"]).exists())
        self.assertEqual(original, {str(path.relative_to(self.source)): path.read_bytes()
                                    for path in self.source.rglob("*") if path.is_file()})

    def test_existing_mismatched_target_is_never_overwritten(self):
        entry, _ = self.checkpoint()
        self.stage([entry])
        target = self.output / "pilot" / entry["task_id"] / "weights.pt"
        target.write_bytes(b"existing target must survive")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.stage([entry])
        self.assertEqual(target.read_bytes(), b"existing target must survive")
        audits = [json.loads(path.read_text()) for path in (self.output / "staging_audits").glob("*.json")]
        self.assertTrue(any(audit["status"] == "failed" for audit in audits))

    def test_training_code_change_is_rejected_in_head_and_working_tree(self):
        verify_training_sources(self.source_revision, self.repo)
        training_file = self.repo / TRAINING_FILES[0]
        training_file.write_text("# changed training source\n")
        with self.assertRaisesRegex(ValueError, "Training source changed"):
            verify_training_sources(self.source_revision, self.repo)
        self.git("add", TRAINING_FILES[0])
        self.git("-c", "user.name=Staging Test", "-c", "user.email=test@example.invalid",
                 "commit", "--quiet", "-m", "changed training source")
        with self.assertRaisesRegex(ValueError, "Training source changed"):
            verify_training_sources(self.source_revision, self.repo)

    def test_manifest_config_and_saved_bank_corruption_are_rejected(self):
        entry, path = self.checkpoint()
        config = json.loads((path / "task_config.json").read_text())
        config["lr"] = .5
        (path / "task_config.json").write_text(json.dumps(config))
        result = self.stage([entry])
        self.assertEqual(result["counts"], {"source_manifest_identity_or_config_mismatch": 1})
        (path / "task_config.json").write_text(json.dumps(entry["config"]))
        state = torch.load(path / "weights.pt", weights_only=True)
        state["bank"]["gain"][0, 0] += 1
        torch.save(state, path / "weights.pt")
        result = self.stage([entry])
        self.assertEqual(result["counts"], {"source_channel_bank_mismatch": 1})


if __name__ == "__main__":
    unittest.main()
