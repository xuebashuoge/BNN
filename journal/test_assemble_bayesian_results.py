"""Final cohort composition is validation-fixed and never test-score adaptive."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal import assemble_bayesian_results as assembly
from journal.build_bayesian_campaign import SEEDS, make_plan, screen_entries, select_validation, selected_entries
from journal.build_bayesian_mnist_scale import build_screen, build_selected, manifest_text, read_entries
from journal.run_bayesian_mnist_scale import capture_original
from journal.test_bayesian_mnist_scale import validation_rows


class AssemblyFixture:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.main, self.control = self.root / "main", self.root / "mnist32"
        (self.main / "manifests").mkdir(parents=True)
        (self.main / "screen").mkdir()
        self.control.mkdir()
        self.code = dict(main_revision="main_revision", mnist_revision="control_revision", assembly_revision="postprocess_revision",
                         scientific_blobs={path: "same_blob" for path in assembly.SOURCE_FILES},
                         mnist_control_blobs={path: "control_blob" for path in assembly.CONTROL_FILES})
        self.screen = self.main / "manifests" / "screen.jsonl"
        self.frozen = self.main / "validation_selection_results.json"
        entries = screen_entries()
        rows = validation_rows(entries, .8)
        self.screen.write_text(manifest_text(entries))
        (self.main / "screen" / "training_results.json").write_text(json.dumps(rows))
        frozen = sorted(({key: value for key, value in row.items() if key != "resumed"} for row in rows), key=lambda row: row["task_id"])
        self.frozen.write_text(json.dumps(frozen))
        state = dict(status="complete", current_stage=None, git_sha=self.code["main_revision"], confirm_draws=20, sweep_draws=10,
                     stages={"screen": dict(status="complete", tasks=2048, manifest_sha256=assembly.fingerprint(self.screen)["sha256"])})
        (self.main / "pipeline_status.json").write_text(json.dumps(state))
        original_manifest, original_results, capture = capture_original(self.main, self.control, self.code["main_revision"])
        extra_manifest = self.control / "manifests" / "screen.jsonl"
        build_screen(original_manifest, extra_manifest)
        (self.control / "screen").mkdir()
        extra_results = self.control / "screen" / "training_results.json"
        extra_results.write_text(json.dumps(validation_rows(read_entries(extra_manifest), .9)))
        protocol = build_selected(original_manifest, original_results, extra_manifest, extra_results, self.control / "selected")
        selected = select_validation(entries, frozen)
        control_state = dict(status="complete", current_stage=None, git_sha=self.code["mnist_revision"],
            main_revision=self.code["main_revision"], main_root=str(self.main), original_capture=capture, validation_selection=protocol,
            source_verification=dict(main_revision=self.code["main_revision"], execution_revision=self.code["mnist_revision"],
                control_files=self.code["mnist_control_blobs"],
                files={path: dict(main_blob=blob, control_blob=blob, working_blob=blob) for path, blob in self.code["scientific_blobs"].items()}),
            stages={})
        for stage in assembly.STAGES:
            final = selected_entries(selected, SEEDS[stage], stage)
            path = self.main / "manifests" / (stage + ".jsonl")
            path.write_text(manifest_text(final))
            source_hashes = {str(path): assembly.fingerprint(path)["sha256"] for path in (self.screen, self.frozen)}
            path.with_suffix(".plan.json").write_text(json.dumps(make_plan(final, stage, SEEDS[stage], source_hashes)))
            state["stages"][stage] = dict(status="complete", tasks=assembly.COUNTS[stage], manifest_sha256=assembly.fingerprint(path)["sha256"])
            control_path = self.control / "selected" / (stage + ".jsonl")
            control_state["stages"][stage] = dict(status="complete", tasks=assembly.MNIST_COUNTS[stage],
                manifest_sha256=assembly.fingerprint(control_path)["sha256"])
            (self.main / stage).mkdir()
            (self.control / stage).mkdir()
        (self.main / "pipeline_status.json").write_text(json.dumps(state))
        (self.control / "control_status.json").write_text(json.dumps(control_state))

    def replay(self):
        return assembly.replay_selections(self.main, self.control, assembly.SourceReader(), self.code)


class ValidationReplayTests(unittest.TestCase):
    def test_replays_both_validation_budgets_before_any_test_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            reader = assembly.SourceReader()
            manifests, exclusions, replay = assembly.replay_selections(fixture.main, fixture.control, reader, fixture.code)
            self.assertFalse(any("test_results" in path for path in reader.files))
            self.assertEqual(sum(map(len, exclusions.values())), 512)
            for stage in assembly.STAGES:
                combined = assembly.choose_cohort(manifests[stage]["main"], manifests[stage]["mnist"], stage)
                self.assertEqual(len(combined), assembly.COUNTS[stage])
                self.assertEqual(sum(cohort == "mnist32" for entry, cohort in combined), assembly.MNIST_COUNTS[stage])
                self.assertTrue(all(cohort == ("mnist32" if entry["config"]["dataset"] == "mnist" else "main16")
                                    for entry, cohort in combined))
                self.assertTrue(all(entry["config"]["rho_lr_scale"] == 0 for entry, cohort in combined if cohort == "mnist32"))
            self.assertEqual(len(replay["mnist_selected_variants"]), 16)

    def test_drop_or_configuration_tamper_cannot_enter_final_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            manifests, _, _ = fixture.replay()
            source = manifests["confirm"]
            with self.assertRaisesRegex(ValueError, "complete cohort"):
                assembly.choose_cohort(source["main"], source["mnist"][:-1], "confirm")
            wrong = copy.deepcopy(source["mnist"])
            wrong[0]["config"]["rho_lr_scale"] = .1
            with self.assertRaisesRegex(ValueError, "configuration/identity"):
                assembly.same_entries(wrong, source["mnist"], "MNIST32")
            wrong = copy.deepcopy(source["mnist"])
            wrong[0]["config"]["dataset"] = "fashionmnist"
            with self.assertRaisesRegex(ValueError, "complete cohort"):
                assembly.choose_cohort(source["main"], wrong, "confirm")

    def test_validation_fingerprint_and_selected_variant_tampering_are_rejected(self):
        for mode in ("fingerprint", "finalist", "cohort_status", "config"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = AssemblyFixture(directory)
                if mode == "fingerprint":
                    path = fixture.control / "screen" / "training_results.json"
                    values = json.loads(path.read_text())
                    values[0]["validation_accuracy"] = .99
                    path.write_text(json.dumps(values))
                elif mode == "finalist":
                    path = fixture.control / "selected" / "selected_variants.json"
                    values = json.loads(path.read_text())
                    values[0]["variant_id"] = "another_candidate"
                    path.write_text(json.dumps(values))
                elif mode == "cohort_status":
                    path = fixture.control / "control_status.json"
                    value = json.loads(path.read_text())
                    value["git_sha"] = fixture.code["main_revision"]
                    path.write_text(json.dumps(value))
                else:
                    path = fixture.control / "selected" / "confirm.jsonl"
                    values = read_entries(path)
                    values[0]["config"]["posterior_std"] = .2
                    path.write_text(manifest_text(values))
                with self.assertRaises(ValueError):
                    fixture.replay()

    def test_original_primary_can_win_validation_and_still_uses_control_cohort(self):
        # The32candidate policy retains original-scale candidates when best;
        # source ownership remains the control run for every final MNIST case.
        from journal.build_bayesian_mnist_scale import combine_and_select, extra_entries
        original = screen_entries(datasets=("mnist",))
        extra = extra_entries(original)
        _, _, selected = combine_and_select(original, validation_rows(original, .9), extra, validation_rows(extra, .8))
        control = selected_entries(selected, SEEDS["confirm"], "confirm")
        self.assertTrue(all(entry["config"]["rho_lr_scale"] == .1 for entry in control))
        full = screen_entries()
        primary = selected_entries(select_validation(full, validation_rows(full, .9)), SEEDS["confirm"], "confirm")
        chosen = assembly.choose_cohort(primary, control, "confirm")
        self.assertTrue(all(cohort == "mnist32" for entry, cohort in chosen if entry["config"]["dataset"] == "mnist"))

    def test_raw_pretest_capture_must_match_frozen_original_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            training = fixture.control / "inputs" / "source_capture" / "training_results.json"
            values = json.loads(training.read_text())
            next(row for row in values if row["config"]["dataset"] == "mnist")["validation_accuracy"] = .85
            training.write_text(json.dumps(values))
            capture_path = fixture.control / "inputs" / "capture_protocol.json"
            capture = json.loads(capture_path.read_text())
            capture["capture_fingerprints"][str(training.resolve())] = assembly.fingerprint(training)
            capture_path.write_text(json.dumps(capture))
            snapshot_path = fixture.control / "inputs" / "original" / "original_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text())
            snapshot["source_fingerprints"] = capture["capture_fingerprints"]
            snapshot_path.write_text(json.dumps(snapshot))
            state_path = fixture.control / "control_status.json"
            state = json.loads(state_path.read_text())
            state["original_capture"] = capture
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Raw pre-test MNIST capture differs"):
                fixture.replay()

    def test_non_mnist_screen_grid_cannot_change_even_with_consistent_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            entries = read_entries(fixture.screen)
            next(entry for entry in entries if entry["config"]["dataset"] == "moons")["config"]["lr"] = .023
            fixture.screen.write_text(manifest_text(entries))
            state_path = fixture.main / "pipeline_status.json"
            state = json.loads(state_path.read_text())
            state["stages"]["screen"]["manifest_sha256"] = assembly.fingerprint(fixture.screen)["sha256"]
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "prescribed main16 screening grid"):
                fixture.replay()


class ExactSourceMetricsTests(unittest.TestCase):
    def test_test_metrics_and_saved_validation_metadata_must_match_exactly(self):
        from journal.build_campaign import make_entry
        from journal.run_experiment import normalize_config
        class Evidence:
            def __init__(self, root, manifest):
                self.rows, self.excluded_task_ids, self.files = [{}], [], {}
        config = normalize_config(dict(dataset="fashionmnist", method="erm", predictor="gaussian",
            training_loss="bounded_ce_smooth", threads=2))
        entry = make_entry(config, 73101, "fashionmnist_analog", "ERM-Matched", "bayes_confirm")
        config = entry["config"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / entry["task_id"]
            (checkpoint / "test_full").mkdir(parents=True)
            revision = "a" * 40
            (checkpoint / "metadata.json").write_text(json.dumps(dict(git_sha=revision)))
            for filename in ("weights.pt", "history.json", "task_config.json", "manifest_entry.json", "train.log"):
                (checkpoint / filename).write_text("synthetic validation fixture")
            report_path = checkpoint / "test_full" / "evaluation.json"
            report_path.write_text(json.dumps(dict(heldout_gibbs_mc={"accuracy": .8, "cross_entropy": .4},
                evaluation_config={**config, "eval_draws": 20, "eval_seed": config["eval_seed"] + 1000003})))
            saved = dict(validation_accuracy=.8, validation_cross_entropy=.4, selected_epoch=1,
                         seconds=1., peak_rss_mb=3., git_sha=revision)
            row = dict(task_id=entry["task_id"], status="complete", checkpoint=str(checkpoint),
                       config=config, seed=config["seed"], **saved,
                       **{key: entry[key] for key in ("group", "method_label", "variant_id", "family", "stage")})
            (root / "training_results.json").write_text(json.dumps([row]))
            (root / "summary.json").write_text(json.dumps(dict(training_trials=1, completed_trials=1,
                failed_trials=[], failed_evaluations=[])))
            code = dict(main_revision=revision, mnist_revision="b" * 40, scientific_blobs={})
            test = {**row, "test_status": "complete", "test_accuracy": .8, "test_cross_entropy": .4,
                    "test_report": str(report_path)}
            def load(cohort="main16"):
                return assembly.load_source_stage(root, root / "manifest.jsonl", [entry], "confirm", cohort,
                    assembly.SourceReader(), code, set())
            with patch("journal.plot_bayesian_campaign.Evidence", Evidence), \
                    patch("journal.stage_bayesian_checkpoints.validate_checkpoint", return_value=(saved, "complete")), \
                    patch("journal.bayesian_bounds.observation", return_value={"posterior_kl_float64": 1., "source_fingerprints": {}}), \
                    patch.object(assembly, "verify_checkpoint_revision"):
                (root / "test_results.json").write_text(json.dumps([test]))
                self.assertEqual(len(load()), 1)
                for metric in ("test_accuracy", "test_cross_entropy", "validation_accuracy"):
                    with self.subTest(metric=metric):
                        changed = {**test, metric: test[metric] + 5e-11}
                        (root / "test_results.json").write_text(json.dumps([changed]))
                        with self.assertRaises(ValueError):
                            load()
                (root / "test_results.json").write_text(json.dumps([test]))
                with self.assertRaisesRegex(ValueError, "control cohort's actual checkpoints"):
                    load("mnist32")


class PublishedStagePortableIdentityTests(unittest.TestCase):
    """Use the real publication-audit schema, with a remounted GPFS device ID."""
    def fixture(self, directory):
        root = Path(directory).resolve()
        source = root / "external" / "stage"
        source.mkdir(parents=True)
        target = root / "main" / "p_train_sweep"
        (target.parent / "manifests").mkdir(parents=True)
        target.symlink_to(source, target_is_directory=True)
        entries = [dict(task_id=f"published_{index}") for index in range(672)]
        text = "".join(json.dumps(entry) + "\n" for entry in entries)
        manifest = target.parent / "manifests" / "p_train_sweep.jsonl"
        external_manifest = source.parent / "p_train_sweep.jsonl"
        manifest.write_text(text)
        external_manifest.write_text(text)
        code = dict(main_revision="f602987cceff4a078d12fd33cb6eec8582a1cd3c",
                    scientific_blobs={path: "same_blob" for path in assembly.SOURCE_FILES})
        records = []
        for entry in entries:
            checkpoint = source / entry["task_id"]
            (checkpoint / "test_full").mkdir(parents=True)
            paths = {"metadata": checkpoint / "metadata.json", "weights": checkpoint / "weights.pt",
                     "evaluation": checkpoint / "test_full" / "evaluation.json"}
            for name, path in paths.items():
                path.write_text("Frozen publication " + name + entry["task_id"])
            records.append(dict(task_id=entry["task_id"], original_checkpoint=str(checkpoint),
                original_evaluation=str(paths["evaluation"]),
                source_fingerprints={name: assembly.fingerprint(path) for name, path in paths.items()}))
        stat = source.stat()
        audit = dict(status="published", publication_kind="atomic_directory_symlink", source=str(source), target=str(target),
            source_must_remain_available=True, stage_identity=dict(device=stat.st_dev + 1, inode=stat.st_ino),
            manifest=str(external_manifest), manifest_fingerprint=assembly.fingerprint(manifest),
            resource_job_id="35239215", training_source_verification=dict(main_revision=code["main_revision"],
                external_revision="d1c3b76653cbb6696aecb83485d72c89d4921def",
                files={path: dict(main_blob=blob, external_blob=blob, working_blob=blob)
                       for path, blob in code["scientific_blobs"].items()}),
            complete_validation=dict(tasks=672, evaluation_draws=10, manifest_fingerprint=assembly.fingerprint(manifest),
                complete_checkpoint_and_full_risk_coverage=True, source_records=records, files={}))
        audit_path = source.parent / "publication_audit.json"
        audit_path.write_text(json.dumps(audit))
        return target, source, entries, code, audit_path, audit

    def test_different_host_mount_device_same_inode_and_all_frozen_bytes_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            target, source, entries, code, audit_path, audit = self.fixture(directory)
            reader = assembly.SourceReader()
            result = assembly.published_stage_records(target, entries, "p_train_sweep", "main16", reader, code)
            self.assertEqual(len(result["records"]), 672)
            self.assertNotEqual(result["identity"]["recorded_device"], result["identity"]["observed_device"])
            self.assertEqual(result["identity"]["recorded_inode"], source.stat().st_ino)
            self.assertEqual(result["identity"]["observed_inode"], source.stat().st_ino)
            self.assertEqual(len(reader.files), 3 * 672 + 3)
            reader.recheck()
            self.assertEqual(json.loads(audit_path.read_text()), audit)

    def test_changed_inode_repointed_link_or_frozen_hash_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target, source, entries, code, audit_path, audit = self.fixture(directory)
            def validate():
                return assembly.published_stage_records(target, entries, "p_train_sweep", "main16", assembly.SourceReader(), code)
            altered = copy.deepcopy(audit)
            altered["stage_identity"]["inode"] += 1
            audit_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "permanent-source audit"):
                validate()
            audit_path.write_text(json.dumps(audit))
            replacement = source.parent / "replacement"
            replacement.mkdir()
            target.unlink()
            target.symlink_to(replacement, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "permanent-source audit"):
                validate()
            target.unlink()
            target.symlink_to(source, target_is_directory=True)
            # Even an excluded downstream MNIST-like case must retain every
            # frozen scientific fingerprint; the check precedes cohort filtering.
            weights = source / entries[-1]["task_id"] / "weights.pt"
            weights.write_text("changed scientific content")
            with self.assertRaisesRegex(ValueError, "frozen publication fingerprints"):
                validate()


class AliasAndAssemblyTests(unittest.TestCase):
    def test_leaf_alias_creation_never_replaces_existing_or_broken_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = root / "alias"
            assembly.link_checkpoint(source, target)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())
            with self.assertRaises(ValueError):
                assembly.link_checkpoint(source, target)
            broken = root / "broken"
            broken.symlink_to(root / "missing", target_is_directory=True)
            with self.assertRaises(ValueError):
                assembly.link_checkpoint(source, broken)
            self.assertTrue(broken.is_symlink())

    def fake_stage(self, root, manifest, entries, stage, cohort, reader, code, cache):
        cases = {}
        for entry in entries:
            if cohort == "main16" and entry["config"]["dataset"] == "mnist":
                continue
            path = Path(root) / entry["task_id"]
            path.mkdir()
            (path / "test_full").mkdir()
            report = dict(heldout_gibbs_mc={"bounded_loss": .2}, empirical_gibbs_mc={"bounded_loss": .1},
                          signed_gap_decomposition_mc={"bounded_loss": {"observed_gap": .1}})
            evaluation = path / "test_full" / "evaluation.json"
            evaluation.write_text(json.dumps(report))
            train = dict(task_id=entry["task_id"], group=entry["group"], method_label=entry["method_label"],
                family=entry["family"], variant_id=entry["variant_id"], stage=entry["stage"], config=entry["config"],
                seed=entry["config"]["seed"], status="complete", checkpoint=str(path),
                validation_accuracy=.8, validation_cross_entropy=.4)
            # MNIST32 always has the lower numeric sentinel. The cohort policy
            # must never fall back to old primary MNIST because of this score.
            accuracy = .2 if cohort == "mnist32" else .8
            if entry["method_label"].startswith("T1"):
                accuracy += .01
            test = {**train, "test_status": "complete", "test_accuracy": accuracy,
                    "test_cross_entropy": .4, "test_report": str(evaluation)}
            cases[entry["task_id"]] = dict(entry=entry, training=train, test=test, cohort=cohort,
                checkpoint=path.resolve(), evaluation=evaluation.resolve(), report=report,
                source_fingerprints={}, checkpoint_revision=code["mnist_revision"] if cohort == "mnist32" else code["main_revision"])
        return cases

    def test_full_fixed_composition_and_tables_have_no_test_based_cohort_switch(self):
        class Evidence:
            def __init__(self, root, manifest):
                self.rows, self.excluded_task_ids = read_entries(manifest), []
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            output = fixture.root / "assembled"
            before = {str(path): path.read_bytes() for root in (fixture.main, fixture.control)
                      for path in root.rglob("*") if path.is_file()}
            with patch.object(assembly, "verify_code_sources", return_value=fixture.code), \
                    patch.object(assembly, "load_source_stage", side_effect=self.fake_stage), \
                    patch("journal.plot_bayesian_campaign.Evidence", Evidence), \
                    patch.object(assembly.subprocess, "run", side_effect=AssertionError("Assembly cannot launch training")):
                state = assembly.assemble(fixture.main, fixture.control, output)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["final_runs"], 2048)
            self.assertEqual(state["train_or_inference_jobs_started"], 0)
            self.assertFalse(state["target_met_in_every_condition"])
            self.assertEqual(len(list((output / "evaluation_reports").rglob("*.json"))), 2048)
            for stage in assembly.STAGES:
                entries = read_entries(output / "manifests" / (stage + ".jsonl"))
                self.assertEqual(len(entries), assembly.COUNTS[stage])
                for entry in entries:
                    alias = output / stage / entry["task_id"]
                    self.assertTrue(alias.is_symlink())
                    expected = fixture.control if entry["config"]["dataset"] == "mnist" else fixture.main
                    self.assertTrue(alias.resolve().is_relative_to(expected))
                    self.assertEqual(entry["assembly_source_cohort"], "mnist32" if expected == fixture.control else "main16")
            with (output / "main_means.csv").open() as stream:
                means = list(csv.DictReader(stream))
            with (output / "goal_summary.csv").open() as stream:
                goals = list(csv.DictReader(stream))
            self.assertEqual(len(means), 64)
            self.assertEqual(len(goals), 8)
            self.assertTrue(all(row["target_met"] == "False" for row in goals))
            self.assertTrue(all(row["n"] == "5" for row in means))
            self.assertTrue(all(row["source_cohort"] == "mnist32" and float(row["accuracy_mean"]) < .3
                                for row in means if row["dataset"] == "mnist"))
            audit = json.loads((output / "assembly_audit.json").read_text())
            self.assertEqual(audit["excluded_original_mnist_total"], 512)
            self.assertEqual(len(audit["source_map"]), 2048)
            self.assertFalse(audit["source_bytes_modified"])
            self.assertTrue(all(Path(path).read_bytes() == value for path, value in before.items()))
            with self.assertRaisesRegex(ValueError, "fresh absent"):
                assembly.assemble(fixture.main, fixture.control, output)

    def test_incomplete_source_status_fails_before_any_stage_test_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AssemblyFixture(directory)
            path = fixture.main / "pipeline_status.json"
            state = json.loads(path.read_text())
            state["status"] = "running"
            path.write_text(json.dumps(state))
            with patch.object(assembly, "verify_code_sources", return_value=fixture.code), \
                    patch.object(assembly, "load_source_stage") as loader:
                with self.assertRaisesRegex(ValueError, "completed prescribed main16"):
                    assembly.assemble(fixture.main, fixture.control, fixture.root / "output")
                loader.assert_not_called()
            self.assertFalse((fixture.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
