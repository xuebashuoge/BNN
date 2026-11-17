"""Exercise atomic publication, complete-stage gating and immutable provenance.

All campaign execution and model inference are mocked. Linux native renameat2
is tested where available; the mandatory production probe still runs on the
actual target filesystem before any expensive computation.
"""
from __future__ import annotations

from contextlib import ExitStack
import ctypes
import errno
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

from journal import run_parallel_p_train_sweep as parallel
from journal.build_bayesian_campaign import main as build_main, screen_entries
from journal.test_bayesian_campaign import fake_results


def model_noreplace(source, target):
    """Sequential filesystem model used only by mocked orchestration tests."""
    source, target = Path(source), Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(errno.EEXIST, "destination exists", str(target))
    os.rename(source, target)


class AtomicPublicationTests(unittest.TestCase):
    def test_non_linux_and_missing_syscall_fail_closed(self):
        with patch.object(parallel.platform, "system", return_value="Darwin"):
            with self.assertRaises(OSError) as error:
                parallel.rename_noreplace("source", "target")
            self.assertEqual(error.exception.errno, errno.ENOTSUP)
        with patch.object(parallel.platform, "system", return_value="Linux"), \
                patch.object(parallel.ctypes, "CDLL", return_value=object()):
            with self.assertRaises(OSError) as error:
                parallel.rename_noreplace("source", "target")
            self.assertEqual(error.exception.errno, errno.ENOSYS)

    def test_libc_is_called_with_atomic_noreplace_flag_and_errno_is_preserved(self):
        class Function:
            def __call__(self, *args):
                self.args = args
                ctypes.set_errno(errno.EEXIST)
                return -1
        function = Function()
        libc = type("FakeLibc", (), {"renameat2": function})()
        with patch.object(parallel.platform, "system", return_value="Linux"), \
                patch.object(parallel.ctypes, "CDLL", return_value=libc), \
                patch.object(parallel.os, "rename", side_effect=AssertionError("Unsafe fallback forbidden")):
            with self.assertRaises(OSError) as error:
                parallel.rename_noreplace("source", "target")
        self.assertEqual(error.exception.errno, errno.EEXIST)
        self.assertEqual(function.args, (-100, b"source", -100, b"target", 1))

    def test_probe_checks_empty_existing_destination_and_cleans_owned_files(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(parallel, "rename_noreplace", side_effect=model_noreplace):
            parent = Path(directory)
            result = parallel.probe_publication(parent)
            self.assertTrue(result["successful_whole_stage_rename"])
            self.assertIn("EEXIST", result["existing_empty_destination"])
            self.assertEqual(list(parent.iterdir()), [])
        # Plain rename replaces an empty target; the probe must detect it.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(parallel, "rename_noreplace", side_effect=os.rename):
            with self.assertRaisesRegex(RuntimeError, "Unsafe rename"):
                parallel.probe_publication(directory)

    @unittest.skipUnless(platform.system() == "Linux", "Native production primitive requires Linux")
    def test_native_renameat2_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            result = parallel.probe_publication(directory)
            self.assertTrue(result["successful_whole_stage_rename"])


class ExactManifestTests(unittest.TestCase):
    def test_original_absolute_validation_inputs_make_future_main_manifest_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            screen = root / "main" / "manifests" / "screen.jsonl"
            screen.parent.mkdir(parents=True)
            frozen = root / "main" / "validation_selection_results.json"
            entries = screen_entries()
            screen.write_text("".join(json.dumps(item) + "\n" for item in entries))
            frozen.write_text(json.dumps(fake_results(entries)))
            outputs = (root / "external" / "p_train_sweep.jsonl", screen.parent / "p_train_sweep.candidate.jsonl")
            for output in outputs:
                argv = ["builder", "--stage", "p_train_sweep", "--from-manifest", str(screen),
                        "--from-results", str(frozen), "--output", str(output)]
                with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO):
                    build_main()
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(outputs[0].with_suffix(".plan.json").read_bytes(),
                             outputs[1].with_suffix(".plan.json").read_bytes())
            self.assertEqual(len(outputs[0].read_text().splitlines()), 672)


class SourceAndIntegrityTests(unittest.TestCase):
    def test_training_builder_and_coordinator_blobs_must_match_committed_sources(self):
        self.changed = None
        def git(command, **kwargs):
            args = command[1:]
            if args[:2] == ["rev-parse", "--verify"]:
                return "main_sha\n"
            if args == ["rev-parse", "HEAD^{commit}"]:
                return "external_sha\n"
            if args[0] == "status":
                return ""
            if args[0] == "rev-parse":
                revision, filename = args[1].split(":", 1)
                return ("changed" if self.changed == filename and revision == "external_sha" else "blob:" + filename) + "\n"
            if args[0] == "hash-object":
                return "blob:" + args[-1] + "\n"
            self.fail("Unexpected git call")
        with patch.object(parallel.subprocess, "check_output", side_effect=git):
            verified = parallel.verify_source_blobs("main_sha")
            self.assertEqual(set(verified["files"]), set(parallel.SOURCE_FILES))
            self.assertIn("journal/campaign.py", verified["files"])
            self.assertIn("journal/build_bayesian_campaign.py", verified["files"])
            self.changed = "journal/campaign.py"
            with self.assertRaisesRegex(ValueError, "source differs"):
                parallel.verify_source_blobs("main_sha")
            self.changed = "journal/run_parallel_p_train_sweep.py"
            with self.assertRaisesRegex(ValueError, "coordinator differs"):
                parallel.verify_source_blobs("main_sha")

    def test_last_source_byte_check_rejects_changed_weights_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps({"task_id": f"task{index}"}) + "\n" for index in range(672)))
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("metadata.json", "weights.pt", "evaluation.json"):
                (checkpoint / name).write_bytes(b"unchanged frozen bytes")
            records = [dict(task_id=f"task{index}", original_checkpoint=str(checkpoint),
                            original_evaluation=str(checkpoint / "evaluation.json"),
                            source_fingerprints={"metadata": parallel.fingerprint(checkpoint / "metadata.json"),
                                                 "weights": parallel.fingerprint(checkpoint / "weights.pt"),
                                                 "evaluation": parallel.fingerprint(checkpoint / "evaluation.json")})
                       for index in range(672)]
            validation = dict(tasks=672, evaluation_draws=10, complete_checkpoint_and_full_risk_coverage=True,
                              manifest_fingerprint=parallel.fingerprint(manifest), source_records=records, files={})
            parallel.recheck_validated_sources(manifest, validation)
            (checkpoint / "weights.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed before publication"):
                parallel.recheck_validated_sources(manifest, validation)
            (checkpoint / "weights.pt").write_bytes(b"unchanged frozen bytes")
            manifest.write_text("changed declaration")
            with self.assertRaisesRegex(ValueError, "unchanged manifest"):
                parallel.recheck_validated_sources(manifest, validation)

    def test_archive_rejects_symbolic_link_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside").mkdir()
            (root / "publication_snapshot").symlink_to(root / "outside", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real owned directory"):
                parallel.archive_publication_sources(root, root / "stage", root / "manifest", root / "screen", root / "frozen")


class PortableCheckpointTests(unittest.TestCase):
    def test_main_resume_reuses_moved_checkpoint_and_full_report_without_subprocess(self):
        import torch
        from journal.bayesian_bounds import fingerprint, observation
        from journal.build_campaign import make_entry
        from journal.campaign import evaluate_trial, train_trial
        from journal.run_experiment import bounded_loss_description, digest, make_models, normalize_config
        config = normalize_config(dict(dataset="moons", predictor="gaussian", method="erm", width=4,
            train_size=8, val_size=4, epochs=1, m=3, threads=2, training_loss="bounded_ce_smooth",
            train_channel={"kind": "outage", "outage_prob": .5},
            deployment_channel={"kind": "outage", "outage_prob": .5}))
        entry = make_entry(config, 75101, "moons_outage_ptr0500", "ERM", "bayes_p_train_sweep")
        config = normalize_config(entry["config"])
        model, bank = make_models(config, 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            old, moved = Path(directory, "external-stage"), Path(directory, "main-stage")
            checkpoint = old / entry["task_id"]
            checkpoint.mkdir(parents=True)
            metadata = dict(config=config, total_seconds=1., peak_rss_mb=3., selected_epoch=1,
                            git_sha="external_source_sha", git_dirty=False,
                            training_channel_bank_hash=digest({"gain": bank.gain.tolist(), "noise": bank.noise.tolist()}))
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            (checkpoint / "history.json").write_text(json.dumps([
                dict(epoch=1, validation=dict(accuracy=.7, cross_entropy=.5))]))
            torch.save(dict(model=model.state_dict(), bank=bank.state_dict()), checkpoint / "weights.pt")
            empirical = dict(channel_source="bank", examples=8, bounded_loss=.2, accuracy=.8,
                             cross_entropy=.4, draws=10, predictor={"stochastic": True},
                             bounded_loss_definition=bounded_loss_description(config))
            heldout = {**empirical, "channel_source": "deployment", "examples": 10000,
                       "accuracy": .7, "bounded_loss": .3, "cross_entropy": .5}
            evaluated = {**config, "eval_seed": config["eval_seed"] + 1000003, "eval_draws": 10}
            report = dict(checkpoint=str(checkpoint), split="test", risk_scope="full",
                          evaluation_config=evaluated, empirical_gibbs_mc=empirical, heldout_gibbs_mc=heldout,
                          checkpoint_source_fingerprints={name: fingerprint(checkpoint / filename) for name, filename in
                              (("metadata", "metadata.json"), ("weights", "weights.pt"))},
                          checkpoint_verification=dict(saved_channel_bank_hash=metadata["training_channel_bank_hash"],
                              bank_hash_matches_metadata=True, split_hash_matches=dict(
                                  train_indices_hash=True, validation_indices_hash=True)))
            (checkpoint / "test_full").mkdir()
            (checkpoint / "test_full" / "evaluation.json").write_text(json.dumps(report))
            before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}
            model_noreplace(old, moved)
            with patch("journal.campaign.subprocess.run", side_effect=AssertionError("Cached resume cannot train or infer")):
                trained = train_trial(entry, moved, threads=2, resume=True, device="cpu")
                evaluated_row = evaluate_trial(trained, draws=10, full_risks=True, resume=True)
            self.assertEqual(trained["status"], "complete")
            self.assertTrue(trained["resumed"])
            self.assertEqual(Path(trained["checkpoint"]), (moved / entry["task_id"]).resolve())
            self.assertEqual(evaluated_row["test_status"], "complete")
            self.assertEqual(Path(evaluated_row["test_report"]),
                             (moved / entry["task_id"] / "test_full" / "evaluation.json").resolve())
            checked = observation(entry, evaluated_row, moved, 1, .025)
            self.assertTrue(checked["posterior_kl_float64"] > 0)
            self.assertEqual(before, {str(path.relative_to(moved)): path.read_bytes()
                                      for path in moved.rglob("*") if path.is_file()})


class WholeStageOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.validation_called = False
        self.source_checks = 0
        self.race_destination = None
        self.fail_validation = False
        self.fail_after_rename = False
        self.fail_before_rename = False

    def fixture(self, directory):
        root = Path(directory).resolve()
        main = root / "main"
        (main / "manifests").mkdir(parents=True)
        screen, frozen = main / "manifests" / "screen.jsonl", main / "validation_selection_results.json"
        screen.write_text("fixed immutable screen\n")
        frozen.write_text("fixed immutable validation\n")
        state = dict(git_sha="main_sha", status="running", current_stage="confirm",
                     pilot_resources={"max_worker_peak_rss_mb": 100.},
                     stages={"screen": {"manifest_sha256": parallel.fingerprint(screen)["sha256"]}})
        (main / "pipeline_status.json").write_text(json.dumps(state))
        output = root / "external"
        verification = dict(main_revision="main_sha", external_revision="external_sha",
                            files={name: {"main_blob": "unchanged", "external_blob": "unchanged", "working_blob": "unchanged"}
                                   for name in parallel.SOURCE_FILES})
        return main, output, verification

    def fake_command(self, args, log, environment):
        self.commands.append((list(args), dict(environment)))
        if args[0] == "journal.build_bayesian_campaign":
            self.assertEqual(args[args.index("--from-manifest") + 1], str(self.main / "manifests" / "screen.jsonl"))
            self.assertEqual(args[args.index("--from-results") + 1], str(self.main / "validation_selection_results.json"))
            output = Path(args[args.index("--output") + 1])
            entries = [dict(task_id=f"task{index}") for index in range(672)]
            output.write_text("".join(json.dumps(item) + "\n" for item in entries))
            output.with_suffix(".plan.json").write_text(json.dumps(dict(tasks=672)))
        elif args[0] == "journal.campaign":
            stage = Path(args[args.index("--output-root") + 1])
            stage.mkdir()
            for filename in parallel.GLOBAL_REPORTS:
                (stage / filename).write_text(json.dumps({"original": filename, "path": str(stage)}))
            (stage / "completed-marker").write_text("all672")
        else:
            self.fail("Unexpected subprocess module")

    def fake_validate(self, stage, manifest, execution_revision):
        self.assertEqual(execution_revision, "external_sha")
        self.assertEqual((Path(stage) / "completed-marker").read_text(), "all672")
        self.validation_called = True
        if self.fail_validation:
            raise ValueError("Incomplete full risk coverage")
        return dict(tasks=672, evaluation_draws=10, complete_checkpoint_and_full_risk_coverage=True,
                    manifest_fingerprint=parallel.fingerprint(manifest),
                    source_records=[dict(task_id=f"task{index}", original_checkpoint=str(Path(stage) / f"task{index}"),
                                         original_evaluation=str(Path(stage) / f"task{index}" / "test_full" / "evaluation.json"),
                                         source_fingerprints={}) for index in range(672)], files={})

    def fake_rename(self, source, destination):
        self.assertTrue(self.validation_called)
        if self.fail_before_rename:
            raise InterruptedError(errno.EINTR, "interrupted before atomic commit")
        if self.race_destination:
            Path(destination).mkdir()
            if self.race_destination == "nonempty":
                (Path(destination) / "main-active").write_text("preserve-main")
        model_noreplace(source, destination)
        if self.fail_after_rename:
            raise OSError("simulated interruption after kernel committed rename")

    def execute(self, main, output, verification, extra=None):
        self.main = main
        with ExitStack() as stack:
            stack.enter_context(patch.object(parallel, "verify_source_blobs", return_value=verification))
            stack.enter_context(patch("journal.batch_campaign.validate_cpu_budget", return_value=32))
            stack.enter_context(patch.object(parallel, "probe_publication", return_value={"probe": "passed"}))
            stack.enter_context(patch.object(parallel, "command", side_effect=self.fake_command))
            stack.enter_context(patch.object(parallel, "validate_stage", side_effect=self.fake_validate))
            stack.enter_context(patch.object(parallel, "recheck_validated_sources"))
            stack.enter_context(patch.object(parallel, "rename_noreplace", side_effect=self.fake_rename))
            stack.enter_context(patch.dict(os.environ, {"JOURNAL_GIT_SHA": "external_sha", "SLURM_MEM_PER_NODE": "32768", "SLURM_JOB_ID": "parallel_job"}))
            if extra:
                extra(stack)
            return parallel.run(main, "main_sha", output)

    def test_only_completed_whole_stage_is_published_and_original_rows_are_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            main, output, verification = self.fixture(directory)
            before_status = (main / "pipeline_status.json").read_bytes()
            result = self.execute(main, output, verification)
            self.assertEqual(result["status"], "published")
            self.assertFalse((output / "stage").exists())
            target = main / "p_train_sweep"
            self.assertEqual((target / "completed-marker").read_text(), "all672")
            self.assertEqual((main / "pipeline_status.json").read_bytes(), before_status)
            self.assertFalse((main / "manifests" / "p_train_sweep.jsonl").exists())
            self.assertEqual([args[0] for args, env in self.commands], ["journal.build_bayesian_campaign", "journal.campaign"])
            campaign = self.commands[1][0]
            self.assertIn("--full-risks", campaign)
            self.assertEqual(campaign[campaign.index("--eval-draws") + 1], "10")
            self.assertEqual(campaign[campaign.index("--workers") + 1], "16")
            audit = json.loads((output / "publication_audit.json").read_text())
            self.assertEqual(audit["status"], "published")
            self.assertEqual(len(audit["task_path_map"]), 672)
            self.assertEqual(audit["resource_job_id"], "parallel_job")
            archived = output / "publication_snapshot" / "test_results.json"
            original = archived.read_bytes()
            (target / "test_results.json").write_text("main resumed and rewrote path strings")
            self.assertEqual(archived.read_bytes(), original)

    def test_existing_empty_or_nonempty_destination_prevents_any_computation(self):
        for nonempty in (False, True):
            with self.subTest(nonempty=nonempty), tempfile.TemporaryDirectory() as directory:
                self.setUp()
                main, output, verification = self.fixture(directory)
                target = main / "p_train_sweep"
                target.mkdir()
                if nonempty:
                    (target / "existing").write_text("main")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.execute(main, output, verification)
                self.assertEqual(self.commands, [])
                self.assertEqual(any(target.iterdir()), nonempty)

    def test_destination_created_during_publication_race_preserves_both_stages(self):
        for kind in ("empty", "nonempty"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                self.setUp()
                main, output, verification = self.fixture(directory)
                self.race_destination = kind
                with self.assertRaises(FileExistsError):
                    self.execute(main, output, verification)
                self.assertTrue((output / "stage" / "completed-marker").exists())
                target = main / "p_train_sweep"
                if kind == "nonempty":
                    self.assertEqual((target / "main-active").read_text(), "preserve-main")
                else:
                    self.assertEqual(list(target.iterdir()), [])
                self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed_unpublished")

    def test_invalid_reports_and_late_main_stage_block_publication(self):
        for mode in ("invalid_reports", "late_stage"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.setUp()
                main, output, verification = self.fixture(directory)
                if mode == "invalid_reports":
                    self.fail_validation = True
                def advance(stack):
                    if mode == "late_stage":
                        original = self.fake_validate
                        def validate(*args):
                            result = original(*args)
                            status = json.loads((main / "pipeline_status.json").read_text())
                            status["current_stage"] = "p_train_sweep"
                            (main / "pipeline_status.json").write_text(json.dumps(status))
                            return result
                        stack.enter_context(patch.object(parallel, "validate_stage", side_effect=validate))
                with self.assertRaises(ValueError):
                    self.execute(main, output, verification, advance)
                self.assertFalse((main / "p_train_sweep").exists())
                self.assertTrue((output / "stage").exists())

    def test_post_commit_interruption_is_never_labeled_unpublished_and_resume_does_no_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            main, output, verification = self.fixture(directory)
            self.fail_after_rename = True
            with self.assertRaisesRegex(OSError, "after kernel committed"):
                self.execute(main, output, verification)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "published_with_postprocessing_error")
            target_inode = (main / "p_train_sweep").stat().st_ino
            main_state = json.loads((main / "pipeline_status.json").read_text())
            main_state["current_stage"] = "p_train_sweep"
            (main / "pipeline_status.json").write_text(json.dumps(main_state))
            self.commands = []
            self.fail_after_rename = False
            result = self.execute(main, output, verification)
            self.assertEqual(result["status"], "published")
            self.assertEqual(self.commands, [])
            self.assertEqual((main / "p_train_sweep").stat().st_ino, target_inode)

    def test_failed_or_later_main_status_refuses_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            main, output, verification = self.fixture(directory)
            state = json.loads((main / "pipeline_status.json").read_text())
            state["status"] = "failed"
            (main / "pipeline_status.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "already reached"):
                self.execute(main, output, verification)
            self.assertEqual(self.commands, [])

    def test_interrupted_unpublished_complete_stage_resumes_without_rewriting_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            main, output, verification = self.fixture(directory)
            self.fail_before_rename = True
            with self.assertRaises(InterruptedError):
                self.execute(main, output, verification)
            archived = (output / "publication_snapshot" / "test_results.json").read_bytes()
            self.commands = []
            self.fail_before_rename = False
            result = self.execute(main, output, verification)
            self.assertTrue(result["resumed_complete_stage"])
            self.assertEqual(result["status"], "published")
            self.assertFalse(any(args[0] == "journal.campaign" for args, env in self.commands))
            self.assertEqual((output / "publication_snapshot" / "test_results.json").read_bytes(), archived)

    def test_memory_and_unsupported_filesystem_gates_precede_expensive_commands(self):
        for mode in ("memory", "filesystem"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.setUp()
                main, output, verification = self.fixture(directory)
                def gate(stack):
                    if mode == "memory":
                        stack.enter_context(patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "4000"}))
                    else:
                        stack.enter_context(patch.object(parallel, "probe_publication",
                            side_effect=OSError(errno.ENOTSUP, "unsupported filesystem")))
                with self.assertRaises((OSError, RuntimeError)):
                    self.execute(main, output, verification, gate)
                self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
