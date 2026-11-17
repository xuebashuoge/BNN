"""Verify transparent seed search, frozen inputs and fixed fresh-MC caching."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from journal.build_bayesian_campaign import CHANNELS, DATASETS, METHODS, SEEDS, base_config, entry
from journal.run_experiment import bounded_loss_description, normalize_config
from journal import verify_bayesian_seed_examples as examples


def scores(label, seed, winning=True):
    if not winning:
        return .79 if label in examples.PROPOSED else .8
    return .9 if label == "T1-Matched" and seed == 73103 else .82 if label == "T2-Matched" else .81 if label == "T1-Matched" else .8


def cohort_rows(winning=True):
    return [dict(task_id=f"{dataset}_{law}_{label}_{seed}", group=f"{dataset}_{law}",
        method_label=label, seed=seed, test_status="complete", test_accuracy=scores(label, seed, winning))
        for dataset in DATASETS for law in CHANNELS for label in examples.LABELS for seed in SEEDS["confirm"]]


class AssembledFixture:
    """Synthetic complete320-case source; no model tensors or inference."""
    def __init__(self, root, winning=True):
        self.root = Path(root).resolve()
        (self.root / "confirm").mkdir(parents=True)
        (self.root / "manifests").mkdir()
        (self.root / "evaluation_reports" / "confirm").mkdir(parents=True)
        from journal.assemble_bayesian_results import POLICY
        from journal.run_bayesian_campaign import ranking_outcome
        entries, rows, cases, audited_files = [], [], [], {}
        self.physical_main = self.root.parent / "physical_sources" / "main"
        self.physical_mnist = self.root.parent / "physical_sources" / "mnist"
        assembly_code = dict(main_revision="main", mnist_revision="mnist", scientific_blobs={})
        for dataset in DATASETS:
            for law in CHANNELS:
                for method, name in METHODS:
                    for regime in ("matched", "perfect"):
                        cfg = base_config(dataset, law)
                        cfg["method"] = method
                        if regime == "perfect":
                            cfg["train_channel"] = dict(kind="perfect")
                        label = f"{name}-{regime.title()}"
                        for seed in SEEDS["confirm"]:
                            source = entry(cfg, seed, dataset, law, label, "confirm", regime)
                            cohort = "mnist32" if dataset == "mnist" else "main16"
                            source["assembly_source_cohort"] = cohort
                            config = normalize_config(source["config"])
                            task_id = source["task_id"]
                            checkpoint = (self.physical_mnist if dataset == "mnist" else self.physical_main) / "confirm" / task_id
                            checkpoint.mkdir(parents=True)
                            revision = "mnist" if dataset == "mnist" else "main"
                            (checkpoint / "metadata.json").write_text(json.dumps(dict(config=config, git_dirty=False, git_sha=revision)))
                            for filename in examples.CHECKPOINT_FILES[1:]:
                                (checkpoint / filename).write_text("Synthetic immutable scientific source")
                            (checkpoint / "task_config.json").write_text(json.dumps(config))
                            (checkpoint / "manifest_entry.json").write_text(json.dumps(source))
                            for filename in examples.CHECKPOINT_FILES:
                                path = checkpoint / filename
                                audited_files[str(path)] = dict(resolved=str(path.resolve()), **examples.fingerprint(path))
                            (self.root / "confirm" / task_id).symlink_to(checkpoint, target_is_directory=True)
                            source_fps = {key: examples.fingerprint(checkpoint / filename) for key, filename in (
                                ("metadata", "metadata.json"), ("weights", "weights.pt"))}
                            accuracy = scores(label, seed, winning)
                            def risk(acc, bounded, ce, source="deployment", count=10000):
                                return dict(accuracy=acc, bounded_loss=bounded, cross_entropy=ce,
                                    channel_source=source, examples=count, draws=20,
                                    bounded_loss_definition=bounded_loss_description(config),
                                    predictor=dict(stochastic=True, family="diagonal_gaussian"),
                                    posterior_rule="one weight draw fixed across entire loader per replicate",
                                    replicate_statistics=[dict(accuracy=acc, bounded_loss=bounded, cross_entropy=ce) for _ in range(20)])
                            report = dict(checkpoint=str(checkpoint.resolve()), split="test", risk_scope="full",
                                evaluation_config={**config, "eval_draws": 20, "eval_seed": config["eval_seed"]+1000003},
                                checkpoint_source_fingerprints=source_fps,
                                heldout_gibbs_mc=risk(accuracy, .3, .4),
                                empirical_gibbs_mc=risk(.85, .2, .3, "bank", config["train_size"]),
                                same_training_data_fresh_training_channel_gibbs_mc=risk(.82, .24, .34, count=config["train_size"]),
                                heldout_fresh_training_channel_gibbs_mc=risk(accuracy, .3, .4),
                                final_posterior_complexities=dict(kl=2., channel_complexity=.2, sample_complexity=.1, joint_complexity=.25),
                                channel_shift_diagnostic=dict(channel_shift_complexity=0.),
                                signed_gap_decomposition_mc={metric: dict(channel_generalization=.04, data_generalization=.06,
                                    channel_shift=0., sum=.1, observed_gap=.1) for metric in ("bounded_loss", "cross_entropy")})
                            report_path = self.root / "evaluation_reports" / "confirm" / f"{task_id}.json"
                            report_path.write_text(json.dumps(report))
                            original_report = checkpoint / "test_full" / "evaluation.json"
                            original_report.parent.mkdir()
                            original_report.write_bytes(report_path.read_bytes())
                            audited_files[str(original_report)] = dict(resolved=str(original_report.resolve()), **examples.fingerprint(original_report))
                            cases.append(dict(stage="confirm", task_id=task_id, cohort=cohort,
                                alias=str(self.root / "confirm" / task_id), actual_checkpoint=str(checkpoint),
                                source_evaluation=str(original_report), exported_evaluation=str(report_path),
                                exported_evaluation_fingerprint=examples.fingerprint(report_path), checkpoint_revision=revision,
                                source_fingerprints={**source_fps, "evaluation": examples.fingerprint(original_report)}))
                            entries.append(source)
                            rows.append({key: source[key] for key in ("task_id", "group", "method_label", "variant_id", "family", "config")}
                                | dict(seed=seed, test_status="complete", test_accuracy=accuracy, test_cross_entropy=.4,
                                       checkpoint=str(self.root / "confirm" / task_id), test_report=str(report_path), assembly_source_cohort=cohort))
        (self.root / "manifests" / "confirm.jsonl").write_text("".join(json.dumps(source)+"\n" for source in entries))
        (self.root / "confirm" / "test_results.json").write_text(json.dumps(rows))
        for stage, count in (("m_sweep", 1056), ("p_train_sweep", 672)):
            cases.extend(dict(stage=stage, task_id=f"unused_{stage}_{index}",
                actual_checkpoint=str(self.physical_main / stage / f"unused_{index}")) for index in range(count))
        (self.root / "assembly_status.json").write_text(json.dumps(dict(status="complete", final_runs=2048, policy=POLICY,
            main_root=str(self.physical_main), mnist_root=str(self.physical_mnist), code_sources=assembly_code,
            stages=dict(confirm=dict(tasks=320, manifest_fingerprint=examples.fingerprint(self.root / "manifests" / "confirm.jsonl")),
                        m_sweep=dict(tasks=1056), p_train_sweep=dict(tasks=672)))))
        (self.root / "assembly_audit.json").write_text(json.dumps(dict(scope="Synthetic fixtures only", policy=POLICY,
            code_sources=assembly_code, source_map=cases, source_files=audited_files, source_bytes_modified=False)))
        means = []
        for case in examples.select_examples(rows):
            for label, stats in case["original_five_seed_means"].items():
                means.append(dict(group=case["group"], method_label=label, n=5, accuracy_mean=stats["mean"],
                                  accuracy_sd=stats["sd"], seeds_json=json.dumps(list(SEEDS["confirm"])) ))
        (self.root / "main_means.json").write_text(json.dumps(means))
        (self.root / "accuracy_outcome.json").write_text(json.dumps({**ranking_outcome(self.root / "confirm", manifest=self.root / "manifests" / "confirm.jsonl"),
                                                                  "policy": POLICY, "accuracy_tolerance": examples.TOLERANCE}))
        self.rows = rows


class SeedSelectionTests(unittest.TestCase):
    def test_committed_evaluator_sources_must_match_original_main_blobs(self):
        with tempfile.TemporaryDirectory(prefix="seed-example-source-test-") as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
            git("init", "--quiet")
            for filename in (*examples.SOURCE_FILES, "journal/verify_bayesian_seed_examples.py"):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixed source\n")
            git("add", "journal")
            git("-c", "user.name=Seed Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "source")
            source = git("rev-parse", "HEAD")
            self.assertEqual(len(examples.verify_code(source, root)["scientific_blobs"]), 7)
            (root / examples.SOURCE_FILES[0]).write_text("# changed evaluator\n")
            git("add", examples.SOURCE_FILES[0])
            git("-c", "user.name=Seed Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "changed")
            with self.assertRaisesRegex(ValueError, "differs from prescribed main"):
                examples.verify_code(source, root)

    def test_common_paired_seed_maximizes_margin_and_all_five_means_remain(self):
        result = examples.select_examples(cohort_rows())
        self.assertEqual(len(result), 8)
        for case in result:
            self.assertEqual(case["selected_seed"], 73103)
            self.assertEqual(case["initial_candidate"], "T1-Matched")
            self.assertEqual(len(case["selected_task_ids"]), 8)
            self.assertEqual(len(case["all_five_seed_search"]), 5)
            self.assertEqual(len(case["original_five_seed_means"]), 8)
            self.assertIn("Test-selected exploratory", case["interpretation"])

    def test_no_strict_win_and_ties_report_none_without_new_seeds(self):
        rows = cohort_rows(False)
        self.assertTrue(all(case["selected_seed"] is None for case in examples.select_examples(rows)))
        for row in rows:
            row["test_accuracy"] = .8
        self.assertTrue(all(case["selected_seed"] is None for case in examples.select_examples(rows)))
        with self.assertRaisesRegex(ValueError, "320-case"):
            examples.select_examples(rows[:-1])

    def test_fresh_winner_change_and_negative_margin_are_reported_without_retry(self):
        case = examples.select_examples(cohort_rows())[0]
        values = []
        for task_id, label in zip(case["selected_task_ids"], examples.LABELS):
            acc = .7 if label == "T1-Matched" else .85 if label == "T2-Matched" else .9 if label == "ERM-Matched" else .8
            values.append(dict(task_id=task_id, method_label=label,
                statistics=dict(accuracy=examples.statistics_of([acc]*500)), accuracy_draws=[acc]*500))
        report = examples.summarize(dict(conditions=[case], original_main_means=[], original_mean_rankings={}), values)
        result = report["conditions"][0]
        self.assertTrue(result["winner_changed"])
        self.assertLess(result["fresh_fixed_initial_candidate"]["margin"], 0)
        self.assertFalse(result["fresh_best_matched_proposed"]["strictly_best"])
        self.assertEqual(len(result["methods"]), 8)
        self.assertFalse(report["retry_after_loss"])
        self.assertEqual(result["paired_mc_differences"]["T1-Matched"]["ERM-Matched"]["n"], 500)


class FrozenVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="exploratory-seed-example-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.output = self.base / "assembled", self.base / "exploratory"
        self.fixture = AssembledFixture(self.source)
        self.code = dict(main_revision="main", execution_revision="execution", scientific_blobs={}, driver_blob="blob")
        self.protocol = examples.freeze_selection(self.source, self.output, self.code)
        chosen = self.protocol["conditions"][0]["selected_task_ids"][0]
        self.task = next(task for task in self.protocol["all_original_tasks"] if task["task_id"] == chosen)
        self.protocol_hash = examples.fingerprint(self.output / "selection_protocol.json")["sha256"]

    def fake_inference(self, command, **kwargs):
        self.assertTrue((self.output / "selection_protocol.json").is_file())
        self.assertEqual(command[command.index("--mode")+1], "evaluate")
        self.assertEqual(command[command.index("--risk-scope")+1], "deployment")
        override = json.loads(Path(command[command.index("--config")+1]).read_text())
        self.assertEqual(override["eval_draws"], 500)
        self.assertEqual(override["eval_seed"], self.task["config"]["eval_seed"]+9000019)
        risk = dict(accuracy=.7, bounded_loss=.3, cross_entropy=.4, draws=500, examples=10000,
            channel_source="deployment", bounded_loss_definition=bounded_loss_description(self.task["config"]),
            posterior_rule="one weight draw fixed across entire loader per replicate",
            predictor=dict(stochastic=True, family="diagonal_gaussian"),
            replicate_statistics=[dict(accuracy=.7, bounded_loss=.3, cross_entropy=.4) for _ in range(500)])
        report = dict(split="test", risk_scope="deployment", checkpoint=self.task["checkpoint"],
            evaluation_config={**self.task["config"], **override},
            checkpoint_source_fingerprints=self.task["checkpoint_source_fingerprints"], heldout_gibbs_mc=risk)
        output = Path(command[command.index("--output")+1])
        (output / "evaluation.json").write_text(json.dumps(report))

    def test_source_freeze_contains_all320_and_detects_changed_original(self):
        self.assertEqual(len(self.protocol["all_original_tasks"]), 320)
        self.assertTrue(all(len(case["all_five_seed_search"]) == 5 for case in self.protocol["conditions"]))
        self.assertEqual(examples.freeze_selection(self.source, self.output, self.code), self.protocol)
        means = self.source / "main_means.json"
        means.write_text(means.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "Cached exploratory selection protocol"):
            examples.freeze_selection(self.source, self.output, self.code)

    def test_fresh_stream_frozen_request_and_exact_cache_without_loss_retry(self):
        before = copy.deepcopy(self.protocol["source_fingerprints"])
        with patch.object(examples.subprocess, "run", side_effect=self.fake_inference) as inference:
            first = examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
            second = examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
            self.assertEqual(first, second)
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(first["statistics"]["accuracy"]["mean"], .7)
            self.assertEqual(len(first["accuracy_draws"]), 500)
            path = self.output / "evaluations" / self.task["task_id"] / "evaluation.json"
            report = json.loads(path.read_text())
            report["evaluation_config"]["eval_seed"] = 0
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "500-draw request"):
                examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
            self.assertEqual(inference.call_count, 1)
        self.assertEqual(before, {path: examples.fingerprint(path) for path in before})

    def test_changed_request_is_not_reused_or_reinferred(self):
        with patch.object(examples.subprocess, "run", side_effect=self.fake_inference) as inference:
            examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
            with self.assertRaisesRegex(ValueError, "Immutable"):
                examples.evaluate_one(self.task, self.output, "another protocol hash", {})
            self.assertEqual(inference.call_count, 1)

    def test_cached_protocol_selection_and_fresh_seed_edits_fail_before_any_request(self):
        path = self.output / "selection_protocol.json"
        original = path.read_text()
        def changed_seed(protocol):
            protocol["conditions"][0]["selected_seed"] = 73101
        def changed_candidate(protocol):
            protocol["conditions"][0]["initial_candidate"] = "T2-Matched"
        def changed_task(protocol):
            protocol["all_original_tasks"][0]["fresh_evaluation_seed"] += 1
        def absolute_task(protocol):
            protocol["all_original_tasks"][0]["task_id"] = "/tmp/escape"
        for change in (changed_seed, changed_candidate, changed_task, absolute_task):
            with self.subTest(change=change.__name__):
                protocol = json.loads(original)
                change(protocol)
                path.write_text(json.dumps(protocol))
                with patch.object(examples.subprocess, "run", side_effect=AssertionError("Tampering must fail before MC")):
                    with self.assertRaisesRegex(ValueError, "complete source replay"):
                        examples.freeze_selection(self.source, self.output, self.code)
                self.assertFalse((self.output / "evaluations").exists())
        path.write_text(original)

    def test_physical_source_overlap_rejected_before_output_or_lock_creation(self):
        for source in (Path(self.task["checkpoint"]), self.fixture.physical_main / "unrelated_source_subfolder"):
            output = source / "new_exploratory_run"
            with self.subTest(output=output):
                with patch.object(examples, "verify_code", return_value=self.code), \
                     patch("journal.batch_campaign.validate_cpu_budget", return_value=32), \
                     patch.dict("os.environ", {"JOURNAL_GIT_SHA": "execution"}):
                    with self.assertRaisesRegex(ValueError, "protected assembled or physical source"):
                        examples.run(self.source, output)
                self.assertFalse(output.exists())
        # The permanent external p-stage may live outside both main/control
        # roots. Protect its audited physical tree without reading p scores.
        audit_path = self.source / "assembly_audit.json"
        audit = json.loads(audit_path.read_text())
        external = self.base / "permanent_external_ptr"
        external_stage = external / "p_train_sweep"
        external_stage.mkdir(parents=True)
        case = next(case for case in audit["source_map"] if case["stage"] == "p_train_sweep")
        case["actual_checkpoint"] = str(external_stage / case["task_id"])
        audit["source_stages"] = {"external_ptr": dict(resolved=str(external_stage))}
        audit_path.write_text(json.dumps(audit))
        output = external / "new_exploratory_run"
        with self.assertRaisesRegex(ValueError, "protected assembled or physical source"):
            examples.preflight_output(self.source, output)
        self.assertFalse(output.exists())

    def test_safe_task_ids_and_writable_cache_aliases_cannot_escape_owned_root(self):
        task = copy.deepcopy(self.task)
        task["task_id"] = "/tmp/forbidden_exploratory_escape"
        with self.assertRaisesRegex(ValueError, "safe ASCII"):
            examples.evaluate_one(task, self.output, self.protocol_hash, {})
        directory = self.output / "evaluations" / self.task["task_id"]
        directory.parent.mkdir()
        directory.symlink_to(self.task["checkpoint"], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
        directory.unlink()
        directory.mkdir()
        for filename in ("request.json", "overrides.json", "evaluation.json.tmp", "evaluate.log"):
            alias = directory / filename
            alias.symlink_to(self.base / "must_remain_absent")
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    examples.evaluate_one(self.task, self.output, self.protocol_hash, {})
                self.assertFalse((self.base / "must_remain_absent").exists())
            alias.unlink()

    def test_assembly_policy_metadata_and_mean_rankings_are_attested(self):
        outcome = self.source / "accuracy_outcome.json"
        original = outcome.read_text()
        changed = json.loads(original)
        changed["target_met_in_every_condition"] = not changed["target_met_in_every_condition"]
        outcome.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "mean-ranking outcome"):
            examples.freeze_selection(self.source, self.output, self.code)
        outcome.write_text(original)
        metadata_path = Path(self.task["checkpoint"]) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["config"]["lr"] = .123
        metadata_path.write_text(json.dumps(metadata))
        audit_path = self.source / "assembly_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["source_files"][str(metadata_path)] = dict(resolved=str(metadata_path), **examples.fingerprint(metadata_path))
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "Actual checkpoint configuration"):
            examples.freeze_selection(self.source, self.output, self.code)

    def test_no_winning_seed_driver_performs_no_new_inference(self):
        protocol = copy.deepcopy(self.protocol)
        for case in protocol["conditions"]:
            case.update(selected_seed=None, selected_task_ids=[], initial_candidate=None,
                        initial_strongest_other=None, initial_margin=None,
                        selection_status="no_strictly_winning_paired_seed")
        output = self.base / "no_winner_verification"
        def freeze(*args):
            examples.write_once(output / "selection_protocol.json", examples.encoded(protocol))
            return protocol
        with patch.object(examples, "verify_code", return_value=self.code), \
             patch.object(examples, "freeze_selection", side_effect=freeze), \
             patch("journal.batch_campaign.validate_cpu_budget", return_value=32), \
             patch.object(examples.subprocess, "run", side_effect=AssertionError("No winning seed must not trigger inference")), \
             patch.dict("os.environ", {"JOURNAL_GIT_SHA": "execution"}):
            report = examples.run(self.source, output)
        self.assertEqual(len(report["conditions"]), 8)
        self.assertTrue(all(case["fresh_verification"] == "not_run_no_initial_winning_seed" for case in report["conditions"]))
        self.assertTrue(report["original_five_seed_means_retained"])


if __name__ == "__main__":
    unittest.main()
