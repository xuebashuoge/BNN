"""Verify declared evidence coverage and publication formats without training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from journal.plot_bayesian_campaign import Evidence, export
from journal.run_experiment import bounded_loss_description, normalize_config


METHODS = dict(ERM="erm", L2="l2", T1="proposed", T2="joint")


class BayesianPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bayesian-plot-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "confirm"
        self.root.mkdir()
        self.manifest = Path(self.temp.name) / "confirm.jsonl"
        self.entries, self.rows = [], []
        self.add_group("main", "moons_outage", None, None, ("matched", "perfect"))
        for m in (2, 10):
            self.add_group("m_sweep", f"moons_outage_m{m}", "m", m, ("matched",))
        for p in (0., .5):
            self.add_group("p_train_sweep", f"moons_outage_ptr{p:g}", "p_train", p,
                           ("perfect" if p == 0 else "matched",))
        self.save()

    def add_group(self, experiment, group, axis, sweep_value, regimes):
        for method, algorithm in METHODS.items():
            for regime in regimes:
                label = method if experiment == "p_train_sweep" else f"{method}-{regime.title()}"
                for seed in (101, 102):
                    task_id = f"{group}_{method}_{regime}_{seed}"
                    train = dict(kind="perfect") if regime == "perfect" else dict(kind="outage", outage_prob=.5)
                    config = normalize_config(dict(dataset="moons", predictor="gaussian", method=algorithm,
                        training_loss="bounded_ce_smooth", train_size=100, val_size=20, width=8,
                        m=sweep_value if axis == "m" else 100, seed=seed, channel_seed=seed+1000,
                        eval_seed=seed+2000, train_channel=train,
                        deployment_channel=dict(kind="outage", outage_prob=.5)))
                    entry = dict(task_id=task_id, mode="train", group=group, method_label=label,
                        variant_id=f"{group}_{method}_{regime}", family="baseline" if method in {"ERM", "L2"} else "proposed",
                        experiment=experiment, channel_name="outage", training_regime=regime,
                        sweep_axis=axis, sweep_value=sweep_value, config=config, stage="confirm")
                    self.entries.append(entry)
                    self.rows.append({key: entry[key] for key in ("task_id", "group", "method_label", "variant_id", "family", "config")}
                        | dict(seed=seed, test_status="complete", test_accuracy=.8))
                    def risk(accuracy, bounded_loss, cross_entropy, source="deployment", examples=10000):
                        return dict(accuracy=accuracy, bounded_loss=bounded_loss, cross_entropy=cross_entropy,
                            channel_source=source, examples=examples, draws=config["eval_draws"],
                            bounded_loss_definition=bounded_loss_description(config),
                            predictor=dict(stochastic=True, family="diagonal_gaussian"),
                            posterior_rule="one weight draw fixed across entire loader per replicate",
                            replicate_statistics=[dict(accuracy=accuracy, bounded_loss=bounded_loss,
                                cross_entropy=cross_entropy) for _ in range(config["eval_draws"])])
                    report = dict(checkpoint=str(self.root / task_id), split="test", risk_scope="full",
                        evaluation_config=config,
                        heldout_gibbs_mc=risk(.8, .3, .4),
                        empirical_gibbs_mc=risk(.85, .2, .3, "bank", config["train_size"]),
                        same_training_data_fresh_training_channel_gibbs_mc=risk(.82, .24, .34, examples=config["train_size"]),
                        heldout_fresh_training_channel_gibbs_mc=risk(.8, .3, .4),
                        final_posterior_complexities=dict(kl=2., channel_complexity=.2, sample_complexity=.1, joint_complexity=.25),
                        channel_shift_diagnostic=dict(channel_shift_complexity=0.),
                        signed_gap_decomposition_mc={metric: dict(channel_generalization=.04, data_generalization=.06,
                            channel_shift=0., sum=.1, observed_gap=.1) for metric in ("bounded_loss", "cross_entropy")})
                    path = self.root / task_id / "test_full" / "evaluation.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(report))

    def save(self):
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in self.entries))
        (self.root / "test_results.json").write_text(json.dumps(self.rows))

    def test_complete_figures_export_png_pdf_and_eps(self):
        output = Path(self.temp.name) / "figures"
        index = export([(self.root, self.manifest)], output)
        self.assertEqual(len(index["figures"]), 4)
        for figure in index["figures"]:
            self.assertEqual({Path(path).suffix for path in figure["files"]}, {".png", ".pdf", ".eps"})
            signatures = {".png": b"\x89PNG", ".pdf": b"%PDF", ".eps": b"%!PS"}
            for path in figure["files"]:
                self.assertTrue(Path(path).read_bytes().startswith(signatures[Path(path).suffix]))
            self.assertTrue(figure["uncertainty_display"].startswith("none"))
        self.assertTrue((output / "observations.csv").is_file())
        self.assertEqual(index["sources"][0]["declared_tasks"], len(self.entries))

    def test_missing_declared_seed_prevents_all_publication(self):
        self.rows.pop()
        self.save()
        output = Path(self.temp.name) / "figures"
        with self.assertRaisesRegex(ValueError, "Incomplete declared test coverage"):
            export([(self.root, self.manifest)], output)
        self.assertFalse(output.exists())

    def test_deterministic_checkpoint_cannot_be_relabelled_bayesian(self):
        path = self.root / self.entries[0]["task_id"] / "test_full" / "evaluation.json"
        report = json.loads(path.read_text())
        report["heldout_gibbs_mc"]["predictor"]["stochastic"] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Deterministic or unspecified predictor"):
            Evidence(self.root, self.manifest)

    def test_risk_metadata_must_match_actual_loader_and_loss(self):
        path = self.root / self.entries[0]["task_id"] / "test_full" / "evaluation.json"
        original = path.read_text()
        for key, wrong, message in (
                ("posterior_rule", "redraw weights per batch", "entire loader"),
                ("channel_source", "deployment", "channel source"),
                ("examples", 1, "example count"),
                ("bounded_loss_definition", "another loss", "loss definition")):
            with self.subTest(key=key):
                report = json.loads(original)
                report["empirical_gibbs_mc"][key] = wrong
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, message):
                    Evidence(self.root, self.manifest)
        path.write_text(original)

    def test_summary_must_match_frozen_evaluation(self):
        self.rows[0]["test_accuracy"] = .99
        self.save()
        with self.assertRaisesRegex(ValueError, "summary/report mismatch"):
            Evidence(self.root, self.manifest)

    def test_declared_curves_cannot_change_optimizer_per_point(self):
        for entry, row in zip(self.entries, self.rows):
            if entry["experiment"] == "m_sweep" and entry["sweep_value"] == 10:
                entry["config"]["lr"] = row["config"]["lr"] = .123
                path = self.root / entry["task_id"] / "test_full" / "evaluation.json"
                report = json.loads(path.read_text())
                report["evaluation_config"]["lr"] = .123
                path.write_text(json.dumps(report))
        self.save()
        with self.assertRaisesRegex(ValueError, "Sweep changes settings"):
            Evidence(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
