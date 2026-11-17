"""Check corrected risk-RHS plotting using synthetic, explicitly separate data."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from journal.plot_bayesian_certificate import Evidence, export, fingerprint
from journal.run_experiment import bounded_loss_description, normalize_config


class CertificatePlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="small-model-plot-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "risk_bounds.json"
        (self.root / "rows").mkdir()
        radius = math.sqrt(math.log(1 / (.2 * .025 / 48)) / 2000)
        confidence = dict(total_failure=.025, simultaneous_runs=48, simultaneous_theorems=2,
                          failure_per_theorem_run=.8*.025/(2*48), mc_failure_per_run=.2*.025/48,
                          empirical_mc_upper_radius=radius)
        rows = []
        for law in ("analog", "outage"):
            deployment = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.) if law == "analog" else dict(kind="outage", outage_prob=.5)
            for index, (method, algorithm) in enumerate((("ERM", "erm"), ("L2", "l2"), ("T1", "proposed"), ("T2", "joint"))):
                for regime in ("matched", "perfect"):
                    for seed in (76101, 76102, 76103):
                        task_id = f"fixture_{law}_{method}_{regime}_{seed}"
                        training = dict(kind="perfect") if regime == "perfect" else deployment
                        config = normalize_config(dict(dataset="moons", architecture="mlp", predictor="gaussian",
                            method=algorithm, width=8, train_size=512, val_size=512, m=2000,
                            seed=seed, channel_seed=seed+10000, eval_seed=seed+20000, eval_draws=20,
                            training_loss="bounded_ce_smooth", train_channel=training, deployment_channel=deployment))
                        empirical = .08 + index*.005
                        upper = empirical + radius
                        channel = .04 if regime == "matched" else 0.
                        sample = .03
                        joint = .05 if regime == "matched" else sample
                        shift = 0. if regime == "matched" else 1. if law == "analog" else 1.-.5**8
                        def bound(complexity):
                            rhs = upper + shift + complexity
                            return dict(rhs_uncapped=rhs, population_risk_upper=min(1., rhs), nonvacuous=rhs < 1)
                        row = dict(task_id=task_id, group=f"moons_{law}_certificate_w8_n512_m2000",
                            method_label=f"{method}-{regime.title()}", seed=seed, dataset="moons",
                            stage="bayes_certificate", channel_name=law, training_regime=regime,
                            config=config, n=512, m=2000, width=8, train_channel=training,
                            deployment_channel=deployment, accuracy=.85+index*.01, test_bounded_risk=.1,
                            empirical_bounded_risk=empirical, confidence_allocation=confidence,
                            posterior_kl_float64=1.+index, channel_complexity=channel, sample_complexity=sample,
                            joint_complexity=joint, shift=dict(channel_shift_complexity=shift),
                            empirical_bounded_risk_upper=upper, theorem_1=bound(channel+sample), theorem_2=bound(joint),
                            certificate_draws=1000, full_test_draws=20, certificate_evaluation_seed=seed+3_000_007,
                            certificate_empirical_gibbs_mc=dict(bounded_loss=empirical, draws=1000, examples=512,
                                channel_source="bank", bounded_loss_definition=bounded_loss_description(config),
                                predictor=dict(stochastic=True, family="diagonal_gaussian"),
                                posterior_rule="one weight draw fixed across entire loader per replicate",
                                replicate_statistics=[dict(bounded_loss=empirical) for _ in range(1000)]),
                            evaluation=f"/remote/test/{task_id}/evaluation.json",
                            source_fingerprints={key: dict(sha256=char*64, bytes=1) for key, char in (
                                ("metadata", "a"), ("weights", "b"), ("evaluation", "c"), ("history", "d"))})
                        path = self.root / "rows" / f"{task_id}.json"
                        # A remote original path exercises portable local lookup.
                        row["certificate_report"] = f"/remote/certificate/rows/{task_id}.json"
                        path.write_text(json.dumps(row))
                        rows.append({**row, "certificate_report_fingerprint": fingerprint(path)})
        self.data = dict(schema_version=2, total_runs=48, delta=.025, draws=1000,
                         definitions=dict(scope="Synthetic fixtures only"), rows=rows)
        self.save()

    def save(self):
        self.source.write_text(json.dumps(self.data))

    def test_complete_figures_use_population_risk_rhs_and_all_formats(self):
        output = self.root / "figures"
        index = export(self.source, output)
        self.assertEqual(len(index["figures"]), 4)
        self.assertEqual(len(index["source_fingerprints"]), 49)
        signatures = {".png": b"\x89PNG", ".pdf": b"%PDF", ".eps": b"%!PS"}
        for figure in index["figures"]:
            self.assertEqual({Path(path).suffix for path in figure["files"]}, set(signatures))
            for path in figure["files"]:
                self.assertTrue(Path(path).read_bytes().startswith(signatures[Path(path).suffix]))
            for item in figure["data"]:
                if "components" in item:
                    self.assertAlmostEqual(sum(s["mean"] for s in item["components"].values()), item["population_risk_rhs"]["mean"])

    def test_missing_checkpoint_prevents_publication(self):
        self.data["rows"].pop()
        self.save()
        output = self.root / "figures"
        with self.assertRaisesRegex(ValueError, "48 distinct checkpoints"):
            export(self.source, output)
        self.assertFalse(output.exists())

    def test_gap_only_number_cannot_be_presented_as_risk_bound(self):
        self.data["rows"][0]["theorem_1"]["rhs_uncapped"] = .07
        self.save()
        with self.assertRaisesRegex(ValueError, "Population-risk RHS"):
            Evidence(self.source)

    def test_modified_certificate_source_is_rejected(self):
        path = self.root / "rows" / f"{self.data['rows'][0]['task_id']}.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            Evidence(self.source)

    def test_main_accuracy_model_cannot_be_mixed_into_demo(self):
        self.data["rows"][0]["config"]["width"] = 128
        self.save()
        with self.assertRaisesRegex(ValueError, "exclusively the IID moons"):
            Evidence(self.source)


if __name__ == "__main__":
    unittest.main()
