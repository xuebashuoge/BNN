"""C1 plotting checks with synthetic fixtures only, never scientific results."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from journal.c1_reference import erasure_first_shift, lognormal_logit_channel_terms, smooth_loss_lipschitz
from journal.plot_c1_bayesian import Evidence, ERASURE_GRID, LOGNORMAL_GRID, export, fingerprint, plt, risk_curves


class C1BayesianPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-c1-plot-test-")
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plt.close, "all")
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic_report.json"
        checkpoint = self.root / "synthetic_checkpoint.txt"
        checkpoint.write_text("Synthetic fixture only. No trained model or scientific result.\n")
        confidence = dict(total_failure=.05, runs=12, data_failure_total=.04, empirical_mc_failure_total=.01,
                          data_failure_per_run=.04/12, empirical_mc_failure_per_run=.01/12,
                          empirical_draws=1000, test_draws=100)
        radius = math.sqrt(math.log(1 / confidence["empirical_mc_failure_per_run"]) / 2000)
        rows = []
        for number, method in enumerate(("ERM", "L2", "T1", "T2")):
            for offset, seed in enumerate((76101, 76102, 76103)):
                empirical, complexity = .1 + number*.01 + offset*.001, .2
                baseline = empirical + radius + complexity
                row = dict(method_label=method+"-Perfect", seed=seed, empirical_bounded_risk_mc=empirical,
                           empirical_mc_radius=radius, data_complexity=complexity,
                           baseline_risk_upper_uncapped=baseline, erasure=[], lognormal=[])
                for severity in ERASURE_GRID:
                    shift = erasure_first_shift(8, severity)
                    row["erasure"].append(dict(severity=severity, test_accuracy=.9-severity*.2,
                        test_bounded_risk=.11+severity*.1, shift_first_exact=shift,
                        risk_upper_uncapped=baseline+shift))
                for severity in LOGNORMAL_GRID:
                    terms = lognormal_logit_channel_terms(severity)
                    row["lognormal"].append(dict(severity=severity, test_accuracy=.9-severity*.05,
                        test_bounded_risk=.11+severity*.04, shift_first_mc_diagnostic=severity*.38,
                        shift_first_mc_se=severity*.001, shift_second_closed=terms["c1_second_shift_upper"],
                        log_reference_moment=terms["log_exponential_moment"], reference_kl=terms["channel_kl"],
                        jensen_shift_upper=terms["jensen_shift_upper"],
                        risk_upper_uncapped=baseline+terms["c1_second_shift_upper"]))
                rows.append(row)
        self.data = dict(schema_version=1, confidence=confidence,
            analytic=dict(epsilon=.01, tau=.1, K=smooth_loss_lipschitz(), benchmark={"status": "synthetic fixture"}),
            reports=rows, source_hashes={str(checkpoint): fingerprint(checkpoint)})
        self.save()

    def save(self):
        self.source.write_text(json.dumps(self.data))

    def test_complete_family_exports_png_pdf_eps_and_all_seed_data(self):
        index = export(self.source, self.root / "figures")
        self.assertEqual(len(index["figures"]), 3)
        signatures = {".png": b"\x89PNG", ".pdf": b"%PDF", ".eps": b"%!PS"}
        for figure in index["figures"]:
            self.assertEqual({Path(path).suffix for path in figure["files"]}, set(signatures))
            for path in figure["files"]:
                self.assertTrue(Path(path).read_bytes().startswith(signatures[Path(path).suffix]))
        for figure in index["figures"][1:]:
            self.assertEqual({cell["method_label"] for cell in figure["data"]}, {m+"-Perfect" for m in ("ERM", "L2", "T1", "T2")})
            for cell in figure["data"]:
                self.assertEqual(cell["seeds"], [76101, 76102, 76103])
                self.assertTrue(all(point["statistics"]["risk_upper_uncapped"]["n"] == 3 for point in cell["points"]))

    def test_missing_or_duplicated_posterior_prevents_export(self):
        self.data["reports"].pop()
        self.save()
        output = self.root / "figures"
        with self.assertRaisesRegex(ValueError, "all 12"):
            export(self.source, output)
        self.assertFalse(output.exists())
        self.data["reports"].append(copy.deepcopy(self.data["reports"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "distinct"):
            Evidence(self.source)

    def test_diagnostic_mc_never_changes_analytic_risk_curves(self):
        before = Evidence(self.source)
        for row in self.data["reports"]:
            for case in row["lognormal"]:
                case["shift_first_mc_diagnostic"] *= 3
                case["shift_first_mc_se"] *= 3
        self.save()
        after = Evidence(self.source)
        data = []
        with mock.patch("journal.plot_c1_bayesian.finish", side_effect=lambda fig, out, name, cells, index: data.append(cells)):
            risk_curves(before, self.root, {}, "lognormal")
            risk_curves(after, self.root, {}, "lognormal")
        self.assertEqual(data[0], data[1])

    def test_gap_only_or_mc_derived_number_cannot_be_published_as_risk_bound(self):
        row = self.data["reports"][0]
        row["lognormal"][1]["risk_upper_uncapped"] = row["lognormal"][1]["shift_first_mc_diagnostic"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Population-risk RHS"):
            Evidence(self.source)

    def test_incorrect_reference_formula_or_family_confidence_rejected(self):
        self.data["reports"][0]["lognormal"][2]["reference_kl"] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, "exact fixed-Gaussian"):
            Evidence(self.source)
        self.data["reports"][0]["lognormal"][2]["reference_kl"] /= 2
        self.data["confidence"]["data_failure_per_run"] = .04
        self.save()
        with self.assertRaisesRegex(ValueError, "all 12"):
            Evidence(self.source)

    def test_source_tampering_is_detected_and_risk_panels_have_headroom(self):
        evidence = Evidence(self.source)
        figures = []
        with mock.patch("journal.plot_c1_bayesian.finish", side_effect=lambda fig, *args: figures.append(fig)):
            risk_curves(evidence, self.root, {}, "erasure")
            risk_curves(evidence, self.root, {}, "lognormal")
        for fig in figures:
            for ax in fig.axes:
                highest = max(float(value) for line in ax.lines for value in line.get_ydata())
                self.assertGreaterEqual(ax.get_ylim()[1]+1e-12, 1.08*max(1., highest))
        Path(next(iter(self.data["source_hashes"]))).write_text("Modified synthetic source\n")
        with self.assertRaisesRegex(ValueError, "Source fingerprint differs"):
            Evidence(self.source)


if __name__ == "__main__":
    unittest.main()
