"""Exercise shared-banks seed selection, pool coverage and prospective test gating without training."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal.build_bayesian_campaign import base_config
from journal.build_campaign import make_entry
from journal import erasure_validation_seed_illustration as module
from journal.full_fading_comparison import METHODS, report_file, write_manifest


class SeedIllustrationTests(unittest.TestCase):
    def test_joint_fair_selection_and_ties(self):
        scores={(m,k,s):Fraction(1,2) for m in (200,500) for k in METHODS for s in module.SEEDS}
        self.assertEqual(module.rank_triples(scores,[200,500])[0]["seeds"],list(module.SEEDS[:3]))
        target=(module.SEEDS[2],module.SEEDS[4],module.SEEDS[6])
        for m in (200,500):
            for s in target:
                scores[m,"T1-Matched",s]+=Fraction(1,100)
        ranks=module.rank_triples(scores,[200,500])
        self.assertEqual(len(ranks),56)
        self.assertEqual(ranks[0]["seeds"],list(target))
        self.assertEqual(ranks[0]["minimum_validation_margin_pp"],1.)
        del scores[500,"L2-Perfect",module.SEEDS[-1]]
        with self.assertRaisesRegex(ValueError,"Missing"):
            module.rank_triples(scores,[200,500])

    def test_complete_mock_pipeline_without_training_or_test_access(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); sources=[]
            for m in (200,500):
                path=root/f"source{m}"/"manifest.jsonl"
                entries=[]
                for label,(method,perfect) in METHODS.items():
                    cfg={**base_config("moons","outage"),"m":m,"method":method,"data_seed":20261118,
                         "train_channel":{"kind":"perfect"} if perfect else module.ERASURE,
                         "deployment_channel":module.ERASURE}
                    entries.extend(make_entry(cfg,s,f"moons_p050_m{m}",label,"erasure_transfer","erasure_confirmation") for s in (1,2,3))
                write_manifest(path,entries)
                (path.parent/"protocol.json").write_text(json.dumps({"manifest":module.file_fingerprint(path)}))
                sources.append(path)
            out=root/"pool"
            manifest=module.prepare(sources,out)
            self.assertEqual(manifest["models"],112)
            with self.assertRaises(FileNotFoundError):
                module.evaluate(out,0)
            design,pool=module.load_pool(out)
            for e in pool:
                cp=out/"training"/e["task_id"];cp.mkdir(parents=True)
                (cp/"metadata.json").write_text("{}");(cp/"weights.pt").write_bytes(b"MOCK")
                cfg=e["config"]
                good=cfg["seed"] in (490103,490105,490107) and e["method_label"]=="T1-Matched"
                val=.60+(.01 if good else 0.)
                r=dict(**e,seed=cfg["seed"],status="complete",checkpoint=str(cp),validation_accuracy=val,
                       validation_cross_entropy=.5,seconds=1.)
                path=report_file(out/"training",e);path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(r))
            def done(cp,cfg):
                return {"validation_accuracy":.60+(.01 if cfg["seed"] in (490103,490105,490107) and cfg["method"]=="proposed" and cfg["train_channel"]==module.ERASURE else 0.)}
            with patch.object(module,"completed_trial",side_effect=done):
                selected=module.freeze(out)
            self.assertEqual(selected["seeds"],[490103,490105,490107])
            def synthetic_test(row,draws,full_risks,resume):
                self.assertEqual(draws,40)
                return {**row,"test_accuracy":.54 if row["method_label"]=="ERM-Matched" else .53,"test_cross_entropy":.7,"test_bounded_loss":.15}
            def check(e,r,cp):
                return {**r,"dataset":"moons","m":e["config"]["m"],"channel":"outage"}
            with patch.object(module,"evaluate_trial",side_effect=synthetic_test),patch.object(module,"validate_test",side_effect=check):
                for i in range(len(pool)):
                    module.evaluate(out,i)
                report=module.report(out)
            self.assertEqual(report["models"],112)
            with (out/"accuracy_by_seed.csv").open() as f:
                obs=list(csv.DictReader(f))
            self.assertEqual(len(obs),112)
            self.assertEqual(sum(r["selected"]=="True" for r in obs),42)
            self.assertTrue(all(g["highest_mean_methods"]==["ERM-Matched"] for g in json.loads((out/"seed_illustration.json").read_text())["groups"]["validation_selected_triple"]))
            for file,n in (("full_pool_cells.csv",8),("validation_selected_triple_cells.csv",3)):
                with (out/file).open() as f:
                    cells=list(csv.DictReader(f))
                self.assertEqual(len(cells),14)
                self.assertEqual({int(row["n_seeds"]) for row in cells},{n})


if __name__=="__main__":
    unittest.main()
