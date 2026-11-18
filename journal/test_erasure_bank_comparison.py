"""Bank-size rescaling and collapse of equivalent perfect-training objectives."""
import json
from pathlib import Path
import tempfile
import unittest

from journal.build_bayesian_campaign import base_config
from journal.build_campaign import make_entry
from journal.erasure_bank_comparison import build, ERASURE


class ErasureBankTests(unittest.TestCase):
    def test_source_strengths_and_paired_seeds_survive_bank_size_change(self):
        source=[]
        for seed in range(5):
            for obj,method in [('ERM','erm'),('L2','l2'),('T1','proposed'),('T2','joint')]:
                for regime in ('Perfect','Matched'):
                    cfg=base_config('moons','outage')
                    cfg.update(m=20,method=method,eta_data=.01,eta_channel=.003,
                               train_channel=ERASURE if regime=='Matched' else {'kind':'perfect'})
                    e=make_entry(cfg,seed,'moons_outage',obj+'-'+regime,'legacy')
                    source.append(dict(e,selected_from_validation_variant=e['variant_id']))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            path=out/'old.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in source))
            build(path,out/'m500','moons',500)
            items=[json.loads(s) for s in (out/'m500/manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(items),35)
            for e in items:
                c=e['config']
                self.assertEqual(c['m'],500)
                self.assertEqual(c['deployment_channel'],ERASURE)
                self.assertEqual(c['data_seed'],20261118)
                if e['method_label']=='T1-Matched':
                    self.assertAlmostEqual(c['eta_data'],.01*(19/499)**.5)
                elif e['method_label']=='T2-Matched':
                    n=c['train_size']
                    self.assertAlmostEqual(c['eta_data'],.01*((n-1+19)/(n-1+499))**.5)
                else:
                    self.assertEqual(c['eta_data'],.01)
            for e in source:
                if e['method_label']=='T2-Perfect':
                    e['config']['posterior_std'] *= 2
            path.write_text(''.join(json.dumps(r)+'\n' for r in source))
            with self.assertRaisesRegex(ValueError,'different configurations'):
                build(path,out/'bad','moons',200)
