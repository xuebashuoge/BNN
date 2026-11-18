"""Compare the seven study methods at independent coordinate erasure probability 0.50."""
import argparse
import json
import math
from pathlib import Path

from .build_campaign import make_entry
from .diagonal_fading_comparison import run_task
from .fading_severity import report
from .full_fading_comparison import METHODS, write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest

ERASURE = {'kind':'outage','outage_prob':.50}
DATA_SEED = 20261118


def build(source, output, ds, m):
    if m not in (100,200,500,1000,2000) or (m==100 and ds!='cifar10'):
        raise ValueError('Unsupported bank-size and dataset combination')
    anchors = [r for r in read_manifest(source) if r['group']==f'{ds}_outage']
    by_label = {(r['method_label'],r['config']['seed']):r for r in anchors}
    seeds = sorted({s for _,s in by_label})
    expected_labels = (set(METHODS)-{'Proposed-Perfect'})|{'T1-Perfect','T2-Perfect'}
    if (len(anchors)!=len(by_label) or len(seeds)!=5 or
            set(by_label)!={(label,s) for label in expected_labels for s in seeds}):
        raise ValueError('Expected a complete five-seed historical erasure confirmation')
    entries=[]
    for seed in seeds:
        p=by_label['T1-Perfect',seed]['config']
        j=by_label['T2-Perfect',seed]['config']
        if p!={**j,'method':'proposed'}:
            raise ValueError('The historical perfect-training objectives have different configurations')
        for label in METHODS:
            old=by_label['T1-Perfect' if label=='Proposed-Perfect' else label,seed]
            c=old['config']; old_m=c['m']; n=c['train_size']
            if (old_m!=20 or c['deployment_channel']!=ERASURE or c['dataset']!=ds or
                    c['train_channel']!=(ERASURE if label.endswith('-Matched') else {'kind':'perfect'})):
                raise ValueError('Invalid erasure confirmation anchor')
            config={**c,'m':m,'data_seed':DATA_SEED,'eval_draws':20}
            if label=='T1-Matched':
                config['eta_data'] *= math.sqrt((old_m-1)/(m-1))
            elif label=='T2-Matched':
                config['eta_data'] *= math.sqrt((1+(old_m-1)/(n-1))/(1+(m-1)/(n-1)))
            item=make_entry(config,seed,f'{ds}_erasure_p050_m{m}',label,'erasure_transfer','erasure_confirmation')
            item.update(source_task_id=old['task_id'],source_variant=old['selected_from_validation_variant'])
            entries.append(item)
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'manifest.jsonl'
    result=write_manifest(manifest,entries)
    protocol=dict(dataset=ds,m=m,source=str(source.resolve()),source_manifest=file_fingerprint(source),
                  source_bank_size=20,manifest=file_fingerprint(manifest),law=ERASURE,seeds=seeds,
                  remapping='Dimensionless matched penalty strength held fixed under the original n,m scaling',
                  data_seed=DATA_SEED,eval_draws=20,
                  perfect_training='T1 and T2 have the same objective and frozen hyperparameters on the perfect link')
    write_once(output/'protocol.json',json.dumps(protocol,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    builder=sub.add_parser('build')
    builder.add_argument('--source',type=Path,required=True)
    builder.add_argument('--output-dir',type=Path,required=True)
    builder.add_argument('--dataset',choices=['moons','mnist','fashionmnist','cifar10'],required=True)
    builder.add_argument('--m',type=int,required=True)
    task=sub.add_parser('task')
    task.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    task.add_argument('--threads',type=int,default=1)
    task.add_argument('--index',type=int)
    for parser in (task,sub.add_parser('report')):
        parser.add_argument('--manifest',type=Path,required=True)
        parser.add_argument('--output-root',type=Path,required=True)
    a=ap.parse_args()
    if a.command=='build':
        out=build(a.source,a.output_dir,a.dataset,a.m)
    elif a.command=='task':
        out=run_task(a)
    else:
        out=report(a.manifest,a.output_root,ERASURE,'erasure_confirmation')
    print(json.dumps(out))
