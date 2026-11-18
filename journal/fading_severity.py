"""Compare the same training configurations and seeds under a harsher diagonal link."""
import argparse
import json
from pathlib import Path

from .build_campaign import make_entry
from .campaign import save_json
from .diagonal_fading_comparison import run_task, ranking
from .full_fading_comparison import METHODS, group_results, read_results, validate_test, write_csv, write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest

BASE = dict(kind='diagonal_fading', rho_c=.35)
HARD = dict(kind='diagonal_fading', rho_c=.85, noise_std=2.)


def transfer(source, output):
    original = read_manifest(source)
    summary_path = source.parent/source.stem/'accuracy_summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['manifest'] != file_fingerprint(source) or len(summary['groups']) != 1:
        raise ValueError(f'Source confirmation differs from its completed summary at {source}')
    entries = []
    for row in original:
        c = row['config']
        if c['deployment_channel'] != BASE or c['train_channel'] not in (BASE, {'kind':'perfect'}):
            raise ValueError('Expected a mild matched/perfect source confirmation')
        training = HARD if c['train_channel']==BASE else c['train_channel']
        config = {**c,'train_channel':training,'deployment_channel':HARD}
        group = f"{c['dataset']}_diagonal_fading_r0p85_noise2_m{c['m']}"
        item = make_entry(config,c['seed'],group,row['method_label'],'severity_transfer','severity_confirmation')
        if item['config'] != config:
            raise ValueError('Original seed schedule cannot be preserved')
        item.update(family=row['family'], source_task_id=row['task_id'], source_variant=row['variant_id'])
        entries.append(item)
    expected = {(method, s) for method in METHODS for s in summary['confirmation_seeds']}
    pairs = [(r['method_label'],r['config']['seed']) for r in entries]
    if len(pairs)!=len(expected) or set(pairs)!=expected:
        raise ValueError('Expected one complete seven-method source confirmation')
    output.mkdir(parents=True,exist_ok=True)
    path = output/'manifest.jsonl'
    result = write_manifest(path,entries)
    protocol = dict(source=str(source.resolve()), source_manifest=file_fingerprint(source),
                    source_summary=file_fingerprint(summary_path), manifest=file_fingerprint(path),
                    seeds=summary['confirmation_seeds'], train_law_matched=HARD, deployment_law=HARD,
                    original_law=BASE, selection='Frozen original validation-selected hyperparameters, retrained from initialization')
    write_once(output/'transfer.json',json.dumps(protocol,indent=2)+'\n')
    return result


def report(manifest, root, law=HARD, role="severity_confirmation"):
    entries = read_manifest(manifest)
    if any(r.get('role')!=role or r['config']['deployment_channel']!=law
           or r['config']['train_channel']!=(law if r['method_label'].endswith('-Matched') else {'kind':'perfect'})
           for r in entries):
        raise ValueError('Expected a confirmation at the declared matched/perfect laws')
    rows = [validate_test(item,row,root) for item,row in zip(entries,read_results(root,entries),strict=True)]
    seeds=sorted({r['seed'] for r in rows})
    groups=group_results(rows,seeds)
    for g in groups:
        g.update({k:v for k,v in law.items() if k!='kind'})
        g.update(mean_ranking=ranking({k:v['test_accuracy']['mean'] for k,v in g['methods'].items()}))
    means=[dict(dataset=g['dataset'],m=g['m'],method=k,mean_accuracy=v['test_accuracy']['mean'],
                sd_accuracy=v['test_accuracy']['sd']) for g in groups for k,v in g['methods'].items()]
    write_csv(root/'accuracy_means.csv',means,list(means[0]))
    write_csv(root/'accuracy_by_seed.csv',rows,['dataset','channel','m','method_label','seed','variant_id','validation_accuracy',
                                                'test_accuracy','test_cross_entropy','test_bounded_loss','task_id'])
    result=dict(stage=entries[0]['stage'], manifest=file_fingerprint(manifest),trials=len(rows),confirmation_seeds=seeds,
                channel=law,groups=groups)
    save_json(root/'accuracy_summary.json',result)
    return {'trials':len(rows),'winners':[g['highest_mean_methods'] for g in groups]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('transfer')
    init.add_argument('--source',type=Path,required=True)
    init.add_argument('--output-dir',type=Path,required=True)
    task=sub.add_parser('task')
    task.add_argument('--threads',type=int,default=1)
    task.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    task.add_argument('--index',type=int)
    for cmd in (task,sub.add_parser('report')):
        cmd.add_argument('--manifest',type=Path,required=True)
        cmd.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='transfer':
        result=transfer(args.source,args.output_dir)
    elif args.command=='report':
        result=report(args.manifest,args.output_root)
    else:
        result=run_task(args)
    print(json.dumps(result))
