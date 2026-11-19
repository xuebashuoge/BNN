"""Build journal table inputs from full, paired confirmations of fading and p=.50 erasure."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from .erasure_bank_comparison import ERASURE
from .fading_severity import HARD
from .full_fading_comparison import METHODS
from .run_experiment import file_fingerprint

DATASETS=('moons','mnist','fashionmnist','cifar10')
BANKS=(100,200,500,1000,2000)
MAIN=(200,500,1000)
TARGET={'T1-Matched','T2-Matched'}


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_order(r):
    # Each follow-up consumes the validation-selected configurations from the preceding stage.
    # The final local grid may have fewer candidates after shortlisting.
    order={'transfer':0, 'retuned':1, 'common/retuned':2, 'expansion/retuned':2, 'scale-grid':2,
           'epochs60':3, 'penalty60/retuned':4, 'prior_width_reference/confirmation':5}
    if r['phase'] not in order:
        raise ValueError(f"Unregistered confirmation phase {r['phase']}")
    return (order[r['phase']],r['candidates_per_label'],r['n_seeds'],r['phase'])


def read_record(path, folder, channel, dataset, m, seed_rows):
    summary=read_json(path)
    case=path.parent.parent
    phase='transfer' if case==folder else str(case.relative_to(folder))
    manifest=case/'manifest.jsonl'
    if summary['manifest']!=file_fingerprint(manifest):
        raise ValueError(f'Stale confirmation manifest at {path}')
    entries=[json.loads(line) for line in manifest.read_text().splitlines()]
    law=HARD if channel=='Fading' else ERASURE
    if summary['channel']!=law or len(summary['groups'])!=1:
        raise ValueError(f'Mixed conditions at {path}')
    group=summary['groups'][0]
    if group['dataset']!=dataset or group['m']!=m or set(group['methods'])!=set(METHODS):
        raise ValueError(f'Unexpected group at {path}')
    seeds=summary['confirmation_seeds']
    rows=read_csv(path.parent/'accuracy_by_seed.csv')
    expected={(k,s) for k in METHODS for s in seeds}
    if (len(rows)!=len(expected) or summary['trials']!=len(expected) or
            {(r['method_label'],int(r['seed'])) for r in rows}!=expected or
            {r['task_id'] for r in rows}!={e['task_id'] for e in entries}):
        raise ValueError(f'Incomplete paired seed grid at {path}')
    for e in entries:
        c=e['config']
        if (c['deployment_channel']!=law or c['train_channel']!=(law if e['method_label'].endswith('-Matched') else {'kind':'perfect'})
                or c['m']!=m or c['dataset']!=dataset or c['data_seed']!=20261118):
            raise ValueError(f'Trial changed its training or deployment conditions: {e["task_id"]}')
    scores=defaultdict(dict)
    tag=dict(dataset=dataset,channel=channel,m=m,phase=phase)
    for r in rows:
        k=r['method_label']; seed=int(r['seed']); value=100*float(r['test_accuracy'])
        if not 0<=value<=100: raise ValueError(f'Invalid accuracy in {path}')
        scores[k][seed]=value
        seed_rows.append(dict(**tag,method=k,seed=seed,accuracy_pct=value,
                              validation_pct=100*float(r['validation_accuracy']),variant=r['variant_id'],task_id=r['task_id']))
    means={k:fmean(v.values()) for k,v in scores.items()}
    compare=lambda k: means[k]
    proposed=max((k for k in METHODS if k in TARGET), key=compare)
    other=max((k for k in METHODS if k not in TARGET), key=compare)
    gaps=[scores[proposed][s]-scores[other][s] for s in seeds]
    budget=0
    selection_path=None
    for p in (case/'manifest.selection.json',case/'selection.json'):
        if p.is_file():
            selection_path=p
            selection=read_json(p)
            trials=selection.get('validation_trials',[])
            if isinstance(trials,list):
                budget=min((sum(v.get('method_label')==k for v in trials) for k in METHODS),default=0)
            break
    winners=[k for k in METHODS if max(means.values())-means[k]<1e-10]
    if set(winners)!=set(group['highest_mean_methods']): raise ValueError(f'Rank disagrees with seed rows at {path}')
    epochs={e['config']['epochs'] for e in entries}
    draws={e['config']['eval_draws'] for e in entries}
    if len(epochs)!=1 or len(draws)!=1:
        raise ValueError(f'Mixed training or evaluation budgets within {path}')
    record=dict(**tag,seeds=' '.join(map(str,seeds)),n_seeds=len(seeds),epochs=epochs.pop(),
                eval_draws=draws.pop(),candidates_per_label=budget,
                top_labels='/'.join(winners),best_proposed=proposed,proposed_pct=means[proposed],
                best_other=other,other_pct=means[other],margin_pp=fmean(gaps),paired_seed_sd_pp=stdev(gaps),
                paired_wins=sum(d>1e-10 for d in gaps),leads=set(winners)<=TARGET,
                source=str(path.resolve()),source_sha256=file_fingerprint(path)['sha256'],
                selection_sha256=file_fingerprint(selection_path)['sha256'] if selection_path else '')
    cells=[]
    for k in METHODS:
        sd=stdev(scores[k].values())
        stat=group['methods'][k]['test_accuracy']
        if not (math.isclose(100*stat['mean'],means[k],rel_tol=0,abs_tol=1e-9) and
                math.isclose(100*stat['sd'],sd,rel_tol=0,abs_tol=1e-9)):
            raise ValueError(f'Mean or SD differs from paired scores at {path}')
        cells.append(dict(**tag, method=k, mean_pct=means[k], sd_pp=sd,
                          n_seeds=len(seeds), seeds=record['seeds'], source=record['source'],
                          rho_c=.85 if channel=='Fading' else '',noise_std=2. if channel=='Fading' else '',
                          outage_prob=.50 if channel=='Erasure' else ''))
    return record,cells


def select_series(case_rows, channel, dataset, mode):
    by_m={m:{(r['phase'],r['seeds']):r for r in case_rows if
             (r['channel'],r['dataset'],r['m'])==(channel,dataset,m)} for m in BANKS}
    common=set.intersection(*(set(by_m[m]) for m in MAIN))
    if not common:
        raise ValueError(f'No common confirmation for {channel}, {dataset}')
    if mode=='common':
        shared=max(common,key=lambda k:case_order(by_m[200][k]))
    else:
        winning={k for k in common if all(by_m[m][k]['leads'] for m in MAIN)}
        shared=min(winning,key=lambda k:case_order(by_m[200][k])) if winning else None
    chosen=[]
    for m in BANKS:
        pool=list(by_m[m].values())
        if not pool and m==100 and dataset!='cifar10': continue
        if m in MAIN and shared is not None:
            r=by_m[m][shared]
        elif mode=='common':
            r=max(pool,key=case_order)
        else:
            r=min((r for r in pool if r['leads']),key=case_order,default=min(pool,key=case_order))
        chosen.append(r)
    return chosen


def mild_cells(fading, dataset, m):
    base=fading/'matched'/f'm{m}'/dataset
    origin=read_json(base/'transfer.json')
    source=Path(origin['source'])
    path=source.parent/source.stem/'accuracy_summary.json'
    if file_fingerprint(path)!=origin['source_summary'] or file_fingerprint(source)!=origin['source_manifest']:
        raise ValueError(f'Changed original fading source at {source}')
    summary=read_json(path)
    group=summary['groups'][0]
    rows=read_csv(path.parent/'accuracy_by_seed.csv')
    result=[]
    for method in METHODS:
        values=[100*float(r['test_accuracy']) for r in rows if r['method_label']==method]
        stats=group['methods'][method]['test_accuracy']
        mean,sd=fmean(values),stdev(values)
        if (len(values)!=len(summary['confirmation_seeds']) or
                not math.isclose(mean,100*stats['mean'],rel_tol=0,abs_tol=1e-9) or
                not math.isclose(sd,100*stats['sd'],rel_tol=0,abs_tol=1e-9)):
            raise ValueError(f'Mild source mismatch in {path}')
        result.append(dict(dataset=dataset,channel='Fading',m=m,phase='original_'+source.stem,method=method,
                           mean_pct=mean,sd_pp=sd,n_seeds=len(values),
                           seeds=' '.join(map(str,summary['confirmation_seeds'])),source=str(path),
                           rho_c=.35,noise_std=1.,outage_prob=''))
    return result


def assemble(fading, erasure, output):
    output.mkdir(exist_ok=True,parents=True)
    records, cells, seed_rows=[],{},[]
    for channel,parent in [('Fading',fading/'matched'),('Erasure',erasure/'p050')]:
        for m in BANKS:
            for ds in DATASETS:
                if m==100 and ds!='cifar10': continue
                folder=parent/f'm{m}'/ds
                paths=sorted(folder.glob('**/runs/accuracy_summary.json'))
                if folder/'runs/accuracy_summary.json' not in paths:
                    raise ValueError(f'Missing primary cohort for {folder}')
                for path in paths:
                    rec,means=read_record(path,folder,channel,ds,m,seed_rows)
                    records.append(rec)
                    cells[channel,ds,m,rec['phase']]=means
    tables={}
    tables['initial']=[r for r in records if r['phase']=='transfer']
    for mode in ['common','illustrative']:
        tables[mode]=[r for channel in ('Fading','Erasure') for ds in DATASETS
                      for r in select_series(records,channel,ds,mode)]
    for mode, rows in tables.items():
        if len(rows)!=34: raise ValueError('Expected 17 settings for each channel')
        ordered=sorted(rows,key=lambda r:(BANKS.index(r['m']),DATASETS.index(r['dataset']),r['channel']!='Fading'))
        write_csv(output/f'{mode}_cohorts.csv',ordered)
        write_csv(output/f'{mode}_cells.csv',[item for r in ordered
                  for item in cells[r['channel'],r['dataset'],r['m'],r['phase']]])
    original=[]
    for row in tables['common']:
        ds,m,ch=row['dataset'],row['m'],row['channel']
        original.extend(mild_cells(fading,ds,m) if ch=='Fading' else cells[ch,ds,m,row['phase']])
    write_csv(output/'milder_fading_cells.csv',original)
    with (output/'selected_model_configs.jsonl').open('w') as f:
        for r in records:
            manifest=Path(r['source']).parent.parent/'manifest.jsonl'
            seen=set()
            for item in map(json.loads,manifest.read_text().splitlines()):
                if item['method_label'] in seen:
                    continue
                seen.add(item['method_label'])
                value=dict(dataset=r['dataset'],m=r['m'],channel=r['channel'],phase=r['phase'],
                           method=item['method_label'],variant_id=item['variant_id'],config=item['config'],
                           confirmation_seeds=r['seeds'],manifest=str(manifest))
                f.write(json.dumps(value,sort_keys=True)+'\n')
    write_csv(output/'all_confirmation_rounds.csv',records)
    write_csv(output/'all_method_means.csv',[item for rows in cells.values() for item in rows])
    write_csv(output/'all_seed_observations.csv',seed_rows)
    print(f'Verified {len(records)} confirmation cohorts and {len(seed_rows)} model/seed accuracies')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fading',type=Path,required=True)
    parser.add_argument('--erasure',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assemble(args.fading.resolve(),args.erasure.resolve(),args.output.resolve())
