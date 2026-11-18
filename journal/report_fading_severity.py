"""Assemble paired degradation, matched-law recovery and seven-method comparisons."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
from statistics import fmean, stdev

from .fading_severity import BASE, HARD
from .full_fading_comparison import METHODS
from .run_experiment import file_fingerprint

DATASETS = ('mnist', 'fashionmnist', 'cifar10', 'moons')
NAMES = dict(zip(DATASETS, ('MNIST','Fashion-MNIST','CIFAR-10','Two moons')))
SIZES = (100,200,500,1000,2000)
MATCHED = [k for k in METHODS if k.endswith('-Matched')]
PROPOSED = ['T1-Matched','T2-Matched']
LAWS = [(.35,1.), (.85,1.), (.35,2.), (.85,2.)]


def csv_read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_json(path):
    return json.loads(path.read_text())


def check_close(a, b):
    if not math.isclose(a,b,abs_tol=1e-8,rel_tol=0):
        raise ValueError(f'Numeric mismatch: {a} vs {b}')


def cells():
    return [(ds,m) for m in SIZES for ds in DATASETS if m != 100 or ds == 'cifar10']


def frozen_inputs(root):
    reference = {}
    for r in csv_read(root/'stress/accuracy_by_seed.csv'):
        case = (r['dataset'],int(r['m']),r['method'],int(r['seed']))
        law = (float(r['rho_c']),float(r['noise_std']))
        if (case,law) in reference or law not in LAWS:
            raise ValueError('Unexpected or duplicate frozen deployment observation')
        reference[case,law] = float(r['accuracy_pct'])
    recovery = {}
    for r in csv_read(root/'stress/recovery/recovery_by_seed.csv'):
        case = (r['dataset'],int(r['m']),r['method'],int(r['seed']))
        if case in recovery:
            raise ValueError(f'Duplicate recovery observation for {case}')
        check_close(float(r['frozen_severe_pct']),reference[case,LAWS[-1]])
        recovery[case] = float(r['hard_severe_pct'])
        check_close(float(r['recovery_pp']),recovery[case]-reference[case,LAWS[-1]])
    if len(reference)!=4*len(recovery) or set(k[0] for k in reference)!=set(recovery):
        raise ValueError('The four frozen evaluations and recovery evaluation must cover the same posteriors')
    return reference, recovery


def channel_table(root, output):
    reference, recovery = frozen_inputs(root)
    seed_rows, means = [], []
    for ds, m in cells():
        seed_set=sorted({seed for (d,b,_,seed) in recovery if (d,b)==(ds,m)})
        if not seed_set:
            raise ValueError(f'No effects for {ds} m={m}')
        paired = []
        for seed in seed_set:
            cases = [(ds,m,method,seed) for method in MATCHED]
            baseline, corr, noise, combined = [fmean(reference[c,law] for c in cases) for law in LAWS]
            hard = fmean(recovery[c] for c in cases)
            metrics = dict(original_pct=baseline, correlation_only_pct=corr, noise_only_pct=noise,
                           combined_frozen_pct=combined, retrained_hard_pct=hard,
                           correlation_change_pp=corr-baseline, noise_change_pp=noise-baseline,
                           combined_change_pp=combined-baseline, interaction_pp=combined-corr-noise+baseline,
                           recovery_pp=hard-combined, retrained_vs_original_pp=hard-baseline)
            paired.append(metrics)
            seed_rows.append(dict(dataset=ds,m=m,seed=seed,**metrics))
        means.append(dict(dataset=ds,m=m,n_seeds=len(seed_set),
                          **{k:fmean(r[k] for r in paired) for k in paired[0]},
                          combined_paired_sd_pp=stdev(r['combined_change_pp'] for r in paired),
                          recovery_paired_sd_pp=stdev(r['recovery_pp'] for r in paired)))
    csv_write(output/'channel_effects.csv',means)
    csv_write(output/'channel_effects_by_seed.csv',seed_rows)
    return means, len(recovery)


def algorithm_tables(root, output):
    cohorts, means, seeds, sources, changes = [], [], [], [], []
    for ds,m in cells():
        base = root/'matched'/f'm{m}'/ds
        origin = load_json(base/'transfer.json')
        mild_manifest = Path(origin['source'])
        mild_root = mild_manifest.parent/mild_manifest.stem
        if file_fingerprint(mild_manifest)!=origin['source_manifest'] or file_fingerprint(mild_root/'accuracy_summary.json')!=origin['source_summary']:
            raise ValueError(f'Original source changed for {ds} at m={m}')
        mild = {r['task_id']:r for r in csv_read(mild_root/'accuracy_by_seed.csv')}
        mild_tasks = {r['task_id']:r for r in map(json.loads,mild_manifest.read_text().splitlines())}
        candidates = sorted(base.glob('**/runs/accuracy_summary.json'))
        if base/'runs/accuracy_summary.json' not in candidates:
            raise ValueError(f'Missing transferred confirmation at {base}')
        for path in candidates:
            parent = path.parent
            case = parent.parent
            phase = 'transfer' if case==base else str(case.relative_to(base))
            manifest = case/'manifest.jsonl'
            summary = load_json(path)
            if summary['channel']!=HARD or summary['manifest']!=file_fingerprint(manifest):
                raise ValueError(f'Unexpected law or manifest at {path}')
            ids = {r['task_id']:r for r in map(json.loads,manifest.read_text().splitlines())}
            samples = csv_read(parent/'accuracy_by_seed.csv')
            seed_set = summary['confirmation_seeds']
            expected = {(method,s) for method in METHODS for s in seed_set}
            if (len(samples)!=len(expected) or {(r['method_label'],int(r['seed'])) for r in samples}!=expected
                    or {r['task_id'] for r in samples}!=set(ids) or len(summary['groups'])!=1):
                raise ValueError(f'Incomplete confirmation at {path}')
            scores = defaultdict(dict)
            for r in samples:
                method,seed = r['method_label'],int(r['seed'])
                item = ids[r['task_id']]
                config = item['config']
                if (config['train_channel']!=(HARD if method in MATCHED else {'kind':'perfect'})
                        or config['deployment_channel']!=HARD or config['dataset']!=ds or config['m']!=m
                        or config['seed']!=seed or item['method_label']!=method):
                    raise ValueError(f'Invalid confirmation row for {item["task_id"]}')
                score = 100*float(r['test_accuracy'])
                scores[method][seed] = score
                seeds.append(dict(dataset=ds,m=m,phase=phase,method=method,seed=seed,accuracy_pct=score,
                                  variant_id=r['variant_id'],task_id=r['task_id']))
                if phase=='transfer':
                    src=item['source_task_id']
                    orig=mild_tasks[src]['config']
                    if (orig['deployment_channel']!=BASE or
                        config!={**orig,'train_channel':config['train_channel'],'deployment_channel':HARD}
                        or mild[src]['method_label']!=method or int(mild[src]['seed'])!=seed):
                        raise ValueError(f'Unpaired transfer {src}')
                    mild_pct=100*float(mild[src]['test_accuracy'])
                    changes.append(dict(dataset=ds,m=m,method=method,seed=seed,original_native_pct=mild_pct,
                                        hard_native_pct=score,change_pp=score-mild_pct))
            avg = {k:fmean(v.values()) for k,v in scores.items()}
            for k,v in avg.items():
                stat=summary['groups'][0]['methods'][k]['test_accuracy']
                check_close(v,100*stat['mean'])
                means.append(dict(dataset=ds,m=m,phase=phase,method=k,mean_pct=v,
                                  sd_pp=stdev(scores[k].values()),n_seeds=len(seed_set)))
            a=max(PROPOSED,key=avg.get)
            b=max((k for k in avg if k not in PROPOSED),key=avg.get)
            diffs=[scores[a][s]-scores[b][s] for s in seed_set]
            cohorts.append(dict(dataset=ds,m=m,phase=phase,seeds=' '.join(map(str,seed_set)),n_seeds=len(seed_set),
                                best_proposed=a,proposed_pct=avg[a],best_other=b,other_pct=avg[b],
                                winner=' / '.join(k for k in METHODS if max(avg.values())-avg[k]<1e-10),
                                margin_pp=fmean(diffs),paired_sd_pp=stdev(diffs),paired_wins=sum(d>1e-10 for d in diffs)))
            sources.append(dict(dataset=ds,m=m,phase=phase,source=str(path),**file_fingerprint(path),
                                manifest=summary['manifest'],mild_source=str(mild_manifest)))
    for name, rows in [('method_comparisons',cohorts),('method_means',means),('accuracy_by_seed',seeds),
                       ('paired_retraining_vs_original',changes)]:
        csv_write(output/(name+'.csv'),rows)
    (output/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    return cohorts, means


def figures(output, effects, cohorts, means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series=[('original_pct','Original law','#3C5260','o'),('correlation_only_pct','Higher correlation','#917006','s'),
            ('noise_only_pct','Higher noise','#A3442B','^'),('combined_frozen_pct','Both changes','#814996','v'),
            ('retrained_hard_pct','Retrained, both changes','#187254','D')]
    fig,axes=plt.subplots(2,2,figsize=(8.1,5.8))
    fig.subplots_adjust(left=.10,right=.985,top=.90,bottom=.21,hspace=.43,wspace=.25)
    for ax,ds in zip(axes.flat,DATASETS):
        group=sorted((r for r in effects if r['dataset']==ds), key=lambda r:r['m'])
        xs=[r['m'] for r in group]
        for col,label,color,marker in series:
            ax.plot(xs,[r[col] for r in group],label=label,c=color,marker=marker,ms=4,lw=1.3,
                    linestyle='--' if col=='retrained_hard_pct' else '-')
        ax.set_xscale('log');ax.set_xticks(xs,[str(x) for x in xs]);ax.minorticks_off()
        ax.set_title(NAMES[ds],fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',lw=.45,alpha=.4)
    fig.supylabel('Deployment accuracy (%)',x=.01,fontsize=10)
    fig.supxlabel('Training channel-bank size',y=.12,fontsize=10)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncols=3,loc='lower center',bbox_to_anchor=(.5,.002),frameon=False,fontsize=8)
    fig.suptitle('Channel severity and matched retraining',y=.985,fontsize=13)
    for ext in ('png','pdf'):fig.savefig(output/f'channel_severity.{ext}',dpi=240)
    plt.close(fig)

    for ds in DATASETS:
        cases=[r for r in cohorts if r['dataset']==ds]
        labels=list(METHODS)
        fig,ax=plt.subplots(figsize=(7.4,3.65))
        baseline=sorted((r for r in cases if r['phase']=='transfer'),key=lambda r:r['m'])
        points={r['m']:i for i,r in enumerate(baseline)}
        for k,marker,color in zip(labels,['o','s','^','D','o','s','D'],
                                  ['#676B74','#975527','#175C88','#107A64','#676B74','#975527','#175C88']):
            vals=sorted((r for r in means if r['dataset']==ds and r['phase']=='transfer' and r['method']==k),
                        key=lambda r:r['m'])
            perfect=k not in MATCHED
            ax.plot([points[r['m']] for r in vals],[r['mean_pct'] for r in vals],marker=marker,c=color,
                    ls='--' if perfect else '-',mfc='white' if perfect else color,ms=4,lw=1.3,label=k)
        ax.set_xticks(list(points.values()),list(points))
        ax.set_xlabel('Training channel-bank size');ax.set_ylabel('Deployment accuracy (%)')
        ax.set_title(f"{NAMES[ds]} · retrained at ρ = 0.85, noise SD = 2",fontsize=11)
        ax.grid(axis='y',lw=.5,alpha=.4);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=8)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'{ds}_transferred_methods.{ext}',dpi=220)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root,output=args.root.resolve(),args.output.resolve()
    output.mkdir(exist_ok=True,parents=True)
    effects,n=channel_table(root,output)
    cohorts,means=algorithm_tables(root,output)
    figures(output,effects,cohorts,means)
    print(f'Verified {n} paired models, {len(effects)} bank-size/dataset settings, {len(cohorts)} training cohorts.')


if __name__=='__main__':
    main()
