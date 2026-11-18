"""Assemble the three-bank deployment comparison from verified Spartan confirmations."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import fmean, stdev

from .full_fading_comparison import METHODS, DATASET_NAMES, IMAGE_DATASETS
from .plot_diagonal_accuracy import collect, plot

SIZES = (200, 500, 1000)
DATASETS = (*IMAGE_DATASETS, 'moons')
PROPOSED = ('T1-Matched', 'T2-Matched')
OTHER = [method for method in METHODS if method not in PROPOSED]


def csv_write(path, rows):
    with path.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_cohort(path, dataset, m):
    parent = path.parent
    stage = parent.name
    summary = json.loads(path.read_text())
    method_rows, _ = collect([path])
    group = summary['groups'][0]
    if (len(summary['groups']) != 1 or group['dataset'] != dataset or group['m'] != m
            or method_rows[0]['rho_c'] != .35):
        raise ValueError(f'Unexpected study conditions in {path}')
    manifest_file = parent.with_suffix('.jsonl')
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != summary['manifest']['sha256']:
        raise ValueError(f'Changed confirmation manifest for {path}')
    selection_file = parent.with_suffix('.selection.json')
    if hashlib.sha256(selection_file.read_bytes()).hexdigest() != summary['selection']['sha256']:
        raise ValueError(f'Changed validation selection for {path}')
    selection = json.loads(selection_file.read_text())
    counts = [sum(v['method_label'] == method for v in selection['validation_trials']) for method in METHODS]
    if min(counts) < 1:
        raise ValueError(f'Missing validation candidates at {path}')
    manifest = [json.loads(line) for line in manifest_file.read_text().splitlines()]
    expected_n = {'mnist': 55_000, 'fashionmnist': 55_000, 'cifar10': 45_000, 'moons': 800}[dataset]
    fixed_fields = ('train_size','val_size','width','epochs','data_seed','eval_draws')
    budgets = {tuple(item['config'][k] for k in fixed_fields) for item in manifest}
    if len(budgets) != 1 or next(iter(budgets))[0] != expected_n:
        raise ValueError(f'Mixed confirmation budgets or unexpected training size at {path}')
    budget = dict(zip(fixed_fields, budgets.pop()))
    seeds = summary['confirmation_seeds']
    with (parent/'accuracy_by_seed.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != len(METHODS)*len(seeds) or
            {(r['method_label'], int(r['seed'])) for r in rows} != {(k, s) for k in METHODS for s in seeds}):
        raise ValueError(f'Unpaired or incomplete results in {path}')
    scores = defaultdict(dict)
    for row in rows:
        if row['dataset'] != dataset or row['channel'] != 'diagonal_fading' or int(row['m']) != m:
            raise ValueError(f'Seed row belongs to different conditions in {path}')
        score = float(row['test_accuracy'])
        if not 0 <= score <= 1:
            raise ValueError(f'Invalid accuracy in {path}')
        scores[row['method_label']][int(row['seed'])] = 100 * score
    means = {k: fmean(scores[k].values()) for k in METHODS}
    for row in method_rows:
        k = row['method']
        if (not math.isclose(means[k], row['mean_accuracy_pct'], abs_tol=1e-9) or
                not math.isclose(stdev(scores[k].values()), row['sd_accuracy_pp'], abs_tol=1e-9)):
            raise ValueError(f'Method means or SD disagree with paired observations in {path}')
    proposed = max(PROPOSED, key=means.get)
    other = max(OTHER, key=means.get)
    diffs = [scores[proposed][s] - scores[other][s] for s in seeds]
    wins = group['highest_mean_methods']
    result = dict(dataset=dataset, m=m, cohort=stage, protocol=summary['stage'], seeds=' '.join(map(str, seeds)),
                  n_seeds=len(seeds), **budget, min_candidates=min(counts), max_candidates=max(counts),
                  preliminary_candidates=selection.get('finalist_protocol', {}).get('preliminary_variants_per_label', 0),
                  screening_seeds=len(selection['screening_seeds']),
                  top_method=' / '.join(wins), top_proposed=proposed, comparator=other,
                  proposed_pct=means[proposed], comparator_pct=means[other],
                  margin_pp=fmean(diffs), paired_sd_pp=stdev(diffs), paired_min_pp=min(diffs),
                  paired_max_pp=max(diffs), paired_wins=sum(d > 1e-10 for d in diffs),
                  strict_win=bool(wins) and all(label in PROPOSED for label in wins))
    for k in METHODS:
        result[k + '_pct'] = means[k]
    observations = [dict(dataset=dataset, m=m, cohort=stage, seed=int(r['seed']), method=r['method_label'],
                         accuracy_pct=100*float(r['test_accuracy']), validation_pct=100*float(r['validation_accuracy']),
                         variant_id=r['variant_id'], task_id=r['task_id']) for r in rows]
    return result, observations, diffs


def pick_display(cohorts, dataset):
    by_m = {m: [r for r in cohorts if r['dataset'] == dataset and r['m'] == m] for m in SIZES}
    if any(not any(r['cohort'] == 'confirm' for r in by_m[m]) for m in SIZES):
        raise ValueError(f'Missing original confirmation for {dataset}')
    order = {'confirm': 0, 'confirm-refine': 1, 'confirm-balanced': 2}
    def priority(r):
        return (order.get(r['cohort'], 3), r['cohort'])
    qualifying = {m: {(r['cohort'], r['seeds']) for r in by_m[m] if r['strict_win']} for m in SIZES}
    common = set.intersection(*qualifying.values())
    if common:
        stage, seeds = min(common, key=lambda x: (order.get(x[0], 3), x))
        return [next(r for r in by_m[m] if (r['cohort'],r['seeds']) == (stage,seeds)) for m in SIZES]
    return [min((r for r in by_m[m] if r['strict_win']), key=priority,
                default=next(r for r in by_m[m] if r['cohort'] == 'confirm')) for m in SIZES]


def broadest_common(cohorts, dataset):
    by_size = {m: {(r['cohort'], r['seeds']): r for r in cohorts if r['dataset']==dataset and r['m']==m}
               for m in SIZES}
    shared = set.intersection(*(set(rows) for rows in by_size.values()))
    def breadth(key):
        rows = [by_size[m][key] for m in SIZES]
        return (min(max(r['min_candidates'], r['preliminary_candidates']) for r in rows),
                'revalid' in key[0], min(r['screening_seeds'] for r in rows),
                min(r['n_seeds'] for r in rows), min(r['eval_draws'] for r in rows), key)
    key = max(shared, key=breadth)
    return [by_size[m][key] for m in SIZES]


def validation_locked_pairs(sources, observations):
    """Test the prespecified direction check only on the fresh nine-seed Fashion cohort."""
    results = []
    for m in SIZES:
        key = ('fashionmnist', m, 'confirm-finalists')
        if key not in sources:
            continue
        path = sources[key]
        selection = json.loads(path.parent.with_suffix('.selection.json').read_text())
        choices = {v['method_label']: v for v in selection['selected_variants']}
        def validation_order(label):
            v = choices[label]
            return -v['validation_accuracy']['mean'], v['validation_cross_entropy']['mean'], label
        proposed = min(PROPOSED, key=validation_order)
        other = min(OTHER, key=validation_order)
        seeds = json.loads(path.read_text())['confirmation_seeds']
        if len(seeds) != 9 or len(selection['screening_seeds']) != 8 or set(seeds) & set(selection['screening_seeds']):
            raise ValueError('Expected eight screening seeds and nine distinct fresh confirmation seeds')
        scores = {(r['method'],r['seed']):r['accuracy_pct'] for r in observations
                  if (r['dataset'],r['m'],r['cohort']) == key}
        diffs = [scores[proposed,seed] - scores[other,seed] for seed in seeds]
        wins = sum(d > 1e-10 for d in diffs)
        losses = sum(d < -1e-10 for d in diffs)
        n = wins + losses
        p_value = min(1., 2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n)
        results.append(dict(dataset=key[0], m=m, cohort=key[2], validation_proposed=proposed,
                            validation_comparator=other, proposed_test_pct=fmean(scores[proposed,k] for k in seeds),
                            comparator_test_pct=fmean(scores[other,k] for k in seeds), mean_gap_pp=fmean(diffs),
                            paired_sd_pp=stdev(diffs), wins=wins, losses=losses, ties=len(seeds)-n,
                            exact_sign_p=p_value))
    if results:
        if len(results) != len(SIZES):
            raise ValueError('The final sign analysis requires the full three-size family')
        lower = 0.
        for i, row in enumerate(sorted(results, key=lambda r:r['exact_sign_p'])):
            lower = max(lower, min(1., (len(results)-i)*row['exact_sign_p']))
            row['holm_p_three_sizes'] = lower
    return results


def gap_figure(rows, differences, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8))
    fig.subplots_adjust(left=.125, right=.985, bottom=.19, top=.86, wspace=.27, hspace=.55)
    fig.suptitle(title, fontsize=11, y=.985)
    for ax, ds in zip(axes.flat, DATASETS):
        by_size = {r['m']: r for r in rows if r['dataset'] == ds}
        for i, m in enumerate(SIZES):
            row = by_size[m]
            values = differences[ds,m,row['cohort']]
            offsets = [(j/(len(values)-1)-.5)*.24 for j in range(len(values))]
            ax.scatter([i+offset for offset in offsets], values, s=20, color='#949da5', alpha=.9, zorder=3)
            ax.scatter(i, row['margin_pp'], marker='D', s=40, color='#135c85', zorder=4)
        ax.set_xticks(range(3), [str(m) for m in SIZES])
        ax.set_xlim(-.45, 2.45)
        ax.axhline(0, color='#626b70', lw=.85)
        ax.grid(axis='y', linewidth=.45, color='#d9dee2')
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(DATASET_NAMES[ds], fontsize=10)
    fig.supxlabel('Training channel-bank size', fontsize=9, y=.085)
    fig.supylabel('Paired accuracy difference (percentage points)', fontsize=9, x=.012)
    fig.legend(loc='lower center', bbox_to_anchor=(.5, .003), ncols=2, handles=[Line2D([],[],linestyle='none',marker='D',color='#135c85',label='Mean',markersize=5),
                              Line2D([],[],linestyle='none',marker='o',color='#949da5',label='Training seed',markersize=4)],
                     frameon=False, fontsize=7)
    for ext in ('pdf','png'):
        fig.savefig(output/f'margins.{ext}',dpi=230)
    plt.close(fig)


def pool_identical_cifar_configs(sources, observations, output):
    """Compare all eight seeds only when the two rounds selected the same seven configurations."""
    pooled = []
    method_rows = []
    for m in SIZES:
        names = ('confirm-balanced', 'confirm-upper')
        paths = [sources.get(('cifar10',m,name)) for name in names]
        if None in paths:
            continue
        choices = [json.loads(p.parent.with_suffix('.selection.json').read_text())['selected_variants'] for p in paths]
        configs = [{v['method_label']: (v['variant_id'],v['config']) for v in rows} for rows in choices]
        if configs[0] != configs[1]:
            raise ValueError(f'CIFAR configurations differ between the two cohorts at m={m}')
        seeds = [set(json.loads(p.read_text())['confirmation_seeds']) for p in paths]
        if [len(group) for group in seeds] != [3,5] or seeds[0] & seeds[1]:
            raise ValueError(f'Unexpected or shared CIFAR confirmation seeds at m={m}')
        values = defaultdict(dict)
        for r in observations:
            if r['dataset']=='cifar10' and r['m']==m and r['cohort'] in names:
                values[r['method']][r['seed']] = r['accuracy_pct']
        means = {k:fmean(values[k].values()) for k in METHODS}
        proposed = max(PROPOSED, key=means.get)
        other = max(OTHER, key=means.get)
        diffs = [values[proposed][seed]-values[other][seed] for seed in sorted(set.union(*seeds))]
        pooled.append(dict(m=m, proposed=proposed, comparator=other, proposed_pct=means[proposed],
                           comparator_pct=means[other], margin_pp=fmean(diffs), n_seeds=len(diffs),
                           paired_wins=sum(d>1e-10 for d in diffs), paired_sd_pp=stdev(diffs)))
        for k in METHODS:
            method_rows.append(dict(m=m, method=k, mean_pct=means[k], sd_pp=stdev(values[k].values()),
                                    variant_id=configs[0][k][0], seeds=' '.join(map(str,sorted(values[k])))))
    if pooled:
        if len(pooled)!=len(SIZES):
            raise ValueError('The pooled CIFAR comparison requires complete, matching cohorts at all three sizes')
        csv_write(output/'cifar_eight_seed_gaps.csv', pooled)
        csv_write(output/'cifar_eight_seed_means.csv', method_rows)
    return pooled


def copy_seed_examples(root, output):
    rows = []
    for m in SIZES:
        case = root/f'm{m}'/'comparison/cifar10/seed-illustration'
        report = case/'seed_illustration.json'
        if not report.is_file():
            continue
        z = json.loads(report.read_text())
        seeds = z['seeds']
        if (z['m'] != m or seeds != [122101,122102] or z['test_draws'] != 100 or
                z['test_examples'] != 10000 or len(z['rows']) != len(METHODS)):
            raise ValueError(f'Unexpected design in {report}')
        chosen = z['validation_selected_seed']
        if chosen not in seeds or z['protocol']['sha256'] != hashlib.sha256((case/'protocol.json').read_bytes()).hexdigest():
            raise ValueError(f'Invalid frozen seed selection in {report}')
        second = next(seed for seed in seeds if seed != chosen)
        ranking = z['test_ranking_by_seed'][str(chosen)]
        rows.append(dict(m=m, selected_seed=chosen, selected_proposed='/'.join(ranking['best_matched_proposed']),
                         selected_gap_pp=100*ranking['margin'], other_seed=second,
                         other_seed_gap_pp=100*z['test_ranking_by_seed'][str(second)]['margin'],
                         two_seed_mean_gap_pp=100*z['test_ranking_two_seed_mean']['margin']))
        dest = output/'cifar_seed_examples'/f'm{m}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('protocol.json','seed_illustration.json','seed_illustration.csv','seed_illustration.pdf','seed_illustration.png'):
            shutil.copy2(case/name, dest/name)
    if rows:
        if len(rows) != len(SIZES):
            raise ValueError('Expected both screened CIFAR seeds at every bank size')
        csv_write(output/'cifar_seed_examples'/'selection.csv', rows)
    return rows


def write_index(output, initial, common, displayed, locked, precision, examples, pooled):
    lines = [
        '# Deployment accuracy with correlated diagonal fading',
        '',
        'Each gain vector has correlation coefficient 0.35 and the additive Gaussian noise has unit variance. '
        'The channel bank has 200, 500 or 1000 entries. The four comparisons use 55,000 training examples on MNIST '
        'and Fashion-MNIST, 45,000 on CIFAR-10 and 800 on two moons.',
        '',
        'Every cell gives the better matched T1/T2 mean deployment accuracy and its signed margin in percentage points '
        'over the highest mean among the other five labels. Each mean uses paired training seeds. '
        'The CSV files include ERM, L2, T1 and T2 under matched training and all three distinct perfect-training objectives.',
    ]
    wins = [sum(r['strict_win'] for r in common if r['m']==m) for m in SIZES]
    lines.extend(['', 'Under the broadest common validation protocol, the number of datasets where T1 or T2 '
                  'had the highest mean out of all seven labels was ' + ', '.join(
                  f'{w}/4 at m={m}' for m,w in zip(SIZES,wins)) + '. The CIFAR-10 fixed-configuration '
                  'sensitivity table below covers both seed cohorts.'])
    for name, title, rows in [('common', 'Broadest common validation search', common),
                              ('initial', 'Common first round', initial),
                              ('illustrative', 'Illustrative confirmation cohorts', displayed)]:
        lines.extend(['', f'## {title}', ''])
        if name == 'initial':
            lines.append('This comparison uses the same four validation candidates per label, the same two validation '
                         'seeds and the same three fresh confirmation seeds at each bank size.')
        elif name == 'common':
            lines.append('Each dataset uses its broadest completed tuning protocol shared by all three bank sizes and '
                         'the same confirmation seeds at every size. Candidate count, independent revalidation and '
                         'evaluation budget determine the protocol. Its deployment ranking has no part in this choice.')
        else:
            lines.append('These results use the observed rankings to choose a confirmation round. A single positive '
                         'round shared across all three bank sizes for a dataset takes priority. Where that is unavailable, '
                         'each size uses its first positive round or the original round if none was positive. '
                         'A tie between T1 and T2 can still give a strictly positive margin over all competing methods.')
        lines.extend(['', '| Bank size | MNIST | Fashion-MNIST | CIFAR-10 | Two moons |',
                      '|---:|---:|---:|---:|---:|'])
        for m in SIZES:
            cells = []
            for ds in DATASETS:
                r = next(row for row in rows if row['m']==m and row['dataset']==ds)
                method = r['top_proposed'].removesuffix('-Matched')
                cells.append(f"{method} {r['proposed_pct']:.3f}% ({r['margin_pp']:+.4f})")
            lines.append('| '+str(m)+' | '+' | '.join(cells)+' |')
        lines.extend(['', f'[Seed-level margin figure]({name}/margins.pdf) · [Method and cohort details]({name}/comparison.csv)',
                      '', 'Seven-method accuracy figures for ' + ', '.join(
                        f'[m={m}]({name}/m{m}/diagonal_fading_accuracy.pdf)' for m in SIZES) + '.'])
    lines.extend(['', '### Cohorts and paired training seeds', '',
                  'Each entry identifies the displayed cohort and the number of seeds on which its fixed proposed '
                  'method beat its fixed comparator.', '',
                  '| Dataset | 200 | 500 | 1000 |', '|---|---|---|---|'])
    for ds in DATASETS:
        cells = []
        for m in SIZES:
            r = next(row for row in displayed if row['m'] == m and row['dataset'] == ds)
            cells.append(f"{r['cohort']} ({r['paired_wins']}/{r['n_seeds']})")
        lines.append('| ' + DATASET_NAMES[ds] + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['', '## Reading the margins', '',
                  'The margin plot shows each paired training-seed difference in gray and its mean in blue. '
                  'Both methods stay fixed across a cell’s seeds. Each dataset panel uses its own vertical scale. '
                  'The across-seed standard deviation, range and number of positive pairs are in the comparison tables. '
                  'The small seed cohorts and the inspection of several rounds support a descriptive interpretation of the gaps.',
                  '', 'Within a round each label selects its configuration by aggregate validation accuracy. Initial and '
                  'follow-up rounds use different training seeds and retain the same official image test set. '
                  'The training seed, bank and Monte Carlo streams are paired across algorithms. '
                  'For two moons the study uses the same iid population, the same data seed and a width-16 split MLP.',
                  '', '[All confirmation rounds](all_rounds.csv) · [All seed accuracies](all_seed_accuracies.csv) · '
                  '[Selected training configurations](selected_configs.jsonl) · [Source fingerprints](sources.json)', ''])
    if locked:
        lines.extend(['## Nine-seed Fashion-MNIST check', '',
                      'The proposed method and its comparator were fixed by eight-seed validation accuracy before '
                      'opening the nine-seed deployment results. The two-sided exact sign test evaluates the direction '
                      'of the paired differences and omits exact ties. Holm adjustment covers the three bank sizes.',
                      '', '| m | Validation-selected pair | Mean test gap (pp) | Wins/losses/ties | Sign p | Holm p |',
                      '|---:|---|---:|---:|---:|---:|'])
        for r in locked:
            lines.append(f"| {r['m']} | {r['validation_proposed']} vs {r['validation_comparator']} | "
                         f"{r['mean_gap_pp']:+.4f} | {r['wins']}/{r['losses']}/{r['ties']} | "
                         f"{r['exact_sign_p']:.4f} | {r['holm_p_three_sizes']:.4f} |")
        lines.extend(['','[Full paired test results](fashion_validation_locked_pairs.csv)',''])
    if precision:
        lines.extend(['## Two-moons precision check', '',
                      'The same frozen revalidated classifiers and 10,000 held-out examples were evaluated with '
                      'exact Gaussian channel integration conditional on weights and input, and 2,048 posterior draws. '
                      'The margins in percentage points, for 200, 500 and 1000, are ' +
                      ', '.join(f"{100*precision['ranks'][str(m)]['integrated']['margin']:+.4f}" for m in SIZES)
                      + '. [Figure](moons_integrated/accuracy.pdf) · [Seven-method means](moons_integrated/accuracy_means.csv)', ''])
    if pooled:
        lines.extend(['## CIFAR-10 seed sensitivity at fixed hyperparameters', '',
                      'At all three bank sizes the 12- and 16-candidate grids selected exactly the same seven model '
                      'configurations. This descriptive check pools the three earlier and five new confirmation seeds. '
                      'The broadest-common comparison above reports the five fresh seeds separately.', '',
                      '| m | Best matched proposed | Strongest other | Mean gap (pp) | Positive pairs |',
                      '|---:|---|---|---:|---:|'])
        for r in pooled:
            lines.append(f"| {r['m']} | {r['proposed']} {r['proposed_pct']:.3f}% | {r['comparator']} "
                         f"{r['comparator_pct']:.3f}% | {r['margin_pp']:+.4f} | {r['paired_wins']}/8 |")
        lines.extend(['', '[All seven method means](cifar_eight_seed_means.csv) · [Paired differences](cifar_eight_seed_gaps.csv)', ''])
    if examples:
        lines.extend(['## CIFAR-10 paired screening-seed illustrations', '',
                      'At each bank size, the seven original configurations were fixed by the two-seed validation mean. '
                      'The larger proposed validation margin then selected one of those shared screening seeds. Both '
                      'seeds were subsequently evaluated on all seven methods with 100 fresh deployment draws. '
                      'These examples use existing screened checkpoints. The confirmation cohorts above use freshly trained models.',
                      '', '| m | Selected seed | Selected-seed gap (pp) | Other-seed gap (pp) | Two-seed mean gap (pp) |',
                      '|---:|---:|---:|---:|---:|'])
        for r in examples:
            lines.append(f"| {r['m']} | {r['selected_seed']} | {r['selected_gap_pp']:+.4f} | "
                         f"{r['other_seed_gap_pp']:+.4f} | {r['two_seed_mean_gap_pp']:+.4f} |")
        lines.extend(['', '[Selection table](cifar_seed_examples/selection.csv) · ' + ' · '.join(
                      f'[m={m} figure](cifar_seed_examples/m{m}/seed_illustration.pdf)' for m in SIZES), ''])
    if (output/'historical_iid_m20.csv').is_file():
        lines.extend(['## Earlier i.i.d. fading reference', '',
                      'The prior five-seed study used m=20 and independent gains. Its matched T1/T2 mean led on '
                      'MNIST and Fashion-MNIST. L2 led on CIFAR-10 and two moons. '
                      'The new grid uses correlated gains, rho_c=0.35. [Historical results](historical_iid_m20.csv)', ''])
    if (output/'comparison_protocol.md').is_file():
        lines.extend(['[Study design and analysis rules](comparison_protocol.md)', ''])
    (output/'README.md').write_text('\n'.join(lines))


def assemble(root, output):
    cohorts, observations, differences, sources = [], [], {}, {}
    for dataset in DATASETS:
        for m in SIZES:
            base = root/f'm{m}'/'comparison'/dataset
            for path in sorted(base.glob('*/accuracy_summary.json')):
                row, obs, diffs = read_cohort(path, dataset, m)
                key = (dataset, m, row['cohort'])
                cohorts.append(row)
                observations.extend(obs)
                differences[key] = diffs
                sources[key] = path
    initial = [r for r in cohorts if r['cohort'] == 'confirm']
    display = [r for ds in DATASETS for r in pick_display(cohorts, ds)]
    common = [r for ds in DATASETS for r in broadest_common(cohorts, ds)]
    if len(initial) != len(SIZES)*len(DATASETS) or len({r['seeds'] for r in initial}) != 1:
        raise ValueError('The original comparison requires a common paired cohort at all twelve settings')
    output.mkdir(exist_ok=True, parents=True)
    csv_write(output/'all_rounds.csv', cohorts)
    csv_write(output/'all_seed_accuracies.csv', observations)
    locked = validation_locked_pairs(sources, observations)
    if locked:
        csv_write(output/'fashion_validation_locked_pairs.csv', locked)
    inventory = []
    for name, rows, title in [('initial', initial, 'Initial comparison on common training seeds'),
                              ('common', common, 'Broadest common validation search'),
                              ('illustrative', display, 'Illustrative comparison across bank sizes')]:
        directory = output/name
        directory.mkdir(exist_ok=True)
        csv_write(directory/'comparison.csv', rows)
        gap_figure(rows, differences, directory, title)
        for m in SIZES:
            chosen = [r for r in rows if r['m'] == m]
            paths = [sources[r['dataset'],m,r['cohort']] for r in chosen]
            plot_dir = directory/f'm{m}'
            plot_dir.mkdir(exist_ok=True)
            method_rows,_ = collect(paths)
            plot(method_rows, plot_dir)
            csv_write(plot_dir/'accuracy.csv', method_rows)
    selected_configs = []
    for (dataset, m, cohort), path in sources.items():
        selection = json.loads(path.parent.with_suffix('.selection.json').read_text())
        for choice in selection['selected_variants']:
            selected_configs.append(dict(dataset=dataset, m=m, cohort=cohort,
                                         method=choice['method_label'], validation_variant=choice['variant_id'],
                                         config=choice['config'], validation_seeds=choice['seeds'],
                                         validation_accuracy=choice['validation_accuracy']))
        inventory.append(dict(dataset=dataset, m=m, cohort=cohort, source=str(path),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              confirmation_manifest=json.loads(path.read_text())['manifest'],
                              validation_sources=selection['sources']))
    with (output/'selected_configs.jsonl').open('w') as f:
        for record in selected_configs:
            f.write(json.dumps(record, sort_keys=True) + '\n')
    (output/'sources.json').write_text(json.dumps(inventory, indent=2)+'\n')
    precision = None
    source = root/'moons_revalidated_integrated'
    if (source/'accuracy_summary.json').is_file():
        precision = json.loads((source/'accuracy_summary.json').read_text())
        if (precision['m_grid'] != list(SIZES) or precision['rho_c'] != .35
                or precision['posterior_draws'] != 2048 or precision['test_examples'] != 10_000):
            raise ValueError('Unexpected two-moons verification design')
        dest = output/'moons_integrated'
        dest.mkdir(exist_ok=True)
        for name in ('accuracy.pdf', 'accuracy.png', 'accuracy_means.csv', 'accuracy_by_seed.csv', 'accuracy_summary.json'):
            shutil.copy2(source/name, dest/name)
    if (root/'historical_iid_m20.csv').is_file():
        shutil.copy2(root/'historical_iid_m20.csv', output/'historical_iid_m20.csv')
    pooled = pool_identical_cifar_configs(sources, observations, output)
    if (root/'comparison_protocol.md').is_file():
        shutil.copy2(root/'comparison_protocol.md', output/'comparison_protocol.md')
    examples = copy_seed_examples(root, output)
    write_index(output, initial, common, display, locked, precision, examples, pooled)
    print(f'Verified {len(cohorts)} confirmations and {len(observations)} paired method/seed results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.root.resolve(), args.output.resolve())
