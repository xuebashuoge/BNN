"""Publish the manuscript tables, their complete seed-level inputs, and comparison index."""
import argparse
from pathlib import Path
import shutil

from .channel_journal_tables import export, SIZES
from .collect_fading_erasure_tables import assemble, read_csv


def publish(fading, erasure, output):
    inputs=output/'data'
    assemble(fading,erasure,inputs)
    modes=('common','illustrative','initial','milder_fading')
    for name in modes:
        dest=output/name
        export(inputs/(name+'_cells.csv'),dest,SIZES,
               {'rho_c':'.35' if name=='milder_fading' else '.85',
                'noise_std':'1' if name=='milder_fading' else '2','outage_prob':'.5'})
        (dest/'all_tables.md').write_text('\n'.join((dest/f'channel_accuracy_m{m}.md').read_text() for m in SIZES))

    lines=['# Deployment accuracy by training channel and bank size','',
           'Each table follows the same layout. For each dataset it pairs a fading row with a 50% feature-erasure row. '
           'Matched ERM, L2, T1 and T2 appear under matched training. The last column is the shared perfect-training '
           'T1/T2 objective. All methods are tested on the channel named in their row. The entries are test accuracy '
           'in percent ± sample standard deviation across training seeds. Bold type identifies the largest unrounded '
           'mean among the displayed columns. The exact CSVs also include perfect-trained ERM and L2.','',
           '## Tables for the journal','',
           'The main fading rows use adjacent-gain correlation 0.85 and receiver-noise standard deviation 2. '
           'The independent erasure probability is 0.50 in every erasure row. The common comparison uses the '
           'broadest completed validation search and the same confirmation seed schedule across the principal sizes '
           'within a dataset and channel. Each reference size uses its most extensively tuned available confirmation.','',
           '| Bank size | PDF | PNG | LaTeX for the manuscript | Exact data |',
           '|---:|---|---|---|---|']
    for m in SIZES:
        s=f'channel_accuracy_m{m}'
        lines.append(f'| {m} | [Table](common/{s}.pdf) | [Preview](common/{s}.png) | '
                     f'[Tabular](common/{s}_tabular.tex) | [CSV](common/{s}.csv) |')
    lines+=['','[All main tables in one PDF](common/channel_accuracy_all.pdf) · [Text tables](common/all_tables.md)',
            '', 'Use the tabular fragment in a table or table* environment with the LaTeX packages booktabs and '
            'multirow. Each page also has a standalone TeX source with caption, channel parameters and seed counts. '
            'The principal grid includes 200, 500 and 1000. The 100 table is the historical CIFAR-10-only reference '
            'design; the 2000 table includes all four datasets.', '',
            '## Comparison views','',
            '- [Initial transfer](initial/channel_accuracy_all.pdf) uses the originally frozen hyperparameters and '
            'keeps all paired primary outcomes. The new erasure data split is aligned with fading. Its prior '
            'erasure-trained configurations came from the historical m=20 validation search.',
            '- [Illustrative confirmations](illustrative/channel_accuracy_all.pdf) prioritize a round in which matched '
            'T1 or T2 had the highest observed test mean. One positive common phase across all principal sizes is '
            'preferred. Otherwise the first qualifying phase in a cell is used. If no positive confirmation exists, '
            'that cell shows its initially transferred cohort. These displayed rounds are selected using observed '
            'rankings and are identified in the cohort CSV.',
            '- [Original mild fading paired with erasure](milder_fading/channel_accuracy_all.pdf) uses the '
            'original rho_c=0.35 and noise SD=1 checkpoints, shown alongside the common erasure results.','',
            '## Rank coverage','',
            'The counts below use strict superiority over all five competing labels for the better matched T1/T2 '
            'mean, including perfect ERM and perfect L2. A T1/T2 tie above all other objectives counts as a lead. '
            'The first number is the number of datasets with a lead; the second is the number evaluated.','',
            '| m | Common fading | Common erasure | Illustrative fading | Illustrative erasure |',
            '|---:|---:|---:|---:|---:|']
    sets={view:read_csv(inputs/(view+'_cohorts.csv')) for view in ('common','illustrative')}
    for m in SIZES:
        counts=[]
        for view in sets:
            for channel in ('Fading','Erasure'):
                cases=[r for r in sets[view] if (int(r['m']),r['channel'])==(m,channel)]
                counts.append(f"{sum(r['leads']=='True' for r in cases)}/{len(cases)}")
        lines.append('| '+str(m)+' | '+' | '.join(counts)+' |')
    lines+=['','[Common cohort choices](data/common_cohorts.csv) · [Illustrative cohort choices](data/illustrative_cohorts.csv) · '
            '[All training rounds and paired margins](data/all_confirmation_rounds.csv) · '
            '[All seed-level scores](data/all_seed_observations.csv) · [Model configurations](data/selected_model_configs.jsonl)','']
    (output/'README.md').write_text('\n'.join(lines))
    for root, name in [(fading,'fading_protocol.md'),(erasure,'erasure_protocol.md')]:
        shutil.copy2(root/'study_protocol.md',output/name)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('fading','erasure','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    publish(a.fading.resolve(),a.erasure.resolve(),a.output.resolve())
