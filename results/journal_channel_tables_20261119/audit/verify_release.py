from pathlib import Path
import csv
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from statistics import fmean,stdev
import hashlib,json,math,re,sys
from PIL import Image

root=Path(sys.argv[1]).resolve()
methods=['ERM-Matched','L2-Matched','T1-Matched','T2-Matched','Proposed-Perfect']
C=('Fading','Erasure')
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def num(s):return Decimal(s).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-8)
observ=defaultdict(list)
for r in rows(root/'data/all_seed_observations.csv'):
 observ[r['dataset'],r['channel'],r['m'],r['phase'],r['method']].append((int(r['seed']),float(r['accuracy_pct'])))
rounds=rows(root/'data/all_confirmation_rounds.csv')
assert len({(r['dataset'],r['channel'],r['m']) for r in rounds})==34
assert {r['m'] for r in rounds}=={'100','200','500','1000','2000'}
for row in rows(root/'data/all_method_means.csv'):
 vals=observ[row['dataset'],row['channel'],row['m'],row['phase'],row['method']]
 assert len(vals)==len(set(k for k,v in vals))==int(row['n_seeds'])
 assert close(row['mean_pct'],fmean(v for k,v in vals))
 assert close(row['sd_pp'],stdev(v for k,v in vals))
for path in sorted(root.glob('*/channel_accuracy_exact.csv')):
 results=rows(path);assert len(results)==238 and sum(r['displayed']=='True' for r in results)==170
 cells={(int(r['m']),r['dataset'],r['channel'],r['method']):r for r in results}
 assert len(cells)==238
 for m in (100,200,500,1000,2000):
  ds=['cifar10'] if m==100 else ['moons','mnist','fashionmnist','cifar10']
  tex=(path.parent/f'channel_accuracy_m{m}.tex').read_text()
  body=[l for l in tex.splitlines() if ' & Fading & ' in l or ' & Erasure & ' in l]
  assert len(body)==2*len(ds)
  for i,d in enumerate(ds):
   for j,ch in enumerate(C):
    group=[cells[m,d,ch,k] for k in methods]
    best=max(Decimal(r['mean_pct']) for r in group)
    actual=body[i*2+j].split(' & ')[2:]
    for r,text in zip(group,actual,strict=True):
     b=best-Decimal(r['mean_pct'])<=Decimal('1e-10')
     assert (r['highest_displayed']=='True')==b
     mean=f'{num(r["mean_pct"]):.2f}'
     if b:mean='\\mathbf{'+mean+'}'
     assert ('$'+mean+'\\,\\pm\\,'+f'{num(r["sd_pp"]):.2f}'+'$') in text,(r,text)
     if ch=='Erasure':assert float(r['outage_prob'])==.50
     else:assert (float(r['rho_c']),float(r['noise_std']))==((.35,1) if path.parent.name=='milder_fading' else (.85,2))
  for ext in ('pdf','png','csv','tex','md'):
   assert (path.parent/f'channel_accuracy_m{m}.{ext}').is_file()
  assert (path.parent/f'channel_accuracy_m{m}_tabular.tex').is_file()
for p in root.rglob('*.png'):
 with Image.open(p) as im:im.verify()
for p in root.rglob('*.pdf'):
 buf=p.read_bytes()
 assert buf.startswith(b'%PDF') and b'%%EOF' in buf[-160:]
for link in re.findall(r'\]\(([^)]+)\)',(root/'README.md').read_text()):
 assert (root/link).is_file(),link
print(f'Verified {len(rounds)} seven-method confirmations, {sum(map(len,observ.values()))} seed-/method-level scores, and every displayed mean, sample SD, bold winner and channel setting in {len(list(root.glob("*/channel_accuracy_exact.csv")))} table sets.')
