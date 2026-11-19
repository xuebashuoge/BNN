"""Predeclared per-bank validation-only sensitivity alongside the primary shared-triple CIFAR pool."""
import argparse
import csv
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import statistics

from .erasure_validation_seed_illustration import METHODS, OTHER, PROPOSED, SEEDS, csv_write, immutable, load_pool
from .run_experiment import file_fingerprint, save_json

SIZES=(200,500,1000,2000)
RULE_SHA="08e018641684a267c0df441d05c5d475dcb64ee2a4ec111c76f07d2f97f12915"


def select_bank(scores):
    """Maximize the exact per-bank margin with a deterministic lexicographic tie break."""
    if set(scores)!={(k,s) for k in METHODS for s in SEEDS}:
        raise ValueError("Require seven methods at all eight validation seeds")
    ranking=[]
    for triple in combinations(SEEDS,3):
        means={k:sum(scores[k,s] for s in triple)/3 for k in METHODS}
        margin=max(means[k] for k in PROPOSED)-max(means[k] for k in OTHER)
        ranking.append((margin,triple,means))
    ranking.sort(key=lambda v:(-v[0],v[1]))
    margin,triple,means=ranking[0]
    return dict(selected_seeds=list(triple),val_margin_pp=100*float(margin),
                validation_means_pct={k:100*float(v) for k,v in means.items()},
                all_triples=[dict(seeds=list(seeds),margin_pp=100*float(value)) for value,seeds,_ in ranking])


def freeze(out):
    declared=out/"bank_specific_declaration.json"
    if file_fingerprint(declared)["sha256"]!=RULE_SHA:
        raise ValueError("Prospective bank-specific rule changed")
    design,entries=load_pool(out)
    primary_path=out/"frozen_triple.json"
    primary=json.loads(primary_path.read_text())
    if (design["dataset"]!="cifar10" or design["bank_sizes"]!=list(SIZES) or design["seeds"]!=list(SEEDS)
            or len(entries)!=224 or primary["design"]!=file_fingerprint(out/"design.json")):
        raise ValueError("Require the original frozen 224-model CIFAR illustration")
    target=out/"bank_specific_frozen.json"
    if not target.exists() and ((out/"tests"/"reports").exists()
            or any((out/"training"/e["task_id"]/"test_deployment"/"evaluation.json").exists() for e in entries)):
        raise ValueError("Cannot first choose per-bank triples after test inference")
    scores={m:{} for m in SIZES}
    for e in entries:
        val=primary["checkpoints"][e["task_id"]]
        cfg=e["config"]
        if cfg["deployment_channel"]!={"kind":"outage","outage_prob":.5}:
            raise ValueError("Unexpected deployment law")
        scores[cfg["m"]][e["method_label"],cfg["seed"]]=Fraction(val["validation_correct"],val["validation_total"])
    banks={m:select_bank(records) for m,records in scores.items()}
    result=dict(declaration=file_fingerprint(declared),design=file_fingerprint(out/"design.json"),
                shared_freeze=file_fingerprint(primary_path),shared_triple=primary["selected_seeds"],
                interpretation="Supplemental bank-specific validation-selected triples; use the original shared triple or full pool for same-seed comparisons.",
                selections=banks)
    immutable(target,result)
    print(json.dumps({m:[r["selected_seeds"],r["val_margin_pp"]] for m,r in banks.items()}))


def report(out):
    path=out/"bank_specific_frozen.json"
    choice=json.loads(path.read_text())
    if (choice["declaration"]!=file_fingerprint(out/"bank_specific_declaration.json")
            or choice["declaration"]["sha256"]!=RULE_SHA or choice["shared_freeze"]!=file_fingerprint(out/"frozen_triple.json")):
        raise ValueError("Validation freeze differs from the prospective protocol")
    primary=json.loads((out/"seed_illustration.json").read_text())
    if primary["selection"]!=choice["shared_freeze"] or primary["evaluated_models"]!=224:
        raise ValueError("Native full-pool report differs from the frozen bank-specific source")
    with (out/"accuracy_by_seed.csv").open(newline="") as stream:
        observations=list(csv.DictReader(stream))
    if len(observations)!=224:
        raise ValueError("Require all original test scores")
    expected={(m,k,s) for m in SIZES for k in METHODS for s in SEEDS}
    found={(int(r["m"]),r["method"],int(r["seed"])) for r in observations}
    if expected!=found:
        raise ValueError("Incomplete full-pool observations")
    lookup={(int(r["m"]),r["method"],int(r["seed"])):float(r["test_pct"]) for r in observations}
    cells,results=[],{}
    for m in SIZES:
        seeds=choice["selections"][str(m)]["selected_seeds"]
        if tuple(seeds) not in combinations(SEEDS,3):
            raise ValueError("Invalid selected triple")
        means={}
        size_cells=[]
        for k in METHODS:
            x=[lookup[m,k,s] for s in seeds]
            means[k]=statistics.fmean(x)
            size_cells.append(dict(dataset="cifar10",channel="Erasure",m=m,method=k,mean_pct=means[k],sd_pp=statistics.stdev(x),
                                   n_seeds=3,seeds=" ".join(map(str,seeds)),phase="bank_specific_validation_triple",
                                   source=str(out/"bank_specific_triple_summary.json"),outage_prob=.5))
        best=max(PROPOSED,key=means.get);other=max(OTHER,key=means.get)
        results[m]=dict(seeds=seeds,best_proposed=best,best_other=other,test_margin_pp=means[best]-means[other],
                        highest_methods=[k for k in METHODS if max(means.values())-means[k]<=1e-10])
        cells.extend(size_cells)
        csv_write(out/f"m{m}"/"bank_specific_validation_triple_cells.csv",size_cells)
    csv_write(out/"bank_specific_validation_triple_cells.csv",cells)
    for row in observations:
        row["selected_at_this_bank"]=int(row["seed"]) in choice["selections"][row["m"]]["selected_seeds"]
    csv_write(out/"bank_specific_accuracy_by_seed.csv",observations)
    save_json(out/"bank_specific_triple_summary.json",dict(dataset="cifar10",channel="Erasure",outage_prob=.5,
        interpretation="Additional bank-specific validation-selected illustration; triples may differ. The shared-triple and full-pool outputs are the primary seed comparisons.",
        selection=file_fingerprint(path),original_seed_scores=file_fingerprint(out/"accuracy_by_seed.csv"),
        original_report=file_fingerprint(out/"seed_illustration.json"),results=results,cells=cells))
    print(json.dumps(results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","report"))
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    (freeze if args.command=="freeze" else report)(args.out)
