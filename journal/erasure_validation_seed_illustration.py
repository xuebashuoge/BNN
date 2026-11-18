"""Prospectively select one common seed triple using only p=0.50 erasure validation results."""
import argparse
import csv
import fcntl
from fractions import Fraction
from itertools import combinations
import json
import os
from pathlib import Path

from .build_campaign import make_entry
from .campaign import completed_trial, evaluate_trial
from .diagonal_fading_comparison import run_task, validation_rows
from .erasure_bank_comparison import ERASURE
from .full_fading_comparison import METHODS, group_results, report_file, validate_test, write_manifest, write_once
from .run_experiment import file_fingerprint, save_json
from .run_task import read_manifest

SEEDS = tuple(range(490101,490109))
PENALTY60_SEEDS = tuple(range(492201,492209))
DRAWS = 40
PROPOSED = ("T1-Matched","T2-Matched")
OTHER = tuple(label for label in METHODS if label not in PROPOSED)
SELECTION_RULE = ("Among all 56 triples use the largest minimum over the declared bank sizes of best matched T1/T2 "
                  "mean validation accuracy minus best other mean validation accuracy. Break ties by larger average "
                  "margin across sizes, then lexicographically smaller triple. All seven methods use the same seeds.")


def immutable(path,obj):
    write_once(path,json.dumps(obj,indent=2,allow_nan=False)+"\n")


def prepare(sources, out, pool_mode="original"):
    if pool_mode not in ("original","penalty60"):
        raise ValueError("Unknown seed-pool mode")
    pool_seeds = PENALTY60_SEEDS if pool_mode=="penalty60" else SEEDS
    dataset=None
    origins={}
    pool=[]
    for path in sources:
        entries=read_manifest(path)
        conditions={(e["config"]["dataset"],e["config"]["m"]) for e in entries}
        if len(conditions)!=1:
            raise ValueError(f"Mixed data conditions in {path}")
        ds,m=conditions.pop()
        if (dataset is not None and dataset!=ds) or m in origins:
            raise ValueError("Designate one frozen validation-selected manifest per bank size for a single dataset")
        dataset=ds
        if m not in (100,200,500,1000,2000) or (m==100 and ds!="cifar10"):
            raise ValueError("Unsupported reference bank size")
        seeds=sorted({e["config"]["seed"] for e in entries})
        penalty_stage="diagonal_fading_accuracy_erasure_penalty60"
        penalty_flags=[e["stage"]==penalty_stage for e in entries]
        if pool_mode=="penalty60":
            if (not all(penalty_flags) or ds!="mnist" or seeds!=list(range(451201,451210))
                    or any(e["config"]["epochs"]!=60 for e in entries)):
                raise ValueError("The separate seed pool requires genuine nine-seed MNIST penalty60 confirmation manifests")
            selection_path=path.with_suffix(".selection.json")
            selected_record=json.loads(selection_path.read_text())
            if (selected_record.get("stage")!="erasure_penalty60" or selected_record.get("confirmation_seeds")!=seeds
                    or selected_record.get("screening_seeds")!=[431201,431202,431203]
                    or len(selected_record.get("selected_variants",[]))!=len(METHODS)):
                raise ValueError("Unexpected penalty60 frozen validation selection")
        elif any(penalty_flags):
            raise ValueError("The penalty60 confirmations require the separate penalty60 pool")
        if (len(entries)!=len(METHODS)*len(seeds) or set(pool_seeds).intersection(seeds)
                or {(e["method_label"],e["config"]["seed"]) for e in entries}!={(k,s) for k in METHODS for s in seeds}):
            raise ValueError("Source must be a complete disjoint seven-method confirmation cohort")
        records=[q for q in (path.with_suffix(".selection.json"),path.parent/"selection.json",path.parent/"protocol.json") if q.is_file()]
        if not records:
            raise ValueError("Missing frozen selection or erasure-transfer protocol beside the source manifest")
        for record in records:
            declaration=json.loads(record.read_text())
            if "manifest" in declaration and declaration["manifest"] != file_fingerprint(path):
                raise ValueError(f"Source manifest differs from its frozen protocol: {path}")
            if "selected_variants" in declaration:
                confirmed={r["method_label"]:r["variant_id"] for r in declaration["selected_variants"]}
                if (set(confirmed)!=set(METHODS) or any(
                        e.get("selected_from_validation_variant",e["variant_id"])!=confirmed[e["method_label"]]
                        for e in entries)):
                    raise ValueError(f"Source confirmation differs from its frozen validation choices: {path}")
        variants={}
        for label,(method,perfect) in METHODS.items():
            group=[e for e in entries if e["method_label"]==label]
            cfg=group[0]["config"]
            if (cfg["dataset"]!=ds or cfg["deployment_channel"]!=ERASURE
                    or cfg["train_channel"]!=({"kind":"perfect"} if perfect else ERASURE)
                    or cfg["method"]!=method or cfg["data_seed"]!=20261118
                    or len({e["variant_id"] for e in group})!=1
                    or any({k:v for k,v in e["config"].items() if k not in ("seed","channel_seed","eval_seed")}
                           !={k:v for k,v in cfg.items() if k not in ("seed","channel_seed","eval_seed")} for e in group)
                    or any(e.get("role") not in ("erasure_confirmation","severity_confirmation") for e in group)):
                raise ValueError(f"Require exactly one frozen p=.50 {label} configuration at each m")
            variants[label]=group[0]["variant_id"]
            for seed in pool_seeds:
                item=make_entry({**cfg,"eval_draws":DRAWS},seed,f"{ds}_erasure_p050_m{m}",label,
                                ("erasure_penalty60_seed_validation_pool" if pool_mode=="penalty60" else "erasure_seed_validation_pool"),"seed_illustration_validation")
                item["source_confirmation_variant"]=variants[label]
                pool.append(item)
        origins[m]=dict(path=str(path.resolve()),fingerprint=file_fingerprint(path),source_seeds=seeds,
                        choices=variants,selection_records={str(q.resolve()):file_fingerprint(q) for q in records})
    out.mkdir(parents=True,exist_ok=True)
    pool.sort(key=lambda e:(e["config"]["m"],tuple(METHODS).index(e["method_label"]),e["config"]["seed"]))
    write_manifest(out/"pool.jsonl",pool)
    design=dict(dataset=dataset,bank_sizes=sorted(origins),source_confirmations=origins,channel=ERASURE,seeds=pool_seeds,
                test_draws=DRAWS,eval_seed_schedule="training seed + 20000 + 1000003, paired across all methods and sizes",
                selection_rule=SELECTION_RULE,pool_manifest=file_fingerprint(out/"pool.jsonl"))
    if pool_mode=="penalty60":
        design["seed_pool"]="penalty60"
    immutable(out/"design.json",design)
    return {"dataset":dataset,"m":design["bank_sizes"],"models":len(pool)}


def rank_triples(scores, sizes, pool_seeds=SEEDS):
    pool_seeds=tuple(pool_seeds)
    if pool_seeds not in (SEEDS,PENALTY60_SEEDS):
        raise ValueError("Require one of the two prospectively reserved seed pools")
    if not sizes or len(set(sizes))!=len(sizes):
        raise ValueError("Require distinct, nonempty bank sizes")
    expected={(m,label,s) for m in sizes for label in METHODS for s in pool_seeds}
    if set(scores)!=expected:
        raise ValueError("Missing validation observations in the common paired seed pool")
    ranks=[]
    for triple in combinations(pool_seeds,3):
        margins={}
        for m in sizes:
            values={label:sum(scores[m,label,s] for s in triple)/3 for label in METHODS}
            margins[m]=max(values[k] for k in PROPOSED)-max(values[k] for k in OTHER)
        ranks.append((min(margins.values()),sum(margins.values())/len(sizes),triple,margins))
    ranks.sort(key=lambda r:(-r[0],-r[1],r[2]))
    return [dict(seeds=list(seeds),minimum_validation_margin_pp=100*float(minimum),
                 average_validation_margin_pp=100*float(average),
                 margins_pp={str(m):100*float(v) for m,v in margins.items()}) for minimum,average,seeds,margins in ranks]


def load_pool(out):
    design=json.loads((out/"design.json").read_text())
    pool_seeds = PENALTY60_SEEDS if design.get("seed_pool")=="penalty60" else SEEDS
    if (design["channel"]!=ERASURE or design["seeds"]!=list(pool_seeds) or design["test_draws"]!=DRAWS
            or design["pool_manifest"]!=file_fingerprint(out/"pool.jsonl")):
        raise ValueError("Frozen design or pool manifest changed")
    return design,read_manifest(out/"pool.jsonl")


def freeze(out):
    design,entries=load_pool(out)
    scores,checkpoints={},{}
    for entry, row in zip(entries,validation_rows(out/"training",entries),strict=True):
        cfg=entry["config"]
        cp=out/"training"/entry["task_id"]
        if not (out/"frozen_triple.json").exists() and (cp/"test_deployment"/"evaluation.json").exists():
            raise ValueError("Test scores predate the frozen validation seed choice")
        status=completed_trial(cp,row["config"])
        if not status or status["validation_accuracy"]!=row["validation_accuracy"]:
            raise ValueError(f"Validation record differs from the frozen checkpoint at {cp}")
        # Accuracy is the total number correct across equally sized validation draws.
        denom=cfg["val_size"]*cfg["val_draws"]
        count=round(row["validation_accuracy"]*denom)
        if abs(count/denom-row["validation_accuracy"])>1e-10:
            raise ValueError("Validation score is not reproducible from its draw budget")
        scores[cfg["m"],entry["method_label"],cfg["seed"]]=Fraction(count,denom)
        checkpoints[entry["task_id"]]=dict(validation_accuracy=row["validation_accuracy"],
                validation_correct=count,validation_total=denom,
                source={k:file_fingerprint(cp/name) for k,name in (("weights","weights.pt"),("metadata","metadata.json"))})
    ranks=rank_triples(scores,design["bank_sizes"],design["seeds"])
    result=dict(design=file_fingerprint(out/"design.json"),pool_manifest=design["pool_manifest"],
                selection_rule=SELECTION_RULE,selected_seeds=ranks[0]["seeds"],all_triples=ranks,checkpoints=checkpoints)
    immutable(out/"frozen_triple.json",result)
    return ranks[0]


def evaluate(out,index):
    design,entries=load_pool(out)
    if not 0 <= index < len(entries):
        raise ValueError("Pool index out of range")
    item=entries[index]
    path=report_file(out/"tests",item)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(".evaluate.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return evaluate_one(out,design,item,path)


def evaluate_one(out,design,item,path):
    selection=json.loads((out/"frozen_triple.json").read_text())
    if selection["design"]!=file_fingerprint(out/"design.json") or selection["pool_manifest"]!=design["pool_manifest"]:
        raise ValueError("Test inference requires the original frozen triple and design")
    train=out/"training"
    record=json.loads(report_file(train,item).read_text())
    if any(k.startswith("test_") for k in record):
        raise ValueError("Training records are validation-only")
    cp=train/item["task_id"]
    frozen=selection["checkpoints"][item["task_id"]]["source"]
    if any(file_fingerprint(cp/name)!=frozen[k] for k,name in (("weights","weights.pt"),("metadata","metadata.json"))):
        raise ValueError("Checkpoint changed after the validation-only triple was frozen")
    if path.is_file():
        previous=json.loads(path.read_text())
        if previous["frozen_selection"]!=file_fingerprint(out/"frozen_triple.json"):
            raise ValueError("Existing test score belongs to a different selection")
        result=validate_test(item,previous,train)
        return {"task_id":item["task_id"],"test_accuracy":result["test_accuracy"]}
    result=evaluate_trial(record,DRAWS,full_risks=False,resume=True)
    result=validate_test(item,result,train)
    if not 0<=result["test_accuracy"]<=1:
        raise ValueError("Invalid held-out accuracy")
    result["frozen_selection"]=file_fingerprint(out/"frozen_triple.json")
    immutable(report_file(out/"tests",item),result)
    return {"task_id":item["task_id"],"test_accuracy":result["test_accuracy"]}


def csv_write(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(out):
    design,entries=load_pool(out)
    frozen_path=out/"frozen_triple.json"
    frozen=json.loads(frozen_path.read_text())
    if frozen["design"]!=file_fingerprint(out/"design.json"):
        raise ValueError("Frozen selection differs from design")
    checked=[]
    observations=[]
    for item in entries:
        row=json.loads(report_file(out/"tests",item).read_text())
        if row["frozen_selection"]!=file_fingerprint(frozen_path):
            raise ValueError("Test release mixes selections")
        row=validate_test(item,row,out/"training")
        checked.append(row)
        observations.append(dict(dataset=design["dataset"],m=item["config"]["m"],channel="Erasure",method=item["method_label"],
                                 seed=row["seed"],selected=row["seed"] in frozen["selected_seeds"],
                                 validation_pct=100*row["validation_accuracy"],test_pct=100*row["test_accuracy"],
                                 source_variant=item["source_confirmation_variant"],task_id=item["task_id"]))
    summaries={}
    for name,seeds in (("full_pool",design["seeds"]),("validation_selected_triple",frozen["selected_seeds"])):
        groups=group_results([r for r in checked if r["seed"] in seeds],seeds)
        summaries[name]=groups
        cells=[]
        for group in groups:
            for label in METHODS:
                stats=group["methods"][label]["test_accuracy"]
                cells.append(dict(dataset=design["dataset"],channel="Erasure",m=group["m"],method=label,
                                  mean_pct=100*stats["mean"],sd_pp=100*stats["sd"],n_seeds=len(seeds),
                                  seeds=" ".join(map(str,seeds)),phase=name,source=str((out/"seed_illustration.json").resolve()),
                                  outage_prob=.5))
        csv_write(out/(name+"_cells.csv"),cells)
        for m in design["bank_sizes"]:
            per_size=out/f"m{m}"
            per_size.mkdir(exist_ok=True)
            csv_write(per_size/(name+"_cells.csv"),[r for r in cells if r["m"]==m])
    csv_write(out/"accuracy_by_seed.csv",observations)
    for m in design["bank_sizes"]:
        csv_write(out/f"m{m}"/"accuracy_by_seed.csv",[r for r in observations if r["m"]==m])
    csv_write(out/"validation_triples.csv",[dict(seeds=" ".join(map(str,r["seeds"])),
              minimum_validation_margin_pp=r["minimum_validation_margin_pp"],average_validation_margin_pp=r["average_validation_margin_pp"],
              **{f"val_margin_m{m}_pp":v for m,v in r["margins_pp"].items()}) for r in frozen["all_triples"]])
    result=dict(design=file_fingerprint(out/"design.json"),selection=file_fingerprint(frozen_path),
                dataset=design["dataset"],bank_sizes=design["bank_sizes"],pool_seeds=design["seeds"],selected_seeds=frozen["selected_seeds"],
                interpretation="Validation-selected training-seed illustration, paired across bank sizes and methods. The full pool describes all eight newly trained seeds.",
                test_sample="The official test examples (or fixed independent moons test sample) are reused with fresh paired posterior/channel Monte Carlo.",
                evaluated_models=len(checked),draws=DRAWS,groups=summaries)
    save_json(out/"seed_illustration.json",result)
    return {"selected":frozen["selected_seeds"],"models":len(checked)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("prepare")
    build.add_argument("--pool",choices=("original","penalty60"),default="original")
    build.add_argument("--source",action="append",type=Path,required=True,help="Designated p=.50 seven-method confirmation manifest, repeat per m")
    train=sub.add_parser("train")
    train.add_argument("--threads",type=int,default=2)
    train.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    test=sub.add_parser("evaluate")
    for p in (train,test):
        p.add_argument("--index",type=int)
    for p in (build,train,test,sub.add_parser("freeze"),sub.add_parser("report")):
        p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare": result=prepare(args.source,args.output,args.pool)
    elif args.command=="train":
        load_pool(args.output)
        args.manifest=args.output/"pool.jsonl";args.output_root=args.output/"training"
        result=run_task(args)
    elif args.command=="freeze":result=freeze(args.output)
    elif args.command=="evaluate":result=evaluate(args.output,args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:result=report(args.output)
    print(json.dumps(result))


if __name__=="__main__":
    main()
