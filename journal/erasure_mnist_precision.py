"""Prespecified 200-draw held-out accuracy check at the original nine penalty60 MNIST posteriors."""
import argparse
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics

import torch

from .erasure_bank_comparison import ERASURE
from .erasure_penalty_final import CONFIRM
from .full_fading_comparison import METHODS, write_once
from .run_experiment import (channel_bank_fingerprint, check_device, checkpoint_bank_state, file_fingerprint,
                             load_data, make_loader, make_models, normalize_config, save_json)
from .run_task import read_manifest

MS = (200,500,1000,2000)
DRAWS = 200
OFFSET = 9_100_027
DECLARATION_SHA = "07aa25cd60cc0a415edfd25f85a7c6ef9cd7b05a9d777272866b6bc7129c7af7"
TARGET = ("T1-Matched","T2-Matched")


def declaration(out):
    sha=file_fingerprint(out/"declaration.json")
    if sha["sha256"]!=DECLARATION_SHA:
        raise ValueError("The prospective RNG schedule or inclusion rule changed")
    return sha


def pin(base,out,m):
    if m not in MS:
        raise ValueError("Unexpected m")
    decl=declaration(out)
    root=base/"p050"/f"m{m}"/"mnist"/"penalty60"/"retuned"
    file=root/"manifest.jsonl"; selection_path=file.with_suffix(".selection.json")
    entries=read_manifest(file); selection=json.loads(selection_path.read_text())
    selected={v["method_label"]:v["variant_id"] for v in selection["selected_variants"]}
    if (selection["stage"]!="erasure_penalty60" or selection["confirmation_seeds"]!=list(CONFIRM)
            or len(entries)!=len(METHODS)*len(CONFIRM) or set(selected)!=set(METHODS)
            or {(r["method_label"],r["config"]["seed"]) for r in entries}!={(k,s) for k in METHODS for s in CONFIRM}):
        raise ValueError(f"Require all seven original nine-seed confirmation methods at {root}")
    budgets={tuple(r["config"][k] for k in ("train_size","val_size","epochs","width","batch_size","eval_draws","data_seed")) for r in entries}
    if len(budgets)!=1:
        raise ValueError("Unpaired model/data/batch sizes or evaluation budgets")
    for item in entries:
        c=item["config"]; label=item["method_label"]
        if (item["role"]!="erasure_confirmation" or item["selected_from_validation_variant"]!=selected[label]
                or c["dataset"]!="mnist" or c["m"]!=m or c["epochs"]!=60 or c["deployment_channel"]!=ERASURE
                or c["train_channel"]!=({"kind":"perfect"} if METHODS[label][1] else ERASURE)
                or c["method"]!=METHODS[label][0] or c["predictor"]!="gaussian" or c["data_seed"]!=20261118
                or c["eval_seed"]!=c["seed"]+20000):
            raise ValueError(f"Unexpected frozen confirmation: {item['task_id']}")
    protocol=dict(declaration=decl, m=m, manifest_path=str(file),manifest=file_fingerprint(file),
                  selection=file_fingerprint(selection_path),seeds=CONFIRM, models=len(entries))
    write_once(out/f"m{m}"/"cohort.json",json.dumps(protocol,indent=2)+"\n")
    return protocol


@torch.no_grad()
def correct_counts(model,bank,batches,eval_seed,draws,device):
    model.eval()
    posterior=torch.Generator(device=device).manual_seed(eval_seed)
    channels=torch.Generator().manual_seed(eval_seed+1)
    result=[]
    for _ in range(draws):
        theta=model.sample(posterior)
        hits=0
        for x,y in batches:
            scores=model(x,theta,bank,"deployment",ERASURE,channels)
            hits+=(scores.argmax(1)==y).sum().item()
        result.append(hits)
    return result


def run(out,m,index):
    decl=declaration(out);root=out/f"m{m}"
    spec=json.loads((root/"cohort.json").read_text())
    manifest_file=Path(spec["manifest_path"])
    if (spec["declaration"]!=decl or spec["manifest"]!=file_fingerprint(manifest_file)
            or spec["selection"]!=file_fingerprint(manifest_file.with_suffix(".selection.json"))):
        raise ValueError("Pinned manifest or validation selection changed")
    entries=read_manifest(manifest_file)
    if not 0<=index<len(entries):
        raise ValueError("Confirmation index out of range")
    item=entries[index];cp=manifest_file.parent/"runs"/item["task_id"]
    native_file=cp/"test_deployment"/"evaluation.json"
    native=json.loads(native_file.read_text())
    sources={k:file_fingerprint(cp/name) for k,name in (("weights","weights.pt"),("metadata","metadata.json"))}
    meta=json.loads((cp/"metadata.json").read_text());cfg=normalize_config(meta["config"]);nc=native["evaluation_config"]
    risk=native["heldout_gibbs_mc"]
    native_seed=cfg["eval_seed"]+1_000_003
    if (any(cfg[k]!=v for k,v in item["config"].items() if k not in ("device","threads"))
            or sources!=native["checkpoint_source_fingerprints"] or nc!=dict(cfg,eval_seed=native_seed)
            or native["split"]!="test" or native["risk_scope"]!="deployment" or risk["channel_source"]!="deployment"
            or not risk["predictor"]["stochastic"] or not native["checkpoint_verification"]["bank_hash_matches_metadata"]
            or risk["examples"]!=10000 or risk["draws"]!=cfg["eval_draws"]):
        raise ValueError(f"Existing held-out evaluation differs from its declared frozen checkpoint: {cp}")
    dest=root/"trials"/(item["task_id"]+".json")
    sig=dict(cohort=file_fingerprint(root/"cohort.json"), checkpoint=sources,native_evaluation=file_fingerprint(native_file))
    if dest.is_file():
        old=json.loads(dest.read_text())
        if old["sources"]!=sig or len(old["precision_counts"])!=DRAWS:
            raise ValueError(f"Stale existing precision result: {dest}")
        return
    check_device(cfg);torch.set_num_threads(cfg["threads"])
    train,val,test,dim,classes,split=load_data(cfg,include_test=True);del train,val
    if split!=meta["split"] or len(test)!=10000:
        raise ValueError("Test samples differ from the original cohort")
    model,bank=make_models(cfg,dim,classes)
    state=torch.load(cp/"weights.pt",map_location=cfg["device"],weights_only=True)
    model.load_state_dict(state["model"]); bank.load_state_dict(checkpoint_bank_state(state,cfg,regenerated_bank=bank))
    bank_sha=channel_bank_fingerprint(bank,meta.get("training_channel_bank_hash_algorithm"))
    if bank_sha!=meta["training_channel_bank_hash"] or bank_sha!=native["checkpoint_verification"]["saved_channel_bank_hash"]:
        raise ValueError("Frozen training bank differs from the native evaluation")
    batches=[(x.to(cfg["device"]),y.to(cfg["device"])) for x,y in make_loader(test,cfg)]
    native_counts=[round(10000*r["accuracy"]) for r in risk["replicate_statistics"]]
    if (len(native_counts)!=risk["draws"] or abs(statistics.fmean(native_counts)/10000-risk["accuracy"])>1e-12
            or correct_counts(model,bank,batches,native_seed,1,cfg["device"])!=native_counts[:1]):
        raise ValueError("Original first replicate or stored native mean failed independent reproduction")
    seed=cfg["eval_seed"]+OFFSET
    counts=correct_counts(model,bank,batches,seed,DRAWS,cfg["device"])
    if sources!={k:file_fingerprint(cp/name) for k,name in (("weights","weights.pt"),("metadata","metadata.json"))} or sig["native_evaluation"]!=file_fingerprint(native_file):
        raise ValueError("Frozen sources changed during evaluation")
    save_json(dest,dict(sources=sig, task_id=item["task_id"],variant_id=item["variant_id"],method=item["method_label"],
                        m=m,seed=cfg["seed"],examples=len(test),native_posterior_seed=native_seed,
                        precision_posterior_seed=seed,precision_mask_seed=seed+1,verified_bank_sha256=bank_sha,
                        native_counts=native_counts,precision_counts=counts))
    print(f"Verified {item['task_id']}",flush=True)


def csv_write(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(out,m):
    root=out/f"m{m}";spec=json.loads((root/"cohort.json").read_text())
    manifest=Path(spec["manifest_path"])
    if spec["declaration"]!=declaration(out) or spec["manifest"]!=file_fingerprint(manifest):
        raise ValueError("Aggregated cohort differs from declaration")
    summary_path=manifest.parent/"runs"/"accuracy_summary.json"
    prior=json.loads(summary_path.read_text())
    if prior["manifest"]!=spec["manifest"] or prior["confirmation_seeds"]!=list(CONFIRM):
        raise ValueError("Native summary differs from pinned cohort")
    arrays={}; byseed=[]
    for item in read_manifest(manifest):
        path=root/"trials"/(item["task_id"]+".json");r=json.loads(path.read_text())
        k,s=item["method_label"],item["config"]["seed"]
        if (r["method"]!=k or r["seed"]!=s or r["task_id"]!=item["task_id"] or r["sources"]["cohort"]!=file_fingerprint(root/"cohort.json")
                or r["precision_posterior_seed"]!=s+20000+OFFSET or len(r["precision_counts"])!=DRAWS
                or r["native_posterior_seed"]!=s+20000+1_000_003 or r["examples"]!=10000):
            raise ValueError(f"Unexpected precision result at {path}")
        record=dict(m=m,method=k,seed=s,task_id=r["task_id"],weight_sha256=r["sources"]["checkpoint"]["weights"]["sha256"],
                    native_eval_sha256=r["sources"]["native_evaluation"]["sha256"],precision_sha256=file_fingerprint(path)["sha256"])
        for stage in ("native","precision"):
            acc=[c/100 for c in r[stage+"_counts"]]
            arrays[k,s,stage]=acc
            record[stage+"_accuracy_pct"]=statistics.fmean(acc)
            record[stage+"_mc_se_pp"]=statistics.stdev(acc)/math.sqrt(len(acc))
        record["precision_minus_native_pp"]=record["precision_accuracy_pct"]-record["native_accuracy_pct"]
        byseed.append(record)
    if len(byseed)!=63 or len(arrays)!=126:
        raise ValueError("Missing paired precision comparisons")
    means=[];diffs=[];ranks={};summary={}
    for k in METHODS:
        a=dict(m=m,dataset="mnist",channel="Erasure",method=k,n_seeds=len(CONFIRM),seeds=" ".join(map(str,CONFIRM)))
        for stage in ("native","precision"):
            vals=[statistics.fmean(arrays[k,s,stage]) for s in CONFIRM]
            a[stage+"_mean_pct"]=statistics.fmean(vals);a[stage+"_training_seed_sd_pp"]=statistics.stdev(vals)
        source=prior["groups"][0]["methods"][k]["test_accuracy"]
        if (abs(a["native_mean_pct"]-100*source["mean"])>1e-9 or abs(a["native_training_seed_sd_pp"]-100*source["sd"])>1e-9):
            raise ValueError("Native nine-seed mean/SD is inconsistent with the official report")
        a["precision_minus_native_pp"]=a["precision_mean_pct"]-a["native_mean_pct"]
        means.append(a)
    for left,right in combinations(METHODS,2):
        row=dict(m=m,method_a=left,method_b=right)
        for stage in ("native","precision"):
            gaps=[[a-b for a,b in zip(arrays[left,s,stage],arrays[right,s,stage],strict=True)] for s in CONFIRM]
            avgs=[statistics.fmean(v) for v in gaps]
            row[stage+"_gap_pp"]=statistics.fmean(avgs)
            row[stage+"_paired_training_seed_sd_pp"]=statistics.stdev(avgs)
            row[stage+"_conditional_mc_se_pp"]=math.sqrt(sum(statistics.variance(v)/len(v) for v in gaps))/len(gaps)
        row["change_in_gap_pp"]=row["precision_gap_pp"]-row["native_gap_pp"]
        row["change_in_gap_mc_se_pp"]=math.hypot(row["precision_conditional_mc_se_pp"],row["native_conditional_mc_se_pp"])
        diffs.append(row)
    for stage in ("native","precision"):
        scoring={r["method"]:r[stage+"_mean_pct"] for r in means}
        p=max(TARGET,key=scoring.get);b=max((k for k in METHODS if k not in TARGET),key=scoring.get)
        pair=next(r for r in diffs if {r["method_a"],r["method_b"]}=={p,b})
        ranks[stage]=dict(highest_methods=[k for k,x in scoring.items() if max(scoring.values())-x<=1e-10],
                          best_proposed=p,competitor=b,gap_pp=scoring[p]-scoring[b],
                          conditional_mc_se_pp=pair[stage+"_conditional_mc_se_pp"],
                          paired_training_seed_sd_pp=pair[stage+"_paired_training_seed_sd_pp"])
    csv_write(root/"accuracy_by_seed.csv",byseed);csv_write(root/"native_vs_precision.csv",means)
    csv_write(root/"paired_differences.csv",diffs)
    cells=[dict(dataset="mnist",channel="Erasure",m=m,method=r["method"],mean_pct=r["precision_mean_pct"],
                sd_pp=r["precision_training_seed_sd_pp"],n_seeds=len(CONFIRM),seeds=r["seeds"],
                phase="penalty60_precision200",source=str(root/"summary.json"),outage_prob=.5) for r in means]
    csv_write(root/"precision_cells.csv",cells)
    summary=dict(scope="Conditional Monte Carlo precision check of Gibbs accuracy on the same 10000 official test examples and frozen training seeds; no retraining. Per-pair MC SE conditions on these models and examples, while paired training-seed SD describes differences between the nine trained pairs.",
                 m=m,seeds=CONFIRM,draws=DRAWS,offset=OFFSET,models=len(byseed),declaration=spec["declaration"],
                 source_manifest=spec["manifest"],source_selection=spec["selection"],native_summary=file_fingerprint(summary_path),rankings=ranks)
    save_json(root/"summary.json",summary)
    return ranks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("pin","run","report"));p.add_argument("--out",type=Path,required=True)
    p.add_argument("--base",type=Path);p.add_argument("--m",type=int,choices=MS,required=True);p.add_argument("--index",type=int)
    a=p.parse_args()
    if a.command=="pin": print(json.dumps(pin(a.base,a.out,a.m)))
    elif a.command=="run": run(a.out,a.m,a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else: print(json.dumps(aggregate(a.out,a.m)))


if __name__=="__main__":
    main()
