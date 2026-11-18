"""Measure paired deployment-law changes at the same frozen diagonal-fading checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics

import torch

from .full_fading_comparison import DATASET_NAMES, METHODS, write_once
from .run_experiment import (channel_bank_fingerprint, checkpoint_bank_state, check_device, file_fingerprint,
                             load_data, make_models, normalize_config, save_json)
from .run_task import read_manifest
from .theorem1 import validate_channel

GRID_COHORTS = {"mnist":"confirm", "fashionmnist":"confirm-finalists",
                "cifar10":"confirm-upper", "moons":"confirm-revalidated"}
OLD_COHORTS = {"mnist":"confirm", "fashionmnist":"confirm-refine",
                "cifar10":"confirm-balanced", "moons":"confirm"}
LAWS = ((.35,1.), (.85,1.), (.35,2.), (.85,2.))
# The two conditional main effects are reported at both levels of the other factor.
CONTRASTS = {"correlation_at_noise1":(1,0), "correlation_at_noise2":(3,2),
             "noise_at_rho035":(2,0), "noise_at_rho085":(3,1), "both_vs_original":(3,0)}
DRAWS = 24
EVAL_OFFSET = 6_102_047


def freeze(path, value):
    write_once(path, json.dumps(value, indent=2, allow_nan=False)+"\n")


def prepare(old, grid, output):
    tasks, sources = [], []
    for m in (100,200,500,1000,2000):
        if m == 100:
            parent, cohorts = old/"m100"/"comparison", {"cifar10":"confirm"}
        elif m == 2000:
            parent, cohorts = old/"comparison", OLD_COHORTS
        else:
            parent, cohorts = grid/f"m{m}"/"comparison", GRID_COHORTS
        for dataset, stage in cohorts.items():
            cohort = parent/dataset/stage
            summary_path = cohort/"accuracy_summary.json"
            selection = cohort.with_suffix(".selection.json")
            manifest = cohort.with_suffix(".jsonl")
            summary = json.loads(summary_path.read_text())
            if (summary["selection"] != file_fingerprint(selection)
                    or summary["manifest"] != file_fingerprint(manifest)):
                raise ValueError(f"Confirmation sources changed: {cohort}")
            group = summary["groups"][0]
            seeds = sorted(summary["confirmation_seeds"])
            entries = read_manifest(manifest)
            if (len(summary["groups"]) != 1 or group["dataset"] != dataset or group["m"] != m
                    or group["channel"] != "diagonal_fading" or len(entries) != len(METHODS)*len(seeds)
                    or {(e["method_label"],e["config"]["seed"]) for e in entries}
                    != {(label,seed) for label in METHODS for seed in seeds}):
                raise ValueError(f"Mixed or incomplete confirmation cohort at {cohort}")
            sources.append(dict(m=m, dataset=dataset, cohort=stage, seeds=seeds, source=str(cohort.resolve()),
                                manifest=summary["manifest"], selection=summary["selection"],
                                summary=file_fingerprint(summary_path)))
            for item in entries:
                checkpoint = cohort/item["task_id"]
                cfg = item["config"]
                previous = json.loads((checkpoint/"test_deployment"/"evaluation.json").read_text())
                origin = {k:file_fingerprint(checkpoint/n) for k,n in (("weights","weights.pt"),("metadata","metadata.json"))}
                if (previous["checkpoint_source_fingerprints"] != origin or previous["split"] != "test"
                        or previous["risk_scope"] != "deployment"
                        or not previous["checkpoint_verification"]["bank_hash_matches_metadata"]
                        or cfg["deployment_channel"] != {"kind":"diagonal_fading","rho_c":.35}
                        or cfg["train_channel"] != ({"kind":"perfect"} if METHODS[item["method_label"]][1] else cfg["deployment_channel"])
                        or cfg["dataset"] != dataset or cfg["m"] != m):
                    raise ValueError(f"Unexpected frozen source {checkpoint}")
                tasks.append(dict(m=m, dataset=dataset, cohort=stage, seed=cfg["seed"], method=item["method_label"],
                                  task_id=item["task_id"], checkpoint=str(checkpoint.resolve()), source=origin,
                                  bank_hash=previous["checkpoint_verification"]["saved_channel_bank_hash"]))
    protocol = dict(draws=DRAWS, eval_seed_offset=EVAL_OFFSET, laws=LAWS, gain_mean=.5, gain_variance=1,
                    metric="Held-out Gibbs accuracy; normalized bounded loss is outside this deployment-only study",
                    pairing="One common weight draw across the full test set, with identical white gain/noise streams across the four laws",
                    sources=sources, tasks=tasks)
    freeze(output/"protocol.json", protocol)
    print(json.dumps(dict(models=len(tasks), cohorts=len(sources), paired_laws=len(LAWS), draws=DRAWS)))


def law_config(rho, std):
    return validate_channel(dict(kind="diagonal_fading", rho_c=rho, noise_std=std))


@torch.no_grad()
def paired_scores(model, bank, batches, seed, draws):
    """Draw one posterior per test sweep, reusing its encoder result under every law."""
    model.eval()
    weights = torch.Generator().manual_seed(seed)
    innovations = [torch.Generator().manual_seed(seed+1) for _ in LAWS]
    laws = [law_config(rho, std) for rho,std in LAWS]
    n = sum(len(y) for _,y in batches)
    history = [[] for _ in laws]
    for _ in range(draws):
        theta = model.split(model.sample(weights))
        correct = [0]*len(laws)
        for x,y in batches:
            hidden = model.transmit(x, theta)
            for i, (law, rng) in enumerate(zip(laws, innovations)):
                rx = bank.apply(hidden, "deployment", law, rng)
                correct[i] += (model.receive(rx, theta).argmax(-1) == y).sum().item()
        for values,count in zip(history,correct):
            values.append(count/n)
    return history


def result_path(output,item):
    return output/f"m{item['m']}"/"trials"/item["dataset"]/(item["task_id"]+".json")


def run(output, index, threads):
    protocol = json.loads((output/"protocol.json").read_text())
    item = protocol["tasks"][index]
    checkpoint = Path(item["checkpoint"])
    metadata = json.loads((checkpoint/"metadata.json").read_text())
    cfg = normalize_config(dict(metadata["config"], device="cpu", threads=threads))
    frozen = file_fingerprint(output/"protocol.json")
    path = result_path(output,item)
    def verify_source():
        if any(file_fingerprint(checkpoint/n) != item["source"][k] for k,n in
               (("weights","weights.pt"),("metadata","metadata.json"))):
            raise ValueError(f"Checkpoint changed: {checkpoint}")
    verify_source()
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["protocol"] != frozen or existing["source"] != item["source"]:
            raise ValueError(f"Different evaluation already exists: {path}")
        return
    check_device(cfg)
    torch.set_num_threads(threads)
    train, val, test, input_dim, classes, split = load_data(cfg,include_test=True)
    del train,val
    if split != metadata["split"] or len(test) != 10000:
        raise ValueError("Evaluation data split differs from the frozen model")
    model, bank = make_models(cfg,input_dim,classes)
    original = torch.load(checkpoint/"weights.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(original["model"])
    bank.load_state_dict(checkpoint_bank_state(original,cfg,regenerated_bank=bank))
    bank_hash = channel_bank_fingerprint(bank,metadata.get("training_channel_bank_hash_algorithm"))
    if bank_hash != item["bank_hash"] or bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError("Restored channel bank differs from training")
    x,y = test.tensors
    batches = list(zip(x.split(cfg["batch_size"]),y.split(cfg["batch_size"])))
    eval_seed = cfg["eval_seed"]+EVAL_OFFSET
    history = paired_scores(model,bank,batches,eval_seed,DRAWS)
    verify_source()
    evaluations = [dict(rho_c=rho, noise_std=std, accuracy=statistics.fmean(draws),
                        replicate_accuracy=draws) for (rho,std),draws in zip(LAWS,history)]
    save_json(path,dict(**item, protocol=frozen, verified_bank_sha256=bank_hash, test_examples=len(test),
                        eval_seed=eval_seed, weight_seed=eval_seed, channel_seed=eval_seed+1,
                        draws=DRAWS, train_size=cfg["train_size"], width=cfg["width"], evaluations=evaluations))
    print(path)


def table(path, rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output):
    protocol_file = output/"protocol.json"
    sha = file_fingerprint(protocol_file)
    protocol = json.loads(protocol_file.read_text())
    seeds, effects, means, differences = [],[],[],[]
    acc_groups, delta_groups = defaultdict(list), defaultdict(list)
    for item in protocol["tasks"]:
        result = json.loads(result_path(output,item).read_text())
        measurements = result["evaluations"]
        if (result["protocol"] != sha or any(result[k] != v for k,v in item.items())
                or result["verified_bank_sha256"] != item["bank_hash"]
                or [(r["rho_c"],r["noise_std"]) for r in measurements] != list(LAWS)
                or any(len(r["replicate_accuracy"]) != DRAWS or not 0 <= r["accuracy"] <= 1
                       or statistics.fmean(r["replicate_accuracy"]) != r["accuracy"] for r in measurements)):
            raise ValueError(f"Unexpected result for {item['task_id']}")
        cell = tuple(item[k] for k in ("m","dataset","cohort","method"))
        common = dict(zip(("m","dataset","cohort","method"),cell))
        for (rho,std), measurement in zip(LAWS,measurements):
            value = 100*measurement["accuracy"]
            seeds.append(dict(**common, seed=item["seed"],rho_c=rho,noise_std=std,accuracy_pct=value,
                              task_id=item["task_id"], weight_seed=result["weight_seed"],channel_seed=result["channel_seed"]))
            acc_groups[cell+(rho,std)].append((item["seed"],value))
        paired = [r["replicate_accuracy"] for r in measurements]
        draw_diffs = {k:[100*(a-b) for a,b in zip(paired[i],paired[j])] for k,(i,j) in CONTRASTS.items()}
        draw_diffs["interaction"] = [a-b for a,b in zip(draw_diffs["noise_at_rho085"],draw_diffs["noise_at_rho035"])]
        for contrast, deltas in draw_diffs.items():
            delta = statistics.fmean(deltas)
            effects.append(dict(**common,seed=item["seed"],contrast=contrast,change_pp=delta,
                                paired_mc_se_pp=statistics.stdev(deltas)/math.sqrt(DRAWS)))
            delta_groups[cell+(contrast,)].append((item["seed"],delta))
    for groups, cols, pos, mean, sd in (
            (acc_groups,("rho_c","noise_std"),means,"mean_accuracy_pct","training_seed_sd_pp"),
            (delta_groups,("contrast",),differences,"mean_change_pp","paired_training_seed_sd_pp")):
        for key, sample in groups.items():
            sample.sort()
            base = dict(zip(("m","dataset","cohort","method",*cols),key))
            values=[v for _,v in sample]
            pos.append(dict(**base, n_seeds=len(sample), seeds=" ".join(str(s) for s,_ in sample),
                            **{mean:statistics.fmean(values), sd:statistics.stdev(values)}))
    rankings=[]
    for source in protocol["sources"]:
        for rho,std in LAWS:
            scores={r["method"]:r["mean_accuracy_pct"] for r in means if
                    (r["m"],r["dataset"],r["rho_c"],r["noise_std"])==(source["m"],source["dataset"],rho,std)}
            best=max(scores.values())
            rankings.append(dict(m=source["m"],dataset=source["dataset"],cohort=source["cohort"],rho_c=rho,
                                 noise_std=std,highest=" ".join(k for k,v in scores.items() if best-v<=1e-10),
                                 highest_accuracy_pct=best))
    tables = {"accuracy_by_seed":seeds,"accuracy_means":means,"effects_by_seed":effects,
              "paired_effects":differences,"rankings":rankings}
    for stem,rows in tables.items():
        table(output/(stem+".csv"),rows)
    for m in sorted({s["m"] for s in seeds}):
        for stem,rows in tables.items():
            table(output/f"m{m}"/(stem+".csv"),[r for r in rows if r["m"]==m])
        plot(output/f"m{m}",m,[r for r in means if r["m"]==m])
    summary=dict(protocol=sha,draws=DRAWS,models=len(protocol["tasks"]),sources=protocol["sources"],
                 laws=LAWS,metrics="Accuracy in percent; changes in percentage points",
                 change_convention="New minus comparison law, evaluated on paired weight and underlying Gaussian draws",
                 accuracy_means=means,paired_effects=differences,rankings=rankings)
    save_json(output/"summary.json",summary)
    print(json.dumps(dict(models=len(protocol["tasks"]),groups=len(means), paired_seed_contrasts=len(effects))))


def plot(out,m,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    datasets=[d for d in GRID_COHORTS if any(r["dataset"]==d for r in rows)]
    count=len(datasets)
    fig,axes=plt.subplots(2 if count>2 else 1,2 if count>1 else 1,squeeze=False,
                          figsize=(9.3,6.6) if count>1 else (5.3,3.6),layout="constrained")
    matrices={}
    for d in datasets:
        data={(r["method"],r["rho_c"],r["noise_std"]):r["mean_accuracy_pct"] for r in rows if r["dataset"]==d}
        matrices[d]=np.array([[data[(label,rho,std)] for rho,std in LAWS] for label in METHODS])
    limit=max(1.,max(np.max(np.abs(a-a[:,:1])) for a in matrices.values()))
    for ax,d in zip(axes.flat,datasets):
        a=matrices[d]; deltas=a-a[:,:1]
        im=ax.imshow(deltas,aspect="auto",cmap="RdBu",norm=TwoSlopeNorm(0,vmin=-limit,vmax=limit))
        ax.set_title(DATASET_NAMES[d],fontsize=10)
        ax.set_xticks(range(4),[f"{rho:g}, {std:g}" for rho,std in LAWS],fontsize=8)
        ax.set_xlabel(r"Deployment $(\rho_c,\sigma_N)$",fontsize=8)
        ax.set_yticks(range(7),[k.replace("Proposed-Perfect","T1/T2-Perfect").replace("-"," · ") for k in METHODS],fontsize=8)
        for i in range(7):
            for j in range(4):
                ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(deltas[i,j])>.62*limit else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="Change from original law (pp)")
    fig.suptitle(f"Frozen-model deployment accuracy (%) · m = {m}",fontsize=11)
    for ext in ("pdf","png"):
        fig.savefig(out/f"deployment_stress.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","aggregate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--old",type=Path)
    parser.add_argument("--grid",type=Path)
    parser.add_argument("--index",type=int)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    if args.command=="prepare":
        if not args.old or not args.grid:
            parser.error("prepare requires --old and --grid data roots")
        prepare(args.old,args.grid,args.output)
    elif args.command=="run":
        index=args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        run(args.output,index,args.threads)
    else:
        aggregate(args.output)


if __name__=="__main__":
    main()
