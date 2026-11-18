"""Pair transferred hard-channel training with the original frozen-model severe deployment draws."""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics

from .diagonal_fading_stress import (DATASET_NAMES,DRAWS,EVAL_OFFSET,LAWS,METHODS,freeze,result_path,table)
from .run_experiment import evaluate_checkpoint, file_fingerprint, save_json
from .run_task import read_manifest


def prepare(stress, matched):
    stress_file = stress/"protocol.json"
    frozen = json.loads(stress_file.read_text())
    sources, tasks = [], []
    for source in frozen["sources"]:
        m, dataset = source["m"],source["dataset"]
        stage = matched/f"m{m}"/dataset
        manifest = stage/"manifest.jsonl"
        transfer = json.loads((stage/"transfer.json").read_text())
        entries = read_manifest(manifest)
        group = {r["task_id"]:r for r in frozen["tasks"] if r["dataset"]==dataset and r["m"]==m}
        if (len(group) != len(entries) or {r["source_task_id"] for r in entries} != set(group)
                or transfer["manifest"] != file_fingerprint(manifest) or transfer["source_manifest"] != source["manifest"]
                or transfer["source_summary"] != source["summary"] or source["seeds"] != sorted(transfer["seeds"])
                or transfer["deployment_law"] != {"kind":"diagonal_fading","rho_c":.85,"noise_std":2.}):
            raise ValueError(f"Hard-training cohort does not pair with original confirmation: {stage}")
        sources.append(dict(m=m,dataset=dataset,cohort=source["cohort"],manifest=transfer["manifest"],
                            root=str(stage.absolute()),transfer=file_fingerprint(stage/"transfer.json"),
                            original_manifest=source["manifest"],seeds=source["seeds"]))
        for i,item in enumerate(entries):
            original=group[item["source_task_id"]]
            if (item["config"]["seed"] != original["seed"] or item["method_label"] != original["method"]
                    or item["config"]["m"] != m or item["config"]["dataset"] != dataset):
                raise ValueError("Source identities differ")
            tasks.append(dict(m=m, dataset=dataset, index=i, method=original["method"], seed=original["seed"],
                              cohort=original["cohort"], hard_root=str(stage.absolute()),hard_id=item["task_id"],
                              source_task_id=original["task_id"],source_checkpoint=original["checkpoint"],
                              eval_seed=item["config"]["eval_seed"]+EVAL_OFFSET,
                              original_checkpoints=original["source"]))
    freeze(stress/"recovery"/"protocol.json",dict(frozen_protocol=file_fingerprint(stress_file),sources=sources,
           draws=DRAWS,baseline="Frozen original model under rho=.85 and noise SD=2",
           difference="Hard-training held-out accuracy minus original frozen severe accuracy, percentage points",
           tasks=tasks))
    print(f"Paired {len(tasks)} hard-training checkpoints in {len(sources)} cohorts")


def folder(stress,item):
    return stress/"recovery"/f"m{item['m']}"/item["dataset"]/item["hard_id"]


def run(stress,m,dataset,index,threads):
    protocol = json.loads((stress/"recovery"/"protocol.json").read_text())
    source = next(s for s in protocol["sources"] if s["m"]==m and s["dataset"]==dataset)
    item = next(r for r in protocol["tasks"] if r["m"]==m and r["dataset"]==dataset and r["index"]==index)
    stage = Path(source["root"])
    native_summary=json.loads((stage/"runs"/"accuracy_summary.json").read_text())
    if (native_summary["manifest"]!=source["manifest"] or native_summary["stage"]!="severity_transfer"
            or sorted(native_summary["confirmation_seeds"])!=source["seeds"]):
        raise ValueError("Hard training report differs from the pinned transfer manifest")
    checkpoint=stage/"runs"/item["hard_id"]
    original=Path(item["source_checkpoint"])
    cfg=json.loads((checkpoint/"metadata.json").read_text())["config"]
    mild=json.loads((original/"metadata.json").read_text())["config"]
    if (file_fingerprint(original/"metadata.json") != item["original_checkpoints"]["metadata"]
            or any(cfg.get(k) != v for k,v in mild.items() if k not in ("train_channel","deployment_channel","threads","device","eval_draws","download"))
            or cfg["deployment_channel"]!={"kind":"diagonal_fading","rho_c":.85,"noise_std":2.}
            or cfg["train_channel"]!=({"kind":"perfect"} if METHODS[item["method"]][1] else cfg["deployment_channel"])
            or cfg["eval_seed"]+EVAL_OFFSET != item["eval_seed"]):
        raise ValueError(f"Severe/original training configurations do not pair: {item['source_task_id']}")
    out = folder(stress,item)
    overrides=out/"overrides.json"
    freeze(overrides,dict(device="cpu",threads=threads,eval_draws=DRAWS,eval_seed=item["eval_seed"]))
    if not (out/"evaluation.json").exists():
        evaluate_checkpoint(checkpoint,str(overrides),out,"test","deployment")
    scores=json.loads((out/"evaluation.json").read_text())
    native=json.loads((checkpoint/"test_deployment"/"evaluation.json").read_text())
    if (scores["checkpoint_source_fingerprints"]!=native["checkpoint_source_fingerprints"]
            or scores["checkpoint_verification"]["saved_channel_bank_hash"]!=native["checkpoint_verification"]["saved_channel_bank_hash"]
            or not scores["checkpoint_verification"]["bank_hash_matches_metadata"]
            or scores["evaluation_config"]["eval_seed"]!=item["eval_seed"] or scores["heldout_gibbs_mc"]["draws"]!=DRAWS):
        raise ValueError("Hard severe-only evaluation differs from the native frozen model")
    print(item["hard_id"])


def aggregate(stress):
    out=stress/"recovery"
    protocol=json.loads((out/"protocol.json").read_text())
    if file_fingerprint(stress/"protocol.json") != protocol["frozen_protocol"]:
        raise ValueError("Original stress-test protocol changed")
    rows, means = [], []
    grouped=defaultdict(list)
    for item in protocol["tasks"]:
        original = dict(item,task_id=item["source_task_id"])
        frozen_file=result_path(stress,original)
        hard_file=folder(stress,item)/"evaluation.json"
        frozen=json.loads(frozen_file.read_text())
        hard=json.loads(hard_file.read_text())
        severe=frozen["evaluations"][LAWS.index((.85,2.))]
        cfg=hard["evaluation_config"]
        risk=hard["heldout_gibbs_mc"]
        if (frozen["source"]!=item["original_checkpoints"] or cfg["eval_seed"]!=frozen["weight_seed"]
                or cfg["eval_seed"]+1!=frozen["channel_seed"] or cfg["eval_draws"]!=frozen["draws"]
                or risk["examples"]!=frozen["test_examples"] or cfg["dataset"]!=item["dataset"] or cfg["m"]!=item["m"]
                or (severe["rho_c"],severe["noise_std"])!=(.85,2.) or cfg["deployment_channel"]["noise_std"]!=2.
                or cfg["deployment_channel"]["rho_c"]!=.85):
            raise ValueError(f"Mismatched paired evaluation at {hard_file}")
        paired=[100*(h["accuracy"]-f) for h,f in zip(risk["replicate_statistics"],severe["replicate_accuracy"])]
        if len(paired)!=DRAWS:
            raise ValueError("Incomplete paired replicate series")
        row=dict(m=item["m"],dataset=item["dataset"],cohort=item["cohort"],method=item["method"],seed=item["seed"],
                 frozen_severe_pct=100*severe["accuracy"],hard_severe_pct=100*risk["accuracy"],
                 recovery_pp=statistics.fmean(paired),paired_mc_se_pp=statistics.stdev(paired)/math.sqrt(DRAWS),
                 mild_source_id=item["source_task_id"],hard_source_id=item["hard_id"],eval_seed=cfg["eval_seed"],
                 original_weight_sha256=frozen["source"]["weights"]["sha256"],
                 hard_weight_sha256=hard["checkpoint_source_fingerprints"]["weights"]["sha256"],
                 original_bank_sha256=frozen["bank_hash"],hard_bank_sha256=hard["checkpoint_verification"]["saved_channel_bank_hash"])
        rows.append(row)
        grouped[(item["m"],item["dataset"],item["cohort"],item["method"])].append(row)
    for (m,d,cohort,method),group in grouped.items():
        changes=[r["recovery_pp"] for r in group]
        means.append(dict(m=m,dataset=d,cohort=cohort,method=method,n_seeds=len(group),
                          seeds=" ".join(map(str,sorted(r["seed"] for r in group))),
                          frozen_severe_pct=statistics.fmean(r["frozen_severe_pct"] for r in group),
                          hard_severe_pct=statistics.fmean(r["hard_severe_pct"] for r in group),
                          recovery_pp=statistics.fmean(changes),paired_training_seed_sd_pp=statistics.stdev(changes)))
    table(out/"recovery_by_seed.csv",rows)
    table(out/"recovery_means.csv",means)
    for m in sorted({r["m"] for r in rows}):
        table(out/f"m{m}"/"recovery_by_seed.csv",[r for r in rows if r["m"]==m])
        table(out/f"m{m}"/"recovery_means.csv",[r for r in means if r["m"]==m])
    draw(out,means)
    save_json(out/"summary.json",dict(draws=DRAWS,models=len(rows),protocol=file_fingerprint(out/"protocol.json"),
              interpretation="Paired severe-deployment comparison of transferred hyperparameters under original versus severe training/selection. Perfect-training methods preserve their noiseless training law and use severe-law validation to select epochs.",
              means=means))
    print(f"Verified {len(rows)} frozen/hard pairs")


def draw(out,means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(2,2,figsize=(9.5,7.1),layout="constrained")
    limit=max(1.,max(abs(r["recovery_pp"]) for r in means))
    for ax,d in zip(axes.flat,("mnist","fashionmnist","cifar10","moons")):
        cohort=[r for r in means if r["dataset"]==d]
        sizes=sorted({r["m"] for r in cohort})
        scores={(r["m"],r["method"]):r["recovery_pp"] for r in cohort}
        values=np.array([[scores[(m,label)] for m in sizes] for label in METHODS])
        im=ax.imshow(values,cmap="RdBu",aspect="auto",norm=TwoSlopeNorm(0.,vmin=-limit,vmax=limit))
        ax.set_title(DATASET_NAMES[d],fontsize=10)
        ax.set_xticks(range(len(sizes)),sizes,fontsize=8)
        ax.set_xlabel("Channel-bank size $m$",fontsize=8)
        ax.set_yticks(range(len(METHODS)),[k.replace("Proposed-Perfect","T1/T2-Perfect").replace("-"," · ") for k in METHODS],fontsize=8)
        for i in range(len(METHODS)):
            for j in range(len(sizes)):
                v=values[i,j]
                ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=8,color="white" if abs(v)>.65*limit else "black")
    fig.suptitle(r"Effect of retraining on severe-law deployment accuracy ($\rho_c=0.85$, $\sigma_N=2$)",fontsize=11)
    fig.colorbar(im,ax=list(axes.flat),label="Retrained minus original accuracy (pp)",shrink=.86)
    for ext in ("png","pdf"):
        fig.savefig(out/f"recovery.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","aggregate"))
    parser.add_argument("--stress",type=Path,required=True)
    parser.add_argument("--matched",type=Path)
    parser.add_argument("--m",type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--index",type=int)
    args=parser.parse_args()
    if args.command=="prepare":
        prepare(args.stress,args.matched)
    elif args.command=="run":
        run(args.stress,args.m,args.dataset,args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]),args.threads)
    else:
        aggregate(args.stress)


if __name__=="__main__":
    main()
