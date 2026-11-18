"""Precision check of frozen two-moons Gibbs deployment accuracy on the original test set."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import torch

from .diagonal_fading_comparison import METHODS, ranking
from .diagonal_moons_revalidation import conditional_correct, save_frozen
from .full_fading import correlation_factor
from .run_experiment import file_fingerprint, load_data, make_models
from .run_task import read_manifest

DRAWS = 2048
TEST_WEIGHT_OFFSET = 24111903
COHORT = "confirm-revalidated"
M_GRID = (200, 500, 1000)


def audit(case_root, index):
    manifest = read_manifest(case_root / f"{COHORT}.jsonl")
    item = manifest[index]
    trained = case_root / COHORT / item["task_id"]
    raw_file = trained / "test_deployment" / "evaluation.json"
    raw = json.loads(raw_file.read_text())
    meta_path, weights = trained / "metadata.json", trained / "weights.pt"
    cfg = json.loads(meta_path.read_text())["config"]
    source = dict(weights=file_fingerprint(weights), metadata=file_fingerprint(meta_path))
    if (raw["checkpoint_source_fingerprints"] != source or raw["split"] != "test"
            or raw["heldout_gibbs_mc"]["draws"] != 100 or cfg["train_size"] != 800 or cfg["width"] != 16
            or cfg["deployment_channel"] != {"kind":"diagonal_fading","rho_c":.35}):
        raise ValueError("Precision check requires exactly the original frozen confirmation")
    torch.set_num_threads(1)
    _, _, test, _, _, _ = load_data(cfg, include_test=True)
    x, y = test.tensors
    if len(x) != raw["heldout_gibbs_mc"]["examples"] or len(x) != 10000:
        raise ValueError("Original held-out example count changed")
    x = x.double()
    model, _ = make_models(cfg, 2, 2)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    factor = correlation_factor(.35, 16)
    generator = torch.Generator().manual_seed(TEST_WEIGHT_OFFSET+cfg["seed"])
    scores = [conditional_correct(x,y,model,model.sample(generator).double(),factor).mean().item()
              for _ in range(DRAWS)]
    if file_fingerprint(weights) != source["weights"] or file_fingerprint(meta_path) != source["metadata"]:
        raise ValueError("Checkpoint changed during accuracy verification")
    save_frozen(case_root / COHORT / "precision-check" / (item["task_id"]+".json"),
                dict(method=item["method_label"], m=cfg["m"], seed=cfg["seed"], test_examples=len(x),
                     test_data_seed=cfg["data_seed"]+1, posterior_draw_seed=TEST_WEIGHT_OFFSET+cfg["seed"],
                     task_id=item["task_id"], source=source, original_evaluation=file_fingerprint(raw_file),
                     channel_mc_100_accuracy=raw["heldout_gibbs_mc"]["accuracy"], posterior_draws=DRAWS,
                     integrated_accuracy=statistics.fmean(scores), accuracy_by_posterior_draw=scores))


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root):
    means, by_seed, ranks, provenance = [], [], {}, {}
    for m in M_GRID:
        source = root / f"m{m}" / "comparison" / "moons"
        manifest = source / f"{COHORT}.jsonl"
        selection = source / f"{COHORT}.selection.json"
        entries = read_manifest(manifest)
        group = {}
        for item in entries:
            stem=item["task_id"]
            audit_path = source / COHORT / "precision-check" / (stem+".json")
            result = json.loads(audit_path.read_text())
            if (result["task_id"] != stem or result["m"] != m or result["seed"] != item["config"]["seed"]
                    or len(result["accuracy_by_posterior_draw"]) != DRAWS
                    or result["method"] != item["method_label"]):
                raise ValueError(f"Unexpected precision report at {audit_path}")
            group[(result["method"],result["seed"])] = result
            by_seed.append(dict(m=m, method=result["method"], seed=result["seed"], task_id=stem,
                                accuracy_integrated_pct=100*result["integrated_accuracy"],
                                accuracy_mc100_pct=100*result["channel_mc_100_accuracy"],
                                source_sha256=file_fingerprint(audit_path)["sha256"]))
        seeds = [163101,163102,163103]
        if set(group) != {(label,s) for label in METHODS for s in seeds}:
            raise ValueError("Incomplete paired seven-method confirmation")
        exact_scores, raw_scores = {}, {}
        for label in METHODS:
            observations = [group[label,s] for s in seeds]
            values = [r["integrated_accuracy"] for r in observations]
            exact_scores[label] = statistics.fmean(values)
            raw_scores[label] = statistics.fmean(r["channel_mc_100_accuracy"] for r in observations)
            means.append(dict(m=m, method=label, n_seeds=len(seeds), seeds=" ".join(map(str,seeds)),
                              accuracy_integrated_pct=100*exact_scores[label],
                              sd_across_seeds_pp=100*statistics.stdev(values), accuracy_mc100_pct=100*raw_scores[label]))
        exact_rank, raw_rank = ranking(exact_scores), ranking(raw_scores)
        p, b = exact_rank["best_matched_proposed"][0], exact_rank["best_other_methods"][0]
        variance_terms, margins = [], []
        for s in seeds:
            diff=[a-bb for a,bb in zip(group[p,s]["accuracy_by_posterior_draw"],
                                       group[b,s]["accuracy_by_posterior_draw"])]
            variance_terms.append(statistics.variance(diff)/DRAWS)
            margins.append(statistics.fmean(diff))
        ranks[m] = dict(integrated=exact_rank, mc_100=raw_rank, paired_difference_pct_points_by_seed=[100*v for v in margins],
                        training_seed_sd_pp=100*statistics.stdev(margins),
                        conditional_posterior_mc_se_pp=100*math.sqrt(sum(variance_terms))/len(seeds))
        provenance[m] = dict(manifest=file_fingerprint(manifest), selection=file_fingerprint(selection),
                             original_summary=file_fingerprint(source/COHORT/"accuracy_summary.json"))
    out=root/"moons_revalidated_integrated"
    out.mkdir(exist_ok=True)
    table(out/"accuracy_by_seed.csv",by_seed)
    table(out/"accuracy_means.csv",means)
    result=dict(method="Gaussian channel integrated exactly conditional on input/label and posterior draw",
                cohort=COHORT, m_grid=M_GRID, train_size=800, width=16, rho_c=.35, test_examples=10000,
                test_data_seed=20261119, posterior_draws=DRAWS, posterior_seed_offset=TEST_WEIGHT_OFFSET,
                uncertainty="Conditional MC SE covers posterior-draw averaging for the observed three posteriors; training-seed SD describes their paired differences.",
                ranking_units="The integrated and mc_100 ranking margins are fractional accuracies; *_pp are percentage points.",
                ranks=ranks, means=means, sources=provenance)
    save_frozen(out/"accuracy_summary.json",result)
    draw(out, means, ranks)
    return ranks


def draw(out, means, ranks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(9.7,3.25),layout="constrained")
    colors=["#555e63","#bf6a25","#0072b2","#009e73"]
    for ax,perfect in zip(axes[:2],(False,True)):
        methods=[k for k,v in METHODS.items() if v[1]==perfect]
        for i,label in enumerate(methods):
            name = label.split("-")[0] if label!="Proposed-Perfect" else "T1/T2"
            values=[r["accuracy_integrated_pct"] for r in means if r["method"]==label]
            ax.plot(range(3),values,marker=("o","s","^","D")[i],ms=4,ls=("-","--","-.",":")[i],
                    color=colors[i],label=name)
        ax.set(title=("Perfect training" if perfect else "Matched training"),ylabel="Deployment accuracy (%)")
        ax.legend(frameon=False, fontsize=8, ncols=2)
    ax=axes[2]
    ax.axhline(0.,color=".45",ls="--",lw=.8)
    for label,key,marker,color in (("100 draws","mc_100","s","#777"),("Integrated","integrated","o","#0072b2")):
        ax.plot(range(3),[100*ranks[m][key]["margin"] for m in M_GRID],marker=marker,color=color,label=label)
    ax.set(title="Best T1/T2 minus best other",ylabel="Accuracy margin (pp)")
    ax.legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3), M_GRID)
        ax.set_xlabel("Channel-bank size $m$")
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","png"):
        fig.savefig(out/f"accuracy.{ext}",dpi=260,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit","collect"))
    parser.add_argument("--root", required=True, type=Path, help="Dataset comparison root for audit, full m-grid root for collect")
    parser.add_argument("--index", type=int, default=0)
    args=parser.parse_args()
    if args.command=="audit":
        audit(args.root,args.index)
    else:
        print(json.dumps(collect(args.root)))


if __name__=="__main__":
    main()
