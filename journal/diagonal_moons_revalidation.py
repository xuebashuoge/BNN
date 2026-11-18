"""Validate frozen two-moons candidates on fresh synthetic examples before confirmation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .campaign import select_variants
from .diagonal_fading_comparison import (METHODS, SCREEN_SEEDS, balanced_entries,
                                        confirmation_entries, grid, conditions, ranking)
from .full_fading import correlation_factor
from .full_fading_comparison import variant_grid, write_manifest, write_once
from .run_experiment import (file_fingerprint, make_models, normalize_config, sample_moons_iid)
from .run_task import read_manifest

N_VALID = 16384
N_WEIGHTS = 128
DATA_SEED = 24111901
WEIGHT_OFFSET = 24111902
SHARD = 8
PROTOCOL = dict(data_seed=DATA_SEED, validation_examples=N_VALID, posterior_draws=N_WEIGHTS,
                posterior_seed="24111902 + original training seed",
                accuracy="Conditional Gaussian probability of correct binary decision given x, y and weights",
                tie_break="Original checkpoint validation cross entropy, then variant ID")


def save_frozen(path, obj):
    write_once(path, json.dumps(obj, indent=2, allow_nan=False)+"\n")


def collect_entries(root):
    stages = {stage: read_manifest(root / (stage+".jsonl")) for stage in ("screen", "refine", "balanced")}
    design = conditions(stages["screen"])
    if (len(design) != 1 or design[0][0] != "moons" or design[0][2:] != (.35, 800)
            or not variant_grid(stages["screen"], grid(design, "screen"))
            or not variant_grid(stages["refine"], grid(design, "refine"))
            or not variant_grid(stages["balanced"], balanced_entries(stages["screen"], []))):
        raise ValueError("Expected all original, refined and balanced candidates for two moons")
    return [(stage,item) for stage, entries in stages.items() for item in entries]


@torch.no_grad()
def conditional_correct(x, y, model, theta, factor, noise_std=1.):
    """For u=w2[:,1]-w2[:,0], class margin is Gaussian with variance (f*u)'R(f*u) + noise_std² ||u||²."""
    w1, b1, w2, b2 = model.split(theta)
    hidden = (x @ w1 + b1).relu()
    receiver = w2[:,1] - w2[:,0]
    weighted = hidden * receiver
    mean = .5*weighted.sum(1) + (b2[1]-b2[0])
    sd = ((weighted @ factor).square().sum(1) + (noise_std*receiver).square().sum()).sqrt()
    p = torch.special.ndtr(((2*y-1)*mean) / sd.clamp_min(torch.finfo(sd.dtype).tiny))
    return torch.where(sd > 0, p, ((mean > 0) == y).to(p.dtype))


def score_path(root, item):
    return root / "revalidation" / "scores" / (item["task_id"]+".json")


def score(root, index):
    records = collect_entries(root)
    if index < 0 or index*SHARD >= len(records):
        raise ValueError("Shard out of range")
    torch.set_num_threads(1)
    x, y = sample_moons_iid(N_VALID, .3, DATA_SEED)
    x = torch.from_numpy(x).double()
    y = torch.from_numpy(y)
    factor = correlation_factor(.35, 16)
    for stage, item in records[index*SHARD:(index+1)*SHARD]:
        checkpoint = root / stage / item["task_id"]
        cfg = normalize_config(item["config"])
        metadata_path = checkpoint / "metadata.json"
        meta = json.loads(metadata_path.read_text())
        weight_path = checkpoint / "weights.pt"
        origin = {name:file_fingerprint(p) for name,p in (("weights",weight_path),("metadata",metadata_path))}
        if any(meta["config"][k] != cfg[k] for k in cfg if k not in ("threads", "device")):
            raise ValueError("Frozen training configuration differs from its manifest")
        dst = score_path(root, item)
        if dst.exists():
            prior = json.loads(dst.read_text())
            if prior["origin"] != origin or prior["protocol"] != PROTOCOL:
                raise ValueError(f"Stale validation result: {dst}")
            continue
        model, _ = make_models(cfg, 2, 2)
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True)["model"])
        model.eval()
        generator = torch.Generator().manual_seed(WEIGHT_OFFSET + cfg["seed"])
        total = 0.
        for _ in range(N_WEIGHTS):
            # The float32 draw matches the trained posterior. Integrate the channel in float64.
            total += conditional_correct(x, y, model, model.sample(generator).double(), factor).mean().item()
        if file_fingerprint(weight_path) != origin["weights"] or file_fingerprint(metadata_path) != origin["metadata"]:
            raise ValueError("Checkpoint changed during validation")
        save_frozen(dst, dict(**item, seed=cfg["seed"], status="complete", origin=origin,
                              protocol=PROTOCOL, validation_accuracy=total/N_WEIGHTS,
                              validation_cross_entropy=meta["selected_validation_cross_entropy"]))


def freeze(root):
    entries, rows = [], []
    digest = hashlib.sha256()
    for stage, item in collect_entries(root):
        payload = score_path(root, item).read_bytes()
        row = json.loads(payload)
        if (row["protocol"] != PROTOCOL or not 0 <= row["validation_accuracy"] <= 1
                or any(row[k] != item[k] for k in item) or any(k.startswith("test_") for k in row)):
            raise ValueError("Invalid validation-only score")
        for name, filename in (("weights","weights.pt"),("metadata","metadata.json")):
            if row["origin"][name] != file_fingerprint(root / stage / item["task_id"] / filename):
                raise ValueError("Source changed since independent validation")
        digest.update(hashlib.sha256(payload).digest())
        entries.append(item)
        rows.append(row)
    variants, choices = select_variants(rows, entries, 1)
    if len(variants) != 20*len(METHODS) or len(choices) != len(METHODS):
        raise ValueError("Incomplete equal-budget validation search")
    group = choices[0]["group"]
    scores = {v["method_label"]:v["validation_accuracy"]["mean"] for v in choices}
    margins = {group:ranking(scores)["margin"]}
    selection = dict(stage="moons_revalidated_confirm", eval_draws=100,
                     screening_seeds=list(SCREEN_SEEDS), selected_variants=choices, validation_trials=variants,
                     sources={s:file_fingerprint(root/(s+".jsonl")) for s in ("screen","refine","balanced")},
                     validation_proposed_margins=margins, revalidation_protocol=PROTOCOL,
                     score_hashes_sha256=digest.hexdigest(), selection_rule="Highest mean independent validation accuracy, then original checkpoint validation CE, then variant ID")
    save_frozen(root/"confirm-revalidated.selection.json", selection)
    return write_manifest(root/"confirm-revalidated.jsonl", confirmation_entries(selection))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score","freeze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    args=parser.parse_args()
    if args.command=="score":
        score(args.root,args.index)
    else:
        print(json.dumps(freeze(args.root)))


if __name__=="__main__":
    main()
