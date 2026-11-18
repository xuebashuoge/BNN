"""Tune seven harsher-link CIFAR learners on validation and freeze an independent cohort."""
import argparse
from itertools import product
import json
from pathlib import Path

from .build_campaign import make_entry
from .campaign import select_variants
from .diagonal_fading_comparison import validation_rows
from .fading_severity import HARD
from .full_fading_comparison import METHODS, variant_grid, write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest


SCREEN_SEEDS = (173101, 173102, 173103)
CONFIRM_SEEDS = (174101, 174102, 174103, 174104, 174105)
RATE_FACTORS = (.5, 1., 1.5)
PENALTY_FACTORS = (.1, 1., 10.)
POSTERIOR_SCALE_FACTORS = (1/3, 1., 3.)


def anchors(transfer_dir):
    manifest = transfer_dir / "manifest.jsonl"
    protocol = json.loads((transfer_dir / "transfer.json").read_text())
    if (protocol["manifest"] != file_fingerprint(manifest)
            or protocol["deployment_law"] != HARD or protocol["train_law_matched"] != HARD):
        raise ValueError("Expected a frozen severe-channel transfer")
    rows = read_manifest(manifest)
    seeds = set(protocol["seeds"])
    pairs = [(r["method_label"], r["config"]["seed"]) for r in rows]
    if (len(pairs) != 7 * len(seeds) or set(pairs) != set(product(METHODS, seeds))
            or seeds.intersection(SCREEN_SEEDS + CONFIRM_SEEDS)
            or len({r["group"] for r in rows}) != 1):
        raise ValueError("Expected a complete seven-method transfer and fresh study seeds")
    first = {}
    for row in rows:
        label = row["method_label"]
        cfg = row["config"]
        expected_method, perfect = METHODS[label]
        if (cfg["dataset"] != "cifar10" or cfg["method"] != expected_method
                or cfg["deployment_channel"] != HARD
                or cfg["train_channel"] != ({"kind":"perfect"} if perfect else HARD)):
            raise ValueError(f"Unexpected CIFAR model/channel configuration in {row['task_id']}")
        original = first.setdefault(label, row)
        if original["variant_id"] != row["variant_id"]:
            raise ValueError(f"The frozen baseline varies across seeds for {label}")
    return [first[label] for label in METHODS]


def screening(transfer_dir):
    candidates = []
    for base in anchors(transfer_dir):
        config = base["config"]
        method = config["method"]
        factors = POSTERIOR_SCALE_FACTORS if method == "erm" else PENALTY_FACTORS
        for rate_factor, second_factor in product(RATE_FACTORS, factors):
            candidate = dict(config, lr=config["lr"] * rate_factor)
            if method == "erm":
                candidate["posterior_std"] = config["posterior_std"] * second_factor
            elif method == "l2":
                candidate["l2"] = config["l2"] * second_factor
            else:
                for key in ("eta_data", "eta_channel"):
                    candidate[key] = config[key] * second_factor
            for seed in SCREEN_SEEDS:
                row = make_entry(candidate, seed, base["group"], base["method_label"],
                                 "severity_cifar_screen", "severity_validation")
                row.update(family=base["family"], transfer_variant=base["variant_id"])
                candidates.append(row)
    return candidates


def select(transfer_dir, root):
    path = root / "screen.jsonl"
    declared = read_manifest(path)
    if not variant_grid(declared, screening(transfer_dir)):
        raise ValueError("Incomplete or altered severe-law validation screen")
    rows = validation_rows(root / "screen-runs", declared)
    variants, choices = select_variants(rows, declared, 1)
    source = {r["task_id"]: r for r in declared}
    confirmation = []
    for best in choices:
        original = source[best["task_ids"][0]]
        for seed in CONFIRM_SEEDS:
            row = make_entry(original["config"], seed, original["group"], original["method_label"],
                             "severity_cifar_retuned", "severity_confirmation")
            row.update(family=original["family"], transfer_variant=original["transfer_variant"],
                       selected_from_validation_variant=best["variant_id"])
            confirmation.append(row)
    protocol = dict(screening_seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
                    selection_rule="Highest mean severe validation accuracy across all screening seeds, then lowest mean validation cross-entropy, then variant ID",
                    candidates_per_method=9, validation_trials=variants, selected_variants=choices,
                    transfer=file_fingerprint(transfer_dir/"manifest.jsonl"), screen=file_fingerprint(path))
    write_once(root/"selection.json",json.dumps(protocol, indent=2,allow_nan=False)+"\n")
    return write_manifest(root/"manifest.jsonl", confirmation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("screen","select"))
    parser.add_argument("--transfer-dir",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    result=(write_manifest(args.root/"screen.jsonl",screening(args.transfer_dir))
            if args.phase=="screen" else select(args.transfer_dir,args.root))
    print(json.dumps(result))


if __name__=="__main__":
    main()
