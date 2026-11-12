"""Explicit full-size validation search for convolutional-receiver CIFAR learners.

Creates manifests only. Existing paired-seed checkpoints are reused as anchors;
new candidates never open the test set during hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_campaign import make_entry


CONFIGS = Path("journal/configs")
LABELS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched")
SEEDS = (7101, 7102)


def read_entries(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_entries(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in entries))


def search_entries(original):
    anchors = [entry for entry in original
               if entry["method_label"] in LABELS and entry["config"]["seed"] in SEEDS]
    if len(anchors) != 8:
        raise ValueError("Require eight original paired-seed anchors")
    base = {label: dict(next(entry["config"] for entry in anchors
                             if entry["method_label"] == label)) for label in LABELS}
    variants = [({**base["ERM-Matched"], "lr": .001, "rho_lr_scale": 1.}, "ERM-Matched"),
                ({**base["L2-Matched"], "l2": 1e-5, "rho_lr_scale": 1.}, "L2-Matched")]
    for label in ("T1-Matched", "T2-Matched"):
        for scale in (.1, .03, .01):
            variants.append(({**base[label], "rho_lr_scale": scale}, label))
        for std in (.003, .001):
            variants.append(({**base[label], "rho_lr_scale": 0., "posterior_std": std}, label))
    new = [make_entry(config, seed, "cifar10_cnn_mid_analog_severe", label, "rho_search")
           for config, label in variants for seed in SEEDS]
    for entry in new:
        entry.update(search_role="full-size validation search; no test evaluation",
                     changed_factors="posterior scale learning rate or one declared baseline alternative")
    cached = []
    for entry in anchors:
        cached.append({**entry, "stage": "rho_search_cached", "reused_training": True,
                       "reuse_source": f"results/journal_cifar_mid/confirm/{entry['task_id']}"})
    return new, cached


def resource_entries(original):
    entries = []
    for method, label, scale in (("proposed", "T1-Matched", .1), ("proposed", "T1-Matched", 0.),
                                 ("joint", "T2-Matched", .03), ("joint", "T2-Matched", 0.)):
        config = dict(next(entry["config"] for entry in original if entry["method_label"] == label))
        config.update(epochs=1, method=method, rho_lr_scale=scale)
        entry = make_entry(config, 8000, "cifar10_cnn_mid_rho_full_pilot", label,
                           f"rho_pilot_{str(scale).replace('.', '_')}", "resource_pilot")
        entry["scientific_result"] = False
        entries.append(entry)
    template = dict(original[0]["config"])
    for architecture in ("cnn_mid_relu", "cnn_rxconv", "cnn_spatial3"):
        for method, label in (("erm", "ERM-Matched"), ("proposed", "T1-Matched")):
            config = {**template, "architecture": architecture, "method": method,
                      "train_size": 2048, "val_size": 512, "epochs": 2, "rho_lr_scale": .1,
                      "posterior_std": .003, "eta_channel": .001, "eta_data": .01}
            entry = make_entry(config, 8000, f"cifar10_{architecture}_development_pilot", label,
                               "architecture_pilot", "resource_pilot")
            entry["scientific_result"] = False
            entries.append(entry)
    return entries


def confirmation_entries(summary, seeds, threads=1):
    if (summary.get("completed_trials") != summary.get("training_trials")
            or summary.get("failed_trials")):
        raise ValueError("Validation search must complete without failures")
    selected = [item for item in summary.get("selected_variants", []) if item.get("rank", 1) == 1]
    if len(selected) != 4 or {item["method_label"] for item in selected} != set(LABELS):
        raise ValueError("Require four validation-selected matched methods")
    if any(not item.get("complete_seed_set") or set(item["seeds"]) != set(SEEDS) for item in selected):
        raise ValueError("Require both paired search seeds for each selected configuration")
    entries = []
    for item in selected:
        config = dict(item["config"])
        config.update(epochs=100, train_size=45000, val_size=5000, eval_draws=10,
                      val_draws=2, device="cpu", threads=threads)
        for seed in seeds:
            entry = make_entry(config, seed, item["group"], item["method_label"], "rho_confirm")
            entry.update(selection_source=("aggregate fresh validation-only Monte Carlo accuracy over seeds 7101 and 7102"
                                          if summary.get("validation_draws") else
                                          "aggregate validation accuracy over seeds 7101 and 7102"),
                         selected_from_validation_variant=item["variant_id"])
            entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "confirm"], default="prepare")
    parser.add_argument("--original", type=Path, default=CONFIGS / "cifar_mid_confirm.jsonl")
    parser.add_argument("--from-summary", type=Path)
    parser.add_argument("--seeds", default="8101,8102,8103")
    parser.add_argument("--threads", type=int, default=1,
                        help="Common execution thread count for every fresh confirmation method")
    args = parser.parse_args()
    if args.stage == "prepare":
        original = read_entries(args.original)
        new, cached = search_entries(original)
        pilots = resource_entries(original)
        write_entries(CONFIGS / "cifar_rho_new.jsonl", new)
        write_entries(CONFIGS / "cifar_rho_all.jsonl", cached + new)
        write_entries(CONFIGS / "cifar_search_pilot.jsonl", pilots)
        plan = dict(new_training_trials=len(new), reused_training_trials=len(cached),
                    aggregate_trials=len(new)+len(cached), paired_search_seeds=list(SEEDS),
                    train_size=45000, val_size=5000, epochs=100, architecture="cnn_mid",
                    methods=list(LABELS), selection="mean validation accuracy; CE tiebreak",
                    test_evaluation_during_search=False, prior_std=1., loss_epsilon=.01,
                    data_seed=20261111, split_and_channel_bank_fixed_within_each_seed=True,
                    fixed_scale_is_positive_gaussian_not_dirac=True,
                    resource_pilots=len(pilots), proposed_goal="highest test accuracy checked only after frozen selection")
        (CONFIGS / "cifar_rho_search.plan.json").write_text(json.dumps(plan, indent=2)+"\n")
        print(json.dumps(plan, indent=2))
    else:
        if args.from_summary is None:
            parser.error("--from-summary is required")
        seeds = [int(value) for value in args.seeds.split(",")]
        if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds) & set(SEEDS):
            raise ValueError("Require three distinct fresh confirmation seeds")
        if args.threads < 1:
            raise ValueError("threads must be positive")
        entries = confirmation_entries(json.loads(args.from_summary.read_text()), seeds, args.threads)
        path = CONFIGS / "cifar_rho_confirm.jsonl"
        write_entries(path, entries)
        print(json.dumps(dict(path=str(path), trials=len(entries), seeds=seeds)))


if __name__ == "__main__":
    main()
