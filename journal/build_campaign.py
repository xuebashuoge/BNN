"""Build explicit, reproducible journal experiment manifests without training.

Screening is exploratory. Confirmation uses configurations chosen by aggregate
validation accuracy and new initialization/channel-bank seeds. All settings,
including deliberately chosen deployment environments, remain in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def variant_config(config):
    return {key: value for key, value in config.items()
            if key not in {"seed", "channel_seed", "eval_seed", "threads", "device"}}


def base_config(dataset, channel):
    return dict(dataset=dataset, method="proposed", training_loss="bounded_ce_smooth",
                loss_epsilon=.01, selection_metric="accuracy", prior_std=1., posterior_std=.01,
                train_size=10000 if dataset == "mnist" else 800,
                val_size=5000 if dataset == "mnist" else 200,
                epochs=25 if dataset == "mnist" else 100,
                width=128 if dataset == "mnist" else 16,
                batch_size=256 if dataset == "mnist" else 128,
                m=100, lr=.001, eta_channel=.001, eta_data=.01,
                data_seed=20261110, eval_draws=5, val_draws=2,
                train_channel=channel, deployment_channel=channel,
                device="cpu", threads=2, download=False, moons_sampler="iid")


def make_entry(config, seed, group, method_label, stage, role="comparison"):
    config = {**config, "seed": seed, "channel_seed": seed + 10000, "eval_seed": seed + 20000}
    variant_id = fingerprint(variant_config(config))
    family = "proposed" if config["method"] in {"proposed", "joint"} and config["train_channel"]["kind"] != "perfect" else "baseline"
    task_id = f"{stage}_{group}_{method_label.replace('-', '_')}_{variant_id}_s{seed}"
    return dict(task_id=task_id, mode="train", config=config, group=group,
                method_label=method_label, family=family, stage=stage,
                variant_id=variant_id, role=role)


def screen_entries(seeds):
    analog = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)
    outage = dict(kind="outage", outage_prob=.5)
    entries = []
    for dataset in ("mnist", "moons"):
        group = f"{dataset}_analog"
        base = base_config(dataset, analog)
        variants = []
        for method, label in (("erm", "ERM-Matched"), ("l2", "L2-Matched"),
                              ("stochastic_erm", "StochERM-Matched"), ("pac_bayes", "PAC-Matched")):
            for index, lr in enumerate((.001, .003)):
                variants.append(({**base, "method": method, "lr": lr,
                                  "l2": (1e-5, 1e-4)[index], "eta_data": (.006, .06)[index],
                                  "posterior_std": (.003, .01)[index]}, label))
        for lr in (.001, .003):
            variants.append(({**base, "method": "erm", "lr": lr, "training_loss": "raw_ce"}, "ERM-RawCE-Matched"))
        # Four explicit configurations for each journal objective. No seed or
        # method is silently dropped when it underperforms.
        for index, strength in enumerate((.0003, .001, .003, .01)):
            lr = .001 if index % 2 == 0 else .003
            variants.append(({**base, "method": "proposed", "lr": lr,
                              "eta_channel": strength, "eta_data": strength * 10,
                              "posterior_std": .003 if index < 2 else .01}, "T1-Matched"))
            variants.append(({**base, "method": "joint", "lr": lr,
                              "eta_data": strength * 2,
                              "posterior_std": .003 if index < 2 else .01}, "T2-Matched"))
        for method, label in (("erm", "ERM-Perfect"), ("pac_bayes", "PAC-Perfect")):
            for index, lr in enumerate((.001, .003)):
                variants.append(({**base, "method": method, "lr": lr,
                                  "eta_data": (.006, .06)[index],
                                  "posterior_std": (.003, .01)[index],
                                  "train_channel": {"kind": "perfect"}}, label))
        for config, label in variants:
            entries.extend(make_entry(config, seed, group, label, "screen") for seed in seeds)
    base = base_config("moons", outage)
    for method, label in (("erm", "ERM-Matched"), ("pac_bayes", "PAC-Matched"),
                          ("proposed", "T1-Matched"), ("joint", "T2-Matched")):
        for index, lr in enumerate((.001, .003)):
            config = {**base, "method": method, "lr": lr,
                      "eta_channel": (.001, .01)[index], "eta_data": (.003, .03)[index],
                      "posterior_std": (.003, .01)[index]}
            entries.extend(make_entry(config, seed, "moons_outage", label, "screen") for seed in seeds)
    return entries


def confirm_entries(summary, seeds, groups=None, epochs=150, width=256):
    entries = []
    for selected in summary["selected_variants"]:
        if selected.get("rank", 1) != 1 or (groups and selected["group"] not in groups):
            continue
        if not groups and selected["config"]["dataset"] != "mnist":
            continue
        config = dict(selected["config"])
        config.update(epochs=epochs, width=width, eval_draws=10, val_draws=2)
        if config["dataset"] in {"mnist", "fashionmnist"}:
            config.update(train_size=55000, val_size=5000)
        for seed in seeds:
            entry = make_entry(config, seed, selected["group"], selected["method_label"], "confirm")
            entry["selected_from_validation_variant"] = selected["variant_id"]
            entry["selection_source"] = "mean validation accuracy across all screening seeds"
            entries.append(entry)
    return entries


def mechanism_entries(seeds):
    base = base_config("moons", {"kind": "outage", "outage_prob": .5})
    base.update(width=8, train_size=512, val_size=512, epochs=150)
    entries = []
    for n, m in [(512, m) for m in (5, 20, 100, 500, 2000)] + [(128, 100), (2048, 100)]:
        for method, label in (("stochastic_erm", "StochERM-Matched"), ("pac_bayes", "PAC-Matched"),
                              ("proposed", "T1-Matched"), ("joint", "T2-Matched")):
            config = {**base, "method": method, "train_size": n, "m": m}
            entries.extend(make_entry(config, seed, f"moons_outage_n{n}_m{m}", label,
                                      "mechanism", "mechanism") for seed in seeds)
    # Perfect training is reused across m; only changing n needs retraining.
    for n in (128, 512, 2048):
        for method, label in (("erm", "ERM-Perfect"), ("pac_bayes", "PAC-Perfect")):
            config = {**base, "method": method, "train_size": n, "train_channel": {"kind": "perfect"}}
            entries.extend(make_entry(config, seed, f"moons_outage_perfect_n{n}", label,
                                      "mechanism", "reference") for seed in seeds)
    return entries


def mismatch_entries(seeds):
    base = base_config("moons", {"kind": "outage", "outage_prob": .5})
    base.update(epochs=150, width=16)
    entries = []
    for probability in (0., .1, .3, .5, .7, .9):
        training = {"kind": "perfect"} if probability == 0 else {"kind": "outage", "outage_prob": probability}
        for method, label in (("stochastic_erm", "StochERM"), ("pac_bayes", "PAC"),
                              ("proposed", "T1"), ("joint", "T2")):
            # Perfect T1 and T2 collapse to PAC; avoid duplicating identical
            # objectives merely to add another labelled algorithm.
            if probability == 0 and method in {"proposed", "joint"}:
                continue
            config = {**base, "method": method, "train_channel": training}
            entries.extend(make_entry(config, seed, f"moons_outage_ptr{int(probability*100):02d}", label,
                                      "mismatch", "mismatch") for seed in seeds)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "confirm", "mechanism", "mismatch"], default="screen")
    parser.add_argument("--output", type=Path, default=Path("journal/configs/screen.jsonl"))
    parser.add_argument("--seeds", help="Comma-separated integer initialization seeds")
    parser.add_argument("--from-summary", type=Path)
    parser.add_argument("--groups", help="Comma-separated selected comparison groups")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    default_seeds = [1001, 1002] if args.stage == "screen" else [9001, 9002, 9003, 9004, 9005]
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else default_seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Require distinct, explicit seeds")
    if args.stage == "screen":
        entries = screen_entries(seeds)
    elif args.stage == "confirm":
        if args.from_summary is None:
            parser.error("confirm requires --from-summary")
        entries = confirm_entries(json.loads(args.from_summary.read_text()), seeds,
                                  args.groups.split(",") if args.groups else None, args.epochs, args.width)
    elif args.stage == "mechanism":
        entries = mechanism_entries(seeds)
    else:
        entries = mismatch_entries(seeds)
    if not entries:
        raise ValueError("No tasks selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in entries))
    plan = dict(stage=args.stage, tasks=len(entries), seeds=seeds,
                groups=sorted({item["group"] for item in entries}),
                selection="validation mean across all seeds; test-based rankings are exploratory only",
                all_trial_outcomes_retained=True)
    args.output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
