"""Build explicit CIFAR-10 resource pilots and validation-selected comparisons.

This module creates configuration manifests only. Training belongs on Spartan.
The CPU pilots establish execution/resource evidence before GPU submission;
pilot accuracies are not scientific performance results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_campaign import make_entry


CHANNELS = {
    "severe": dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.),
    "moderate": dict(kind="gaussian", gain_mean=1., gain_std=.3, noise_std=.3),
    "near_ideal": dict(kind="gaussian", gain_mean=1., gain_std=.01, noise_std=.01),
}
METHODS = (
    ("erm", "ERM-Matched"),
    ("l2", "L2-Matched"),
    ("proposed", "T1-Matched"),
    ("joint", "T2-Matched"),
    ("erm", "ERM-Perfect"),
    ("l2", "L2-Perfect"),
    ("pac_bayes", "PAC-Perfect"),
)


def base_config(channel="severe", architecture="cnn", width=128):
    return dict(dataset="cifar10", architecture=architecture, width=width,
                training_loss="bounded_ce_smooth", loss_epsilon=.01,
                selection_metric="accuracy", validation_seed_policy="fixed",
                train_size=10000, val_size=5000, epochs=30, batch_size=128,
                lr=.001, prior_std=1., posterior_std=.001,
                eta_channel=.0003, eta_data=.003, l2=1e-6, m=100,
                data_seed=20261111, eval_draws=10, val_draws=2,
                train_channel=dict(CHANNELS[channel]),
                deployment_channel=dict(CHANNELS[channel]),
                device="cpu", threads=2, download=False)


def make_cifar_entry(config, seed, channel, label, stage, role="comparison"):
    group = f"cifar10_{config['architecture']}_analog_{channel}"
    entry = make_entry(config, seed, group, label, f"cifar_{stage}", role)
    entry["channel_preset"] = channel
    entry["model_family"] = "compact split CNN" if config["architecture"] == "cnn" else "split MLP"
    return entry


def pilot_entries(seeds, channel, architecture, width, target=None):
    base = base_config(channel, architecture, width)
    is_shape = target is not None
    if is_shape:
        base.update(train_size=45000 if target == "confirm" else 10000,
                    val_size=5000, epochs=1)
    else:
        base.update(train_size=2048, val_size=512, epochs=2)
    entries = []
    for method, label in METHODS if is_shape else METHODS[:4]:
        config = {**base, "method": method}
        if label.endswith("Perfect"):
            config["train_channel"] = {"kind": "perfect"}
        for seed in seeds:
            entry = make_cifar_entry(config, seed, channel, label,
                                     f"shape_{target}" if is_shape else "pilot",
                                     role="resource_pilot")
            entry["scientific_result"] = False
            entry["cpu_pilot_target"] = target or "development"
            entries.append(entry)
    return entries


def screen_entries(seeds, channel, architecture, width):
    base = base_config(channel, architecture, width)
    variants = []
    for lr in (.001, .003):
        variants.append(({**base, "method": "erm", "lr": lr}, "ERM-Matched"))
        for coefficient in (1e-6, 1e-5):
            variants.append(({**base, "method": "l2", "lr": lr,
                              "l2": coefficient}, "L2-Matched"))
    # Cover both LR/noise combinations in each proposed family. Coefficients
    # remain deliberately weak for the much larger CNN posterior dimension.
    for lr in (.001, .003):
        for posterior_std, strength in ((.001, .0003), (.003, .001)):
            variants.append(({**base, "method": "proposed", "lr": lr,
                              "posterior_std": posterior_std,
                              "eta_channel": strength, "eta_data": 10 * strength}, "T1-Matched"))
            variants.append(({**base, "method": "joint", "lr": lr,
                              "posterior_std": posterior_std,
                              "eta_data": 2 * strength}, "T2-Matched"))
    variants.extend([
        ({**base, "method": "erm", "train_channel": {"kind": "perfect"}}, "ERM-Perfect"),
        ({**base, "method": "l2", "l2": 1e-6,
          "train_channel": {"kind": "perfect"}}, "L2-Perfect"),
        ({**base, "method": "pac_bayes", "eta_data": .01,
          "train_channel": {"kind": "perfect"}}, "PAC-Perfect"),
    ])
    entries = [make_cifar_entry(config, seed, channel, label, "screen")
               for config, label in variants for seed in seeds]
    return entries


def confirm_entries(summary_path, seeds, epochs=100, width=None):
    summary = json.loads(Path(summary_path).read_text())
    entries = []
    for selected in summary["selected_variants"]:
        if selected.get("rank", 1) != 1:
            continue
        config = dict(selected["config"])
        if config.get("dataset") != "cifar10":
            continue
        config.update(train_size=45000, val_size=5000, epochs=epochs,
                      eval_draws=10, val_draws=2, device="cpu")
        if width is not None:
            config["width"] = width
        # The channel name is part of the explicit screening group.
        channel = next((name for name in CHANNELS
                        if config["deployment_channel"] == CHANNELS[name]), None)
        if channel is None:
            raise ValueError("Screening uses an unknown channel preset")
        for seed in seeds:
            entry = make_cifar_entry(config, seed, channel, selected["method_label"], "confirm")
            entry["selected_from_validation_variant"] = selected["variant_id"]
            entry["selection_source"] = "aggregate validation accuracy across every screening seed"
            entries.append(entry)
    if not entries:
        raise ValueError("No validation-selected CIFAR-10 configurations found")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "shape-pilot", "screen", "confirm"], default="pilot")
    parser.add_argument("--target-stage", choices=["screen", "confirm"], default="screen",
                        help="Which dataset sizes the one-epoch shape pilot represents")
    parser.add_argument("--channel", choices=CHANNELS, default="severe")
    parser.add_argument("--architecture", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--width", type=int,
                        help="Feature width, default 128; confirmation preserves the selected width unless overridden")
    parser.add_argument("--seeds", help="Comma-separated distinct integer seeds")
    parser.add_argument("--epochs", type=int, help="Override stage epochs; confirmation defaults to 100")
    parser.add_argument("--from-summary", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Defaults to a stage-specific path under journal/configs")
    args = parser.parse_args()
    defaults = [4101] if args.stage in {"pilot", "shape-pilot"} else ([4101, 4102] if args.stage == "screen" else [5101, 5102, 5103])
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else defaults
    if not seeds or len(seeds) != len(set(seeds)) or (args.width is not None and args.width < 1):
        raise ValueError("Require distinct seeds and positive width")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    width = args.width or 128
    if args.output is None:
        suffix = f"shape_{args.target_stage}" if args.stage == "shape-pilot" else args.stage
        args.output = Path("journal/configs") / f"cifar_{suffix}.jsonl"
    if args.stage == "pilot":
        entries = pilot_entries(seeds, args.channel, args.architecture, width)
    elif args.stage == "shape-pilot":
        entries = pilot_entries(seeds, args.channel, args.architecture, width, args.target_stage)
    elif args.stage == "screen":
        entries = screen_entries(seeds, args.channel, args.architecture, width)
    else:
        if args.from_summary is None:
            parser.error("Confirmation requires --from-summary")
        entries = confirm_entries(args.from_summary, seeds,
                                  args.epochs if args.epochs is not None else 100, args.width)
    if args.epochs is not None and args.stage != "confirm":
        for entry in entries:
            # Rebuild IDs after a requested epoch override.
            old = entry
            config = {**entry["config"], "epochs": args.epochs}
            stage = entry["stage"].removeprefix("cifar_")
            entry.update(make_cifar_entry(config, config["seed"], args.channel,
                                         old["method_label"], stage, old["role"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries))
    plan = dict(stage=args.stage, trials=len(entries), seeds=seeds,
                dataset="cifar10", architectures=sorted({entry["config"]["architecture"] for entry in entries}),
                channel_presets=sorted({entry["channel_preset"] for entry in entries}),
                groups=sorted({entry["group"] for entry in entries}),
                data_sizes=sorted({(entry["config"]["train_size"], entry["config"]["val_size"]) for entry in entries}),
                checkpoint_selection="validation accuracy; CE tiebreak; fixed validation CRN",
                hyperparameter_selection="mean validation accuracy over every seed",
                pilot_is_scientific_result=False if args.stage in {"pilot", "shape-pilot"} else None,
                gpu_requires_completed_matching_cpu_pilot=True,
                resource_note="Measure CPU epoch time and peak RSS before choosing GPU/CPU walltime; no training was performed by this builder",
                main_baselines=["ERM", "L2"],
                perfect_specialization="PAC-Perfect is displayed as Proposed Perfect; no separate data-PAC matched competitor")
    args.output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
