"""Train and decompose finite-sample Bayesian bounds for the two channel laws."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .c1_full_fading import perfect_shift
from .campaign import completed_trial
from .run_experiment import (bounded_losses_from_ce, check_device, evaluate, file_fingerprint,
                            load_data, make_loader, make_models, normalize_config, save_json, train)
from .theorem1 import channel_shift_diagnostic, gaussian_kl, theorem1_terms

DATASETS = {"moons": "Two moons", "mnist": "MNIST", "fashionmnist": "Fashion-MNIST",
            "cifar10": "CIFAR-10"}
FAMILIES = {"fading": "Correlated Gaussian fading", "erasure": "Feature erasure"}
REGIMES = ("perfect", "matched", "mismatch")
FAMILY_SIZE = (len(DATASETS) + 1) * len(FAMILIES) * len(REGIMES)
SEED = 85101
COMPACT_PROFILES = ("certificate", "validation")
COMPACT_PROTOCOL = Path(__file__).with_name("configs") / "full_fading_compact_validation.json"


def design(args):
    if args.fallback and args.dataset != "moons":
        raise ValueError("The compact fallback is a two-moons experiment")
    if not args.fallback and args.fallback_profile != "certificate":
        raise ValueError("The validation profile applies to the compact fallback")
    deploy = (dict(kind="full_fading", rx_correlation=.25, tx_correlation=.25, noise_std=1.)
              if args.family == "fading" else dict(kind="outage", outage_prob=.5))
    training = deploy.copy()
    if args.regime == "perfect":
        training = dict(kind="perfect")
    elif args.regime == "mismatch":
        if args.family == "fading":
            training["rx_correlation"] = .20
        else:
            training["outage_prob"] = .45
    dataset = args.dataset
    config = dict(dataset=dataset, architecture="cnn_spatial3" if dataset == "cifar10" else "mlp",
                  method="proposed", predictor="gaussian", width=16 if dataset == "moons" else 128,
                  train_size={"moons": 800, "cifar10": 45000}.get(dataset, 55000),
                  val_size=200 if dataset == "moons" else 5000,
                  train_channel=training, deployment_channel=deploy, m=args.m,
                  training_loss="bounded_ce_smooth", loss_epsilon=.01, prior_std=1.,
                  posterior_std=.003, eta_data=.01, eta_channel=.01, lr=.003,
                  batch_size=128 if dataset == "moons" else 512,
                  epochs=35 if dataset == "moons" else 12, val_draws=2,
                  selection_metric="cross_entropy", seed=SEED, channel_seed=SEED+1,
                  eval_seed=SEED+2, device=args.device, threads=args.threads)
    if args.fallback:
        config.update(architecture="bounded_mlp", width=8, epochs=100, lr=.01,
                      bounded_feature_norm=1., bounded_receiver_norm=.14,
                      prior_std=.2, posterior_std=.1, posterior_mean_clip=.15, rho_lr_scale=0.,
                      eta_data=1., eta_channel=1.)
        if args.fallback_profile == "validation":
            protocol = json.loads(COMPACT_PROTOCOL.read_text())
            config.update(protocol["pilot"]["fixed_overrides"], **protocol["confirmation"])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    return normalize_config(config)


def case_path(root, dataset, family, regime, fallback, profile="certificate"):
    cohort = "paper"
    if fallback:
        cohort = "compact_validation" if profile == "validation" else "compact"
    return root / cohort / dataset / family / regime


@torch.no_grad()
def empirical_mc(model, bank, train_data, config, draws, compact):
    """Average independent bounded observations of the Cartesian bank risk.

    Each primary observation shares one posterior draw across a minibatch of
    independently sampled data/channel indices. For the 42-parameter fallback
    one weight realization per triple is efficient. Hoeffding counts batches
    in the first case and individual triples in the fallback.
    """
    model.eval()
    cluster_size = 1 if compact else min(256, config["batch_size"])
    seed = config["eval_seed"] + 77557
    weights = torch.Generator(device=config["device"]).manual_seed(seed)
    channel_device = config["device"] if config["train_channel"]["kind"] == "full_fading" else "cpu"
    channels = torch.Generator(device=channel_device).manual_seed(seed+1)
    data_rng = torch.Generator().manual_seed(seed+2)
    indices = torch.randint(len(train_data), (draws * cluster_size,), generator=data_rng).tolist()
    batches = DataLoader(Subset(train_data, indices),
                         batch_size=config["batch_size"] if compact else cluster_size,
                         pin_memory=config["device"].startswith("cuda"))
    total = 0.
    for x, target in batches:
        x, target = (item.to(config["device"], non_blocking=True) for item in (x, target))
        if compact:
            logits = model.forward_iid_weights(x, bank, weights, channels)
        else:
            logits = model(x, model.sample(weights), bank, "bank", generator=channels)
        losses = F.cross_entropy(logits, target, reduction="none")
        total += bounded_losses_from_ce(losses, config).double().sum().item()
    return dict(mean=total / len(indices), iid_observations=draws,
                examples_per_observation=cluster_size, total_evaluations=len(indices), seed=seed)


def risk_decomposition(config, model, empirical, family_size=FAMILY_SIZE):
    # Allocate a theorem and an empirical MC event to each declared cell.
    failure = config["delta"] / (2 * family_size)
    radius = math.sqrt(math.log(1 / failure) / (2 * empirical["iid_observations"]))
    kl = gaussian_kl(model.mu.double(), model.rho.double(), config["prior_std"]).item()
    perfect = config["train_channel"]["kind"] == "perfect"
    terms = theorem1_terms(torch.tensor(kl, dtype=torch.float64), n=config["train_size"],
                           m=config["m"], delta=failure, perfect_training=perfect)
    if perfect:
        shift = perfect_shift(config["deployment_channel"], config["width"], model,
                              config["loss_epsilon"])
    else:
        diagnostic = channel_shift_diagnostic(config["deployment_channel"],
                                              config["train_channel"], config["width"])
        divergence = diagnostic["channel_kl_deployment_to_training"]
        if divergence is None or not math.isfinite(divergence):
            raise ValueError("Matched and mismatch regimes require finite analytic channel KL")
        shift = dict(method="T1 channel KL", analytic=True,
                     upper=math.sqrt(max(0., divergence) / 2), channel_kl=divergence)
    components = dict(empirical=empirical["mean"], mc=min(radius, 1. - empirical["mean"]),
                      sample=terms["sample_complexity"].item(),
                      channel=terms["channel_complexity"].item(), shift=shift["upper"])
    uncapped = sum(components.values())
    return dict(theory="Corollary 1" if perfect else "Theorem 1", posterior_kl=kl,
                confidence=dict(family_failure=config["delta"], family_size=family_size,
                                failure_each_theorem_and_mc=failure),
                empirical_mc_radius=radius, components=components, channel_shift=shift,
                uncapped_risk_upper=uncapped, nonvacuous=uncapped < 1.)


def run(args):
    config = design(args)
    profile = args.fallback_profile
    supplemental = profile == "validation"
    protocol_fingerprint = file_fingerprint(COMPACT_PROTOCOL) if supplemental else None
    case = case_path(args.output, args.dataset, args.family, args.regime, args.fallback, profile)
    checkpoint = case / "checkpoint"
    report_path = case / "bound.json"
    draws = args.empirical_draws or (10000 if args.fallback else 128)
    if report_path.is_file():
        cached = json.loads(report_path.read_text())
        if (cached["config"] != config or cached["empirical"]["iid_observations"] != draws
                or cached["test"]["draws"] != args.test_draws
                or cached.get("compact_selection_protocol") != protocol_fingerprint):
            raise ValueError("Output already contains a different experiment")
        if cached["checkpoint_weights"] != file_fingerprint(checkpoint / "weights.pt"):
            raise ValueError("Saved model changed after evaluation")
        return cached
    torch.set_num_threads(config["threads"])
    check_device(config)
    if not completed_trial(checkpoint, config):
        train(config, checkpoint)
    train_data, _, test, input_dim, classes, split = load_data(config, include_test=True)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata["split"] != split:
        raise ValueError("Data split changed since checkpoint selection")
    from .run_experiment import channel_bank_fingerprint, checkpoint_bank_state
    model, bank = make_models(config, input_dim, classes)
    path = checkpoint / "weights.pt"
    weight_fingerprint = file_fingerprint(path)
    state = torch.load(path, map_location=config["device"], weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(checkpoint_bank_state(state, config, regenerated_bank=bank))
    bank_hash = channel_bank_fingerprint(bank, metadata.get("training_channel_bank_hash_algorithm"))
    if bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError("Saved bank differs from checkpoint metadata")
    empirical = empirical_mc(model, bank, train_data, config, draws, args.fallback)
    family_size = len(FAMILIES) * len(REGIMES) if supplemental else FAMILY_SIZE
    decomposition = risk_decomposition(config, model, empirical, family_size)
    test = evaluate(model, bank, make_loader(test, config), config, args.test_draws,
                    config["eval_seed"] + 97557)
    if file_fingerprint(path) != weight_fingerprint:
        raise ValueError("Saved weights changed during evaluation")
    report = dict(dataset=args.dataset, family=args.family, regime=args.regime, fallback=args.fallback,
                  config=config, checkpoint_weights=weight_fingerprint, saved_bank_hash=bank_hash,
                  selected_epoch=metadata["selected_epoch"], empirical=empirical, test=test,
                  fallback_profile=profile, compact_selection_protocol=protocol_fingerprint,
                  **decomposition)
    save_json(report_path, report)
    return report


def draw_figure(rows, output, families):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bnn-full-fading-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    compact = rows[0]["fallback"]
    labels = {"empirical": ("Empirical Gibbs loss", "#777777"),
              "mc": ("Monte Carlo correction", "#CBD0D5"),
              "sample": ("Sample complexity", "#009E73"),
              "channel": ("Channel complexity", "#0072B2"),
              "shift": ("Channel shift", "#E69F00")}
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.5)) if compact else plt.subplots(
        2, 2, figsize=(8.4, 6.4))
    panels = (("moons", family) for family in families) if compact else (
        (dataset, families[0]) for dataset in DATASETS)
    for ax, (dataset, family) in zip(axes.flat, panels):
        cells = sorted((r for r in rows if r["dataset"] == dataset and r["family"] == family),
                       key=lambda r: REGIMES.index(r["regime"]))
        bottom = [0.] * len(cells)
        for component, (_, color) in labels.items():
            height = [r["components"][component] for r in cells]
            ax.bar(range(len(cells)), height, bottom=bottom, width=.64, color=color)
            bottom = [b+h for b, h in zip(bottom, height)]
        ax.scatter(range(len(cells)), [r["test"]["bounded_loss"] for r in cells],
                   color="black", marker="D", s=16, zorder=4)
        for position, value in enumerate(bottom):
            ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(1., color="#474747", linestyle=":", linewidth=.8)
        ax.set_ylim(0., max(1.05, max(bottom) * 1.13))
        ax.set_xticks(range(len(cells)), [r["regime"].title() for r in cells])
        ax.set_title(FAMILIES[family] if compact else DATASETS[dataset])
        ax.set_ylabel("Population-risk upper bound")
        ax.grid(axis="y", alpha=.19)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=color, label=label) for label, color in labels.values()]
    handles.append(Line2D([], [], marker="D", markersize=4, linestyle="None",
                          color="black", label="Held-out Gibbs loss"))
    figure.suptitle((f"Two moons, bounded FC (d=8), m={rows[0]['config']['m']}" if compact else
                     f"{FAMILIES[families[0]]}, paper models, m={rows[0]['config']['m']}"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91),
                  ncol=3, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .80 if compact else .83))
    for extension in ("pdf", "png"):
        figure.savefig(output.with_suffix(f".{extension}"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def aggregate(output, compact_only=False, profile="certificate"):
    if not compact_only and profile != "certificate":
        raise ValueError("Use --compact-only for the independent validation-selected confirmation")
    rows = []
    cohorts = [(True, ("moons",))] if compact_only else [(False, DATASETS), (True, ("moons",))]
    for fallback, datasets in cohorts:
        for dataset in datasets:
            for family in FAMILIES:
                for regime in REGIMES:
                    path = case_path(output, dataset, family, regime, fallback, profile) / "bound.json"
                    row = json.loads(path.read_text())
                    if (row["dataset"], row["family"], row["regime"], row["fallback"],
                            row.get("fallback_profile", "certificate")) != (
                            dataset, family, regime, fallback, profile):
                        raise ValueError(f"Case dimensions differ from the report at {path}")
                    rows.append(row)
    if len({r["config"]["m"] for r in rows}) != 1:
        raise ValueError("Use the same m in each case in this decomposition")
    stem = "compact_validation" if profile == "validation" else "compact"
    if not compact_only:
        for family in FAMILIES:
            draw_figure([r for r in rows if not r["fallback"] and r["family"] == family],
                        output / f"{family}_bound_decomposition", (family,))
    draw_figure([r for r in rows if r["fallback"]], output / f"{stem}_bound_decomposition", FAMILIES)
    report = dict(cases=len(rows), seed=rows[0]["config"]["seed"], m=rows[0]["config"]["m"],
                  compact_profile=profile,
                  paper_nonvacuous=(None if compact_only else
                                     sum(r["nonvacuous"] for r in rows if not r["fallback"])),
                  compact_nonvacuous=sum(r["nonvacuous"] for r in rows if r["fallback"]),
                  rows=rows)
    filename = f"{stem}_bound_decomposition.json" if compact_only else "bound_decomposition.json"
    save_json(output / filename, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("run")
    cell.add_argument("--dataset", choices=DATASETS, required=True)
    cell.add_argument("--family", choices=FAMILIES, required=True)
    cell.add_argument("--regime", choices=REGIMES, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--fallback", action="store_true")
    cell.add_argument("--fallback-profile", choices=COMPACT_PROFILES, default="certificate")
    cell.add_argument("--m", type=int, default=2000)
    cell.add_argument("--device", default="cpu")
    cell.add_argument("--threads", type=int, default=2)
    cell.add_argument("--epochs", type=int, help="Override epochs for a resource or smoke run")
    cell.add_argument("--empirical-draws", type=int)
    cell.add_argument("--test-draws", type=int, default=8)
    all_cells = subparsers.add_parser("aggregate")
    all_cells.add_argument("--output", type=Path, required=True)
    all_cells.add_argument("--compact-only", action="store_true")
    all_cells.add_argument("--fallback-profile", choices=COMPACT_PROFILES, default="certificate")
    args = parser.parse_args()
    if args.command == "aggregate":
        result = aggregate(args.output, args.compact_only, args.fallback_profile)
        print(json.dumps({key: val for key, val in result.items() if key != "rows"}))
    else:
        if args.test_draws < 1 or (args.empirical_draws is not None and args.empirical_draws < 1):
            parser.error("Evaluation draw counts must be positive")
        result = run(args)
        print(json.dumps({key: result[key] for key in ("dataset", "family", "regime", "fallback",
                                                      "uncapped_risk_upper", "nonvacuous")}))


if __name__ == "__main__":
    main()
