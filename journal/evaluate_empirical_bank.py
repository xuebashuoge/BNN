"""Evaluate a declared queue of frozen checkpoints on their saved training bank.

There is no optimizer or training fallback. Each replicate draws one posterior
weight vector and samples a bank index per training example, so it is an
unbiased [0,1]-valued Monte Carlo observation of the full Cartesian empirical
risk. Hoeffding is applied to the number of independent full-loader replicates,
not to n*m or to the number of minibatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .run_experiment import (check_device, digest, evaluate, load_data, make_loader,
                             make_models, normalize_config, save_json)
from .theorem1 import (channel_shift_diagnostic, gaussian_kl, joint_complexity,
                       theorem1_terms)


REPO = Path(__file__).resolve().parents[1]
SEED_OFFSET = 3_000_007


def confidence_allocation(delta: float, total_runs: int, draws: int) -> dict:
    if (not 0 < delta < 1 or isinstance(total_runs, bool) or isinstance(draws, bool)
            or total_runs < 1 or int(total_runs) != total_runs
            or draws < 1 or int(draws) != draws):
        raise ValueError("Require 0<delta<1 and positive integer run/draw counts")
    theorem_failure = .8 * delta / (2 * total_runs)
    mc_failure = .2 * delta / total_runs
    return dict(total_failure=delta, total_runs=int(total_runs), simultaneous_runs=int(total_runs),
                simultaneous_theorems=2,
                posterior_failure_per_theorem_run=theorem_failure,
                posterior_union_failure=2 * total_runs * theorem_failure,
                monte_carlo_failure_per_run=mc_failure,
                mc_failure_per_run=mc_failure,
                monte_carlo_union_failure=total_runs * mc_failure,
                empirical_mc_upper_radius=math.sqrt(math.log(1 / mc_failure) / (2 * draws)),
                scope="T1/T2 normalized bounded-loss bounds for every declared run; zero-one empirical loss is descriptive only")


def fingerprint(path: Path) -> dict:
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def sources(checkpoint: Path) -> dict:
    return {name: fingerprint(checkpoint / name) for name in ("metadata.json", "weights.pt")}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def load_queue(path: Path) -> list[dict]:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("The empirical-evaluation queue is empty")
    checkpoints, outputs = set(), set()
    for entry in entries:
        checkpoint = resolve_path(entry["checkpoint"])
        output = resolve_path(entry["output"])
        if checkpoint in checkpoints or output in outputs:
            raise ValueError("Each checkpoint and output may appear only once in the queue")
        if output == checkpoint:
            raise ValueError("Evaluation output must not replace the checkpoint directory")
        checkpoints.add(checkpoint)
        outputs.add(output)
    return entries


def bounds_from_empirical(model, config: dict, empirical: dict, allocation: dict) -> dict:
    """Restore all unweighted terms; deterministic models have no Gaussian KL."""
    if not model.stochastic:
        return dict(bound_status="not_applicable_deterministic_dirac",
                    final_posterior_complexities={}, theorem_1=None, theorem_2=None,
                    bound_note="The evaluated deterministic Dirac predictor has infinite KL to the continuous Gaussian prior; no zero-KL placeholder is used")
    kl = gaussian_kl(model.mu.detach().double(), model.rho.detach().double(), config["prior_std"])
    if not torch.isfinite(kl) or kl < 0:
        raise ValueError("The frozen posterior needs finite nonnegative Gaussian KL")
    kwargs = dict(n=config["train_size"], m=config["m"],
                  delta=allocation["posterior_failure_per_theorem_run"],
                  perfect_training=config["train_channel"]["kind"] == "perfect")
    terms = {key: value.item() for key, value in theorem1_terms(kl, **kwargs).items()}
    joint = joint_complexity(kl, **kwargs).item()
    shift = channel_shift_diagnostic(config["deployment_channel"], config["train_channel"], config["width"])
    empirical_upper = min(1., empirical["bounded_loss"] + allocation["empirical_mc_upper_radius"])

    def bound(complexity):
        uncapped = empirical_upper + shift["channel_shift_complexity"] + complexity
        return dict(population_risk_upper_uncapped=uncapped,
                    population_risk_upper=min(1., uncapped), nonvacuous=uncapped < 1.)

    return dict(bound_status="bounded_loss_theorem_evaluation_with_union_and_mc_corrections",
                posterior_kl_float64=kl.item(),
                final_posterior_complexities=dict(kl=kl.item(), **terms, joint_complexity=joint),
                empirical_bounded_risk_upper=empirical_upper,
                channel_shift_diagnostic=shift,
                theorem_1=bound(terms["channel_complexity"] + terms["sample_complexity"]),
                theorem_2=bound(joint),
                bound_note="Unweighted T1/T2 and separately valid [0,1]-loss TV/Pinsker shift, under the standing IID data/bank and fixed-prior assumptions. This does not assert a C1 channel-Lipschitz certificate")


def evaluate_entry(entry: dict, *, draws: int, delta: float, total_runs: int,
                   threads: int, seed_offset: int = SEED_OFFSET) -> dict:
    allocation = confidence_allocation(delta, total_runs, draws)
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    checkpoint, output = resolve_path(entry["checkpoint"]), resolve_path(entry["output"])
    before = sources(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    saved_config = metadata["config"].copy()
    if saved_config["dataset"] == "moons" and "moons_sampler" not in saved_config:
        saved_config["moons_sampler"] = "legacy_grid"
    training_config = normalize_config(saved_config)
    # Use a fresh declared stream after checkpoint/configuration selection.
    seed = training_config["eval_seed"] + seed_offset
    config = normalize_config({**training_config, "device": "cpu", "threads": threads,
                               "eval_draws": draws, "eval_seed": seed, "download": False})
    if entry.get("task_id") and checkpoint.name != entry["task_id"]:
        raise ValueError("Queue task_id differs from checkpoint directory name")
    if "seed" in entry and entry["seed"] != config["seed"]:
        raise ValueError("Queue seed differs from frozen training seed")
    if "config" in entry and normalize_config(entry["config"]) != training_config:
        raise ValueError("Queue configuration differs from the frozen checkpoint")
    check_device(config)
    torch.set_num_threads(threads)
    train_data, _, _, input_dim, classes, split = load_data(config, include_test=False)
    for key in ("train_indices_hash", "validation_indices_hash"):
        if split[key] != metadata["split"][key]:
            raise ValueError(f"Recreated data split differs from the frozen checkpoint: {key}")
    model, bank = make_models(config, input_dim, classes)
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    bank_hash = digest({"gain": bank.gain.cpu().tolist(), "noise": bank.noise.cpu().tolist()})
    if bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError("Saved training bank differs from the recorded hash")
    empirical = evaluate(model, bank, make_loader(train_data, config), config,
                         draws, seed, source="bank")
    empirical["zero_one_loss"] = 1. - empirical["accuracy"]
    for replicate in empirical["replicate_statistics"]:
        replicate["zero_one_loss"] = 1. - replicate["accuracy"]
    report = dict(schema_version=1, checkpoint=str(checkpoint), task_id=checkpoint.name,
                  seed=config["seed"], selected_epoch=metadata.get("selected_epoch"),
                  split="train", risk_scope="empirical_bank", training_config=training_config,
                  evaluation_config=config, empirical_gibbs_mc=empirical,
                  empirical_target="E_q [1/(n*m) sum_i sum_j loss(W,C_j,Z_i)] at saved S,U; bank-index MC is unbiased",
                  evaluation_seed=seed, confidence_allocation=allocation,
                  training_channel_bank_hash=bank_hash, split_hashes=split,
                  source_fingerprints=before, no_training=True,
                  **bounds_from_empirical(model, config, empirical, allocation))
    if sources(checkpoint) != before:
        raise ValueError("Checkpoint sources changed during empirical evaluation")
    save_json(output / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--index", type=int, help="Zero-based row for an array task; confidence still covers the entire queue")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET)
    args = parser.parse_args()
    entries = load_queue(args.queue)
    if args.index is not None and not 0 <= args.index < len(entries):
        parser.error("index is outside the declared queue")
    selected = entries if args.index is None else [entries[args.index]]
    for entry in selected:
        report = evaluate_entry(entry, draws=args.draws, delta=args.delta,
                                total_runs=len(entries), threads=args.threads,
                                seed_offset=args.seed_offset)
        print(json.dumps(dict(task_id=report["task_id"],
                              empirical_bounded_loss=report["empirical_gibbs_mc"]["bounded_loss"],
                              empirical_zero_one_loss=report["empirical_gibbs_mc"]["zero_one_loss"],
                              draws=args.draws, bound_status=report["bound_status"],
                              output=str(resolve_path(entry["output"]) / "evaluation.json"))), flush=True)


if __name__ == "__main__":
    main()
