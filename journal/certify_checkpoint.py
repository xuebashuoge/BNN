"""Certify one selected IID-toy Gibbs posterior using bounded loss; inference only.

The setup count covers post-hoc selection among a declared finite collection
of data/channel-bank experiments. The fresh Monte Carlo evaluation must take
place after selecting the checkpoint; choosing among several certificates
using their Monte Carlo outcomes requires a further confidence allocation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import subprocess

import torch

from .run_experiment import (digest, evaluate, load_data, make_loader,
                             make_models, normalize_config, sample_moons_iid, save_json)
from .theorem1 import channel_shift_diagnostic, gaussian_kl, joint_complexity, theorem1_terms


def confidence_allocation(delta: float, setups: int, draws: int) -> dict:
    if not 0 < delta < 1 or setups < 1 or int(setups) != setups or draws < 1 or int(draws) != draws:
        raise ValueError("Require 0<delta<1, a positive integer setup count and positive integer draws")
    theorem_count = 2
    posterior_failure = .8 * delta / (theorem_count * setups)
    mc_failure = .2 * delta
    return dict(total_failure=delta, simultaneous_setups=int(setups), simultaneous_theorems=theorem_count,
                posterior_failure_per_theorem_setup=posterior_failure,
                posterior_union_failure=theorem_count * setups * posterior_failure,
                monte_carlo_failure=mc_failure,
                empirical_mc_upper_radius=math.sqrt(math.log(1 / mc_failure) / (2 * draws)))


def bank_hash(bank) -> str:
    return digest({"gain": bank.gain.cpu().tolist(), "noise": bank.noise.cpu().tolist()})


def verify_sampler_source(saved_sha: str) -> dict:
    """Check the saved revision uses the same IID-data generation functions.

    Old metadata hashes split indices, not feature values. Comparing the sampler
    source in addition to configuration and index hashes prevents silently using
    a different toy population after later code changes. No saved code executes.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{40}", saved_sha):
        raise ValueError("Checkpoint needs a valid recorded Git SHA for sampler verification")
    source = subprocess.check_output(
        ["git", "show", f"{saved_sha}:journal/run_experiment.py"],
        text=True, cwd=Path(__file__).resolve().parent)
    parsed = ast.parse(source)
    previous = {node.name: node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    matches = {}
    for function in (sample_moons_iid, load_data):
        current = ast.parse(inspect.getsource(function)).body[0]
        matches[function.__name__] = (function.__name__ in previous and
            ast.dump(previous[function.__name__], include_attributes=False) ==
            ast.dump(current, include_attributes=False))
    if not all(matches.values()):
        raise ValueError(f"Data-generation source differs from checkpoint revision: {matches}")
    return matches


def certify(checkpoint: Path, output: Path, draws: int, delta: float,
            simultaneous_setups: int, seed: int, threads: int) -> dict:
    allocation = confidence_allocation(delta, simultaneous_setups, draws)
    if threads < 1:
        raise ValueError("threads must be positive")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = normalize_config({**metadata["config"], "device": "cpu", "threads": threads})
    if config["dataset"] != "moons" or config["moons_sampler"] != "iid":
        raise ValueError("This proof-oriented certificate requires an IID two-moons checkpoint")
    if config["width"] != 8:
        raise ValueError("This certificate's declared mechanism candidate family has width eight")
    if config["training_loss"] != "bounded_ce_smooth":
        raise ValueError("This certificate requires the campaign's smooth bounded loss")
    if config["method"] in {"erm", "l2"}:
        raise ValueError("A deterministic Dirac predictor has no finite KL to the Gaussian prior")
    sampler_matches = verify_sampler_source(metadata["git_sha"])
    torch.set_num_threads(threads)
    train_data, _, _, input_dim, classes, split = load_data(config, include_test=False)
    for key in ("train_indices_hash", "validation_indices_hash"):
        if split[key] != metadata["split"][key]:
            raise ValueError(f"Recreated data split differs from checkpoint: {key}")
    model, bank = make_models(config, input_dim, classes)
    regenerated_bank_hash = bank_hash(bank)
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    loaded_bank_hash = bank_hash(bank)
    if loaded_bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError("Saved channel-bank buffers differ from the recorded bank hash")
    empirical = evaluate(model, bank, make_loader(train_data, config), config,
                         draws=draws, seed=seed, source="bank")
    # Closed-form diagonal-Gaussian KL computed in float64. All certificate
    # terms are unweighted, regardless of coefficients used during training.
    kl = gaussian_kl(model.mu.detach().double(), model.rho.detach().double(), config["prior_std"])
    if not torch.isfinite(kl) or kl < 0:
        raise ValueError("Checkpoint Gaussian KL is not finite and nonnegative")
    terms = theorem1_terms(kl, n=config["train_size"], m=config["m"],
                           delta=allocation["posterior_failure_per_theorem_setup"],
                           perfect_training=config["train_channel"]["kind"] == "perfect")
    channel_term, sample_term = (terms[key].item() for key in ("channel_complexity", "sample_complexity"))
    joint_term = joint_complexity(kl, n=config["train_size"], m=config["m"],
                                 delta=allocation["posterior_failure_per_theorem_setup"],
                                 perfect_training=config["train_channel"]["kind"] == "perfect").item()
    shift = channel_shift_diagnostic(config["deployment_channel"], config["train_channel"], config["width"])
    radius = allocation["empirical_mc_upper_radius"]
    empirical_upper_uncapped = empirical["bounded_loss"] + radius
    empirical_upper = min(1., empirical_upper_uncapped)
    shift_value = shift["channel_shift_complexity"]
    separate_uncapped = empirical_upper_uncapped + channel_term + sample_term + shift_value
    joint_uncapped = empirical_upper_uncapped + joint_term + shift_value
    uncapped = min(separate_uncapped, joint_uncapped)
    report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_git_sha=metadata["git_sha"],
        certificate_type="Theorems 1/2 bounded-loss Gibbs population risk, with theorem/setup-union and fresh-MC confidence",
        loss_definition=empirical["bounded_loss_definition"], evaluation_seed=seed,
        posterior_kl_float64=kl.item(), confidence_allocation=allocation,
        empirical_gibbs_mc=empirical, empirical_bounded_risk_upper=empirical_upper,
        empirical_bounded_risk_upper_uncapped=empirical_upper_uncapped,
        channel_complexity=channel_term, sample_complexity=sample_term,
        joint_complexity=joint_term,
        channel_shift=shift, population_risk_upper_uncapped=uncapped,
        population_risk_upper=min(1., uncapped), nonvacuous=uncapped < 1.,
        theorem_1=dict(population_risk_upper_uncapped=separate_uncapped,
                       population_risk_upper=min(1., separate_uncapped), nonvacuous=separate_uncapped < 1.,
                       channel_complexity=channel_term, sample_complexity=sample_term),
        theorem_2=dict(population_risk_upper_uncapped=joint_uncapped,
                       population_risk_upper=min(1., joint_uncapped), nonvacuous=joint_uncapped < 1.,
                       joint_complexity=joint_term),
        bound_selection="Minimum of two simultaneously valid theorem bounds, using one shared fresh MC evaluation",
        candidate_family=dict(dataset="iid_two_moons", width=8, fixed_prior_std=config["prior_std"],
                              fixed_loss_epsilon=config["loss_epsilon"],
                              training_channel=config["train_channel"], simultaneous_setups=simultaneous_setups,
                              description="Declared mechanism n/m/bank-seed family only; other architectures and search campaigns excluded"),
        verification=dict(sampler_source_matches_checkpoint=sampler_matches,
            train_indices_hash=split["train_indices_hash"], validation_indices_hash=split["validation_indices_hash"],
            saved_bank_hash=loaded_bank_hash,
            regenerated_bank_hash_matches=regenerated_bank_hash == loaded_bank_hash,
            actual_saved_bank_used=True),
        scope=["Bound concerns expected normalized smooth loss over posterior weights and fresh deployment data/channels.",
            "It is not a direct numerical upper bound on classification error, unbounded CE, or each sampled deployed network.",
            "Conditional on correctness of the supplied Theorems 1/2 and their IID/data-independent-prior assumptions; their proofs are not verified by this numerical calculation.",
            "Declared setup count must include every considered data/channel-bank, sample-size and training-law setup with the fixed prior, architecture and loss.",
            "Uniformity over the posterior permits algorithm/hyperparameter/initialization selection within each covered setup without another algorithm multiplicity factor.",
            "The checkpoint must be selected before this fresh certificate Monte Carlo stream; selecting among repeated MC certificates needs another union allocation."])
    save_json(output / "certificate.json", report)
    print(json.dumps({"output": str(output / "certificate.json"), "draws": draws,
                      "kl": kl.item(), "empirical_bounded_loss_mc": empirical["bounded_loss"],
                      "mc_radius": radius, "channel_complexity": channel_term, "sample_complexity": sample_term,
                      "joint_complexity": joint_term,
                      "channel_shift": shift["channel_shift_complexity"],
                      "theorem_1_risk_upper": report["theorem_1"]["population_risk_upper"],
                      "theorem_2_risk_upper": report["theorem_2"]["population_risk_upper"],
                      "risk_upper": report["population_risk_upper"], "nonvacuous": report["nonvacuous"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--simultaneous-setups", type=int, default=21)
    parser.add_argument("--seed", type=int, default=991234,
                        help="Fresh certificate MC seed, chosen after fixing the checkpoint")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (args.output / "certificate.json").exists():
        parser.error("Certificate output already exists; use a fresh directory")
    certify(args.checkpoint, args.output, args.draws, args.delta,
            args.simultaneous_setups, args.seed, args.threads)


if __name__ == "__main__":
    main()
