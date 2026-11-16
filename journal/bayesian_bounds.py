"""Evaluate both unweighted risk bounds for every declared Gaussian checkpoint.

Reads frozen full-risk evaluations and checkpoint tensors; it never trains or
opens a dataset. The confidence family covers all supplied manifests together.
Values before clipping are retained so a vacuous bound remains visible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from .run_experiment import bounded_loss_description, digest, normalize_config, save_json
from .theorem1 import channel_shift_diagnostic, gaussian_kl, joint_complexity, theorem1_terms


EVALUATION_FIELDS = {"device", "threads", "batch_size", "download", "eval_draws", "eval_seed"}
TRAINING_RUNTIME_FIELDS = {"device", "threads"}


def allocation(delta, runs, draws):
    if (not 0 < delta < 1 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                for x in (runs, draws))):
        raise ValueError("Require 0<delta<1 and positive integer run/draw counts")
    theorem_delta, mc_delta = .8 * delta / (2 * runs), .2 * delta / runs
    return dict(total_failure=delta, simultaneous_runs=runs, simultaneous_theorems=2,
                failure_per_theorem_run=theorem_delta, mc_failure_per_run=mc_delta,
                union_failure=2 * runs * theorem_delta + runs * mc_delta,
                empirical_mc_upper_radius=math.sqrt(math.log(1 / mc_delta) / (2 * draws)))


def corrected_bounds(kl, config, empirical, confidence):
    """The risk RHS uses the loss range, a fresh MC radius, shift, and raw terms."""
    if not math.isfinite(kl) or kl < 0 or not 0 <= empirical <= 1:
        raise ValueError("Require finite nonnegative KL and bounded empirical risk")
    kwargs = dict(n=config["train_size"], m=config["m"],
                  delta=confidence["failure_per_theorem_run"],
                  perfect_training=config["train_channel"]["kind"] == "perfect")
    terms = {key: value.item() for key, value in
             theorem1_terms(torch.tensor(kl, dtype=torch.float64), **kwargs).items()}
    joint = joint_complexity(torch.tensor(kl, dtype=torch.float64), **kwargs).item()
    shift = channel_shift_diagnostic(config["deployment_channel"], config["train_channel"], config["width"])
    empirical_upper = min(1., empirical + confidence["empirical_mc_upper_radius"])

    def result(complexity):
        value = empirical_upper + complexity + shift["channel_shift_complexity"]
        return dict(rhs_uncapped=value, population_risk_upper=min(1., value), nonvacuous=value < 1.)

    return dict(posterior_kl_float64=kl, channel_complexity=terms["channel_complexity"],
                sample_complexity=terms["sample_complexity"], joint_complexity=joint,
                shift=shift, empirical_bounded_risk_upper=empirical_upper,
                theorem_1=result(sum(terms.values())), theorem_2=result(joint))


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def collect_trials(campaigns):
    """Require one complete test result per declared trial, including every seed."""
    selected, seen = [], set()
    for manifest, root in campaigns:
        entries = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not entries or len({item["task_id"] for item in entries}) != len(entries):
            raise ValueError("Manifest must declare distinct nonempty training trials")
        rows = json.loads((Path(root) / "test_results.json").read_text())
        by_id = {row["task_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != {item["task_id"] for item in entries}:
            raise ValueError(f"Incomplete or duplicated test-result coverage for {manifest}")
        for entry in entries:
            row = by_id[entry["task_id"]]
            identity = str((Path(root) / row["task_id"]).resolve())
            if identity in seen or row.get("test_status") != "complete":
                raise ValueError("A declared checkpoint is duplicated or has no successful test evaluation")
            seen.add(identity)
            expected = normalize_config(entry["config"])
            actual = normalize_config(row["config"])
            if any(actual[key] != expected[key] for key in expected if key not in TRAINING_RUNTIME_FIELDS):
                raise ValueError("Test result differs from the declared training configuration")
            if actual["predictor"] != "gaussian":
                raise ValueError("The all-Bayesian report requires predictor='gaussian' for every method")
            selected.append((entry, row, Path(root)))
    if not selected:
        raise ValueError("No trials declared")
    return selected


def observation(entry, row, root, runs, delta):
    checkpoint = root / entry["task_id"]
    metadata_path, weights_path = checkpoint / "metadata.json", checkpoint / "weights.pt"
    evaluation_path = Path(row["test_report"])
    if not evaluation_path.is_file():
        evaluation_path = checkpoint / "test_full" / "evaluation.json"
    before = {name: fingerprint(path) for name, path in
              (("metadata", metadata_path), ("weights", weights_path), ("evaluation", evaluation_path))}
    metadata = json.loads(metadata_path.read_text())
    report = json.loads(evaluation_path.read_text())
    config, evaluated = normalize_config(metadata["config"]), normalize_config(report["evaluation_config"])
    expected = normalize_config(entry["config"])
    if any(config[key] != expected[key] for key in config if key not in TRAINING_RUNTIME_FIELDS):
        raise ValueError("Saved checkpoint differs from the declared training configuration")
    if (config["predictor"] != "gaussian" or report.get("split") != "test" or report.get("risk_scope") != "full"
            or Path(report["checkpoint"]).name != entry["task_id"]
            or any(config[key] != evaluated[key] for key in config if key not in EVALUATION_FIELDS)):
        raise ValueError("Bounds require the declared Gaussian checkpoint and its full test-risk evaluation")
    recorded_sources = report.get("checkpoint_source_fingerprints")
    if recorded_sources != {key: before[key] for key in ("metadata", "weights")}:
        raise ValueError("Evaluation source fingerprints do not match the frozen checkpoint")
    validation_seeds = ({config["eval_seed"]} if config["validation_seed_policy"] == "fixed" else
                        {config["eval_seed"] + epoch * 1009 for epoch in range(config["epochs"])})
    if evaluated["eval_seed"] in validation_seeds:
        raise ValueError("Final risk evaluation requires a fresh MC stream after validation selection")
    empirical, heldout = report["empirical_gibbs_mc"], report["heldout_gibbs_mc"]
    if (config["training_loss"] != "bounded_ce_smooth" or empirical["channel_source"] != "bank"
            or empirical["examples"] != config["train_size"] or heldout["channel_source"] != "deployment"
            or empirical["bounded_loss_definition"] != heldout["bounded_loss_definition"]
            or empirical["bounded_loss_definition"] != bounded_loss_description(evaluated)
            or empirical.get("predictor", {}).get("stochastic") is not True
            or heldout.get("predictor", {}).get("stochastic") is not True
            or empirical["draws"] != heldout["draws"] or empirical["draws"] != evaluated["eval_draws"]):
        raise ValueError("Empirical and deployment risks must use the same bounded Gibbs loss")
    # Recompute KL from the saved tensors in double precision. Method labels
    # and tuned eta/L2 strengths never enter this calculation.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    saved_bank_hash = digest({key: state["bank"][key].tolist() for key in ("gain", "noise")})
    if saved_bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError("Saved channel bank hash differs from checkpoint metadata")
    verification = report.get("checkpoint_verification", {})
    if (verification.get("saved_channel_bank_hash") != saved_bank_hash
            or verification.get("bank_hash_matches_metadata") is not True
            or any(verification.get("split_hash_matches", {}).get(key) is not True
                   for key in ("train_indices_hash", "validation_indices_hash"))):
        raise ValueError("Full evaluation must verify both saved data splits and the saved channel bank")
    kl = gaussian_kl(state["model"]["mu"].double(), state["model"]["rho"].double(), config["prior_std"]).item()
    confidence = allocation(delta, runs, empirical["draws"])
    bounds = corrected_bounds(kl, config, empirical["bounded_loss"], confidence)
    if before != {name: fingerprint(path) for name, path in
                  (("metadata", metadata_path), ("weights", weights_path), ("evaluation", evaluation_path))}:
        raise ValueError("A frozen source changed during bound calculation")
    return dict(task_id=entry["task_id"], group=entry["group"], method_label=entry["method_label"],
                seed=config["seed"], dataset=config["dataset"], stage=entry["stage"],
                m=config["m"], n=config["train_size"], train_channel=config["train_channel"],
                deployment_channel=config["deployment_channel"], config=config,
                accuracy=heldout["accuracy"], test_bounded_risk=heldout["bounded_loss"],
                empirical_bounded_risk=empirical["bounded_loss"], confidence_allocation=confidence,
                source_fingerprints=before, evaluation=str(evaluation_path.resolve()), **bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", nargs=2, action="append", required=True, metavar=("MANIFEST", "ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=.05)
    args = parser.parse_args()
    trials = collect_trials(args.campaign)
    rows = [observation(entry, row, root, len(trials), args.delta) for entry, row, root in trials]
    definitions = dict(predictor="One diagonal-Gaussian Gibbs posterior for every method, including ERM/L2.",
        risk="Expected smooth bounded loss -log(.01+.99*p_y)/log(100); test risk is a finite held-out estimate.",
        bounds="Both unweighted theorem RHSs apply to each trained posterior, independent of its training objective. Report uncapped RHSs and clipped bounds.",
        confidence="Overall delta split 80% across both theorem/run events and 20% across empirical MC/run events; full-loader replicates are the MC sample size.",
        assumptions="Conditional on the supplied theorems, IID data and independently sampled channel banks, and the fixed data-independent prior. Numerical evaluation does not verify the theorem proofs.",
        shift="Additional uniform bounded-loss TV/Pinsker shift; perfect-to-continuous noisy shift can be the trivial value one.",
        scope="Bounds concern normalized smooth Gibbs loss, not classification error or unbounded cross entropy.")
    save_json(args.output / "risk_bounds.json", dict(schema_version=1, total_runs=len(rows),
              delta=args.delta, definitions=definitions, rows=rows))
    fields = ["dataset", "group", "method_label", "seed", "n", "m", "accuracy", "test_bounded_risk",
              "empirical_bounded_risk", "posterior_kl_float64", "channel_complexity", "sample_complexity",
              "joint_complexity", "shift", "t1_rhs", "t1_bound", "t1_nonvacuous", "t2_rhs", "t2_bound", "t2_nonvacuous"]
    with (args.output / "risk_bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {**row, "shift": row["shift"]["channel_shift_complexity"]}
            for name, key in (("t1", "theorem_1"), ("t2", "theorem_2")):
                flat.update({name + "_rhs": row[key]["rhs_uncapped"], name + "_bound": row[key]["population_risk_upper"],
                             name + "_nonvacuous": row[key]["nonvacuous"]})
            writer.writerow(flat)
    print(json.dumps(dict(runs=len(rows), output=str(args.output.resolve()),
                         nonvacuous_t1=sum(row["theorem_1"]["nonvacuous"] for row in rows),
                         nonvacuous_t2=sum(row["theorem_2"]["nonvacuous"] for row in rows))))


if __name__ == "__main__":
    main()
