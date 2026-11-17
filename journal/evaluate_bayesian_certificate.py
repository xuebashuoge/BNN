"""Fresh bank-only certificates for the separate 48-checkpoint IID toy family.

All training and the saved twenty-draw full test evaluation must be complete.
This command opens only the training/validation population during fresh Monte
Carlo, restores a frozen checkpoint, and never falls back to training. Bounds
concern normalized smooth Gibbs loss; nonvacuity is an observed outcome.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import json
import math
from pathlib import Path

import torch

from .bayesian_bounds import (allocation, collect_trials, corrected_bounds, fingerprint,
                              observation)
from .build_bayesian_campaign import CHANNELS
from .build_bayesian_certificate import diagnostic_entries
from .campaign import completed_trial
from .certify_checkpoint import verify_sampler_source
from .run_experiment import (bounded_loss_description, check_device, digest, evaluate, load_data,
                             make_loader, make_models, normalize_config, save_json)
from .stage_bayesian_checkpoints import verify_training_sources
from .theorem1 import gaussian_kl


FAMILY_RUNS = 48
FAMILY_DELTA = .025
DEFAULT_DRAWS = 1000
SEED_OFFSET = 5000011
FULL_TEST_DRAWS = 20
METHODS = {"ERM": "erm", "L2": "l2", "T1": "proposed", "T2": "joint"}
LABELS = {f"{method}-{regime}" for method in METHODS for regime in ("Matched", "Perfect")}
SEEDS = {76101, 76102, 76103}


def declared_family(manifest, root):
    """Require all prespecified methods, laws and seeds before any new MC."""
    manifest, root = Path(manifest).resolve(), Path(root).resolve()
    trials = collect_trials([(manifest, root)])
    if len(trials) != FAMILY_RUNS:
        raise ValueError("The small-model certificate family requires exactly 48 declared checkpoints")
    prescribed = {entry["task_id"]: normalize_config(entry["config"]) for entry in diagnostic_entries()}
    grouped = defaultdict(lambda: defaultdict(set))
    for entry, row, _ in trials:
        config = normalize_config(entry["config"])
        label = entry["method_label"]
        channel_name = entry.get("channel_name")
        regime = "perfect" if label.endswith("-Perfect") else "matched"
        if (entry.get("stage") != "bayes_certificate" or label not in LABELS
                or channel_name not in CHANNELS or config["dataset"] != "moons"
                or config["architecture"] != "mlp" or config["predictor"] != "gaussian"
                or config["moons_sampler"] != "iid" or config["width"] != 8
                or config["train_size"] != 512 or config["val_size"] != 512 or config["m"] != 2000
                or config["training_loss"] != "bounded_ce_smooth" or config["loss_epsilon"] != .01
                or config["prior_std"] != 1. or config["method"] != METHODS[label.split("-")[0]]
                or config["seed"] not in SEEDS or config["device"] != "cpu"
                or config["deployment_channel"] != CHANNELS[channel_name]
                or config["train_channel"] != ({"kind": "perfect"} if regime == "perfect" else CHANNELS[channel_name])):
            raise ValueError("A checkpoint is outside the prespecified IID width-eight certificate family")
        if prescribed.get(entry["task_id"]) != config:
            raise ValueError("Certificate checkpoint differs from the complete prespecified training configuration")
        if config["seed"] in grouped[channel_name][label]:
            raise ValueError("Duplicate method/law/seed in the certificate family")
        grouped[channel_name][label].add(config["seed"])
        actual = normalize_config(row["config"])
        summary = completed_trial(root / entry["task_id"], actual)
        if summary is None:
            raise ValueError("All declared checkpoints must have complete matching training records")
    if (set(grouped) != set(CHANNELS) or any(set(methods) != LABELS for methods in grouped.values())
            or any(seeds != SEEDS for methods in grouped.values() for seeds in methods.values())):
        raise ValueError("The certificate family requires both laws and every paired seed for all eight labels")
    return trials


def protocol_identity(manifest, root, trials, draws, delta):
    allocation(delta, len(trials), draws)
    if delta != FAMILY_DELTA:
        raise ValueError("This separate certificate family has a fixed failure budget delta=0.025")
    return dict(schema_version=2, simultaneous_runs=FAMILY_RUNS, delta=delta, draws=draws,
                seed_offset=SEED_OFFSET, full_test_draws=FULL_TEST_DRAWS,
                manifest_fingerprint=fingerprint(manifest),
                complete_test_results_fingerprint=fingerprint(Path(root) / "test_results.json"),
                task_ids=[entry["task_id"] for entry, _, _ in trials],
                evaluator_source_fingerprint=fingerprint(Path(__file__)))


def retain_protocol(output, protocol):
    """Exclusive creation permits bounded independent --index processes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "certificate_family.json"
    # Lock only protocol publication. Distinct row MC processes run separately.
    with (output / ".certificate_family.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError("Certificate output belongs to different draws, family, budget or source files")
        else:
            save_json(path, protocol)
    return digest(protocol)


def source_paths(entry, row, root):
    checkpoint = Path(root) / entry["task_id"]
    evaluation = Path(row["test_report"])
    if not evaluation.is_file():
        evaluation = checkpoint / "test_full" / "evaluation.json"
    paths = {name: checkpoint / filename for name, filename in
             (("metadata", "metadata.json"), ("weights", "weights.pt"), ("history", "history.json"),
              ("task_config", "task_config.json"), ("manifest_entry", "manifest_entry.json"))}
    paths["evaluation"] = evaluation
    return checkpoint, paths


def fresh_seed(config, full_test_config):
    seed = config["eval_seed"] + SEED_OFFSET
    validation = ({config["eval_seed"]} if config["validation_seed_policy"] == "fixed" else
                  {config["eval_seed"] + epoch * 1009 for epoch in range(config["epochs"])})
    used = {value + offset for value in validation | {full_test_config["eval_seed"]} for offset in (0, 1)}
    if {seed, seed + 1} & used:
        raise ValueError("Certificate Monte Carlo streams overlap validation or full-test streams")
    return seed


def validate_empirical(empirical, config, draws):
    replicates = empirical.get("replicate_statistics", [])
    if (empirical.get("channel_source") != "bank" or empirical.get("draws") != draws
            or empirical.get("examples") != 512 or empirical.get("predictor", {}).get("stochastic") is not True
            or empirical.get("bounded_loss_definition") != bounded_loss_description(config)
            or empirical.get("posterior_rule") != "one weight draw fixed across entire loader per replicate"
            or len(replicates) != draws):
        raise ValueError("Certificate MC must use complete training-loader Gibbs replicates on the saved bank")
    for metric in ("bounded_loss", "accuracy", "cross_entropy"):
        values = [replicate.get(metric, math.nan) for replicate in replicates]
        if (any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                or (metric != "cross_entropy" and value > 1) for value in values)
                or not math.isclose(empirical.get(metric, math.nan), math.fsum(values) / draws,
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Certificate Monte Carlo means or replicate statistics are invalid")


def evaluate_one(entry, row, root, output, protocol, family_hash, threads=2, resume=False):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    checkpoint, paths = source_paths(entry, row, root)
    before = {name: fingerprint(path) for name, path in paths.items()}
    # This validates full-test provenance, the declared Gaussian configuration,
    # fresh test stream, loss, saved-bank hash and both recorded data splits.
    full = observation(entry, row, root, FAMILY_RUNS, protocol["delta"])
    metadata = json.loads(paths["metadata"].read_text())
    if metadata.get("git_dirty") is not False:
        raise ValueError("Certificate source checkpoint must record a clean committed training checkout")
    training_sources = verify_training_sources(metadata["git_sha"])
    sampler_matches = verify_sampler_source(metadata["git_sha"])
    config = normalize_config(metadata["config"])
    full_report = json.loads(paths["evaluation"].read_text())
    if full_report["heldout_gibbs_mc"]["draws"] != FULL_TEST_DRAWS:
        raise ValueError("The frozen full-test evaluation must contain exactly twenty Gibbs draws")
    if (full["accuracy"] != row["test_accuracy"]
            or full_report["heldout_gibbs_mc"]["cross_entropy"] != row["test_cross_entropy"]):
        raise ValueError("Frozen full-test metrics differ from the complete campaign test results")
    for name in ("task_config", "manifest_entry"):
        source_config = json.loads(paths[name].read_text())
        if name == "manifest_entry":
            if source_config.get("task_id") != entry["task_id"]:
                raise ValueError("Saved manifest entry differs from the declared checkpoint")
            source_config = source_config["config"]
        if normalize_config(source_config) != config:
            raise ValueError("Saved training configuration differs from checkpoint metadata")
    seed = fresh_seed(config, full_report["evaluation_config"])
    request = dict(family_sha256=family_hash, source_fingerprints=before, draws=protocol["draws"],
                   delta=protocol["delta"], simultaneous_runs=FAMILY_RUNS, seed=seed, threads=threads,
                   training_source_verification=training_sources, sampler_source_matches=sampler_matches)
    row_path = Path(output) / "rows" / (entry["task_id"] + ".json")
    state = torch.load(paths["weights"], map_location="cpu", weights_only=True)
    kl = gaussian_kl(state["model"]["mu"].double(), state["model"]["rho"].double(), config["prior_std"]).item()
    if not math.isfinite(kl) or kl < 0:
        raise ValueError("Saved Gaussian posterior KL must be finite and nonnegative")
    confidence = allocation(protocol["delta"], FAMILY_RUNS, protocol["draws"])
    if row_path.exists():
        if not resume:
            raise FileExistsError("Certificate row already exists; use --resume to validate its exact provenance")
        result = json.loads(row_path.read_text())
        if result.get("request") != request:
            raise ValueError("Existing certificate row has different family, draws, budget, source hashes or streams")
        fixed_fields = ("task_id", "group", "method_label", "seed", "dataset", "stage", "m", "n",
                        "train_channel", "deployment_channel", "config", "accuracy", "test_bounded_risk")
        if (any(result.get(key) != full[key] for key in fixed_fields)
                or result.get("source_fingerprints") != before
                or result.get("full_test_draws") != FULL_TEST_DRAWS
                or result.get("certificate_draws") != protocol["draws"]
                or result.get("certificate_evaluation_seed") != seed
                or result.get("full_test_empirical_bounded_risk") != full["empirical_bounded_risk"]
                or result.get("full_test_signed_gap_decomposition_mc") != full_report["signed_gap_decomposition_mc"]):
            raise ValueError("Existing certificate row disagrees with the frozen full-test identity or metrics")
        validate_empirical(result["certificate_empirical_gibbs_mc"], config, protocol["draws"])
        bounds = corrected_bounds(kl, config, result["empirical_bounded_risk"], confidence)
        if (result["empirical_bounded_risk"] != result["certificate_empirical_gibbs_mc"]["bounded_loss"]
                or result["confidence_allocation"] != confidence
                or any(result.get(key) != value for key, value in bounds.items())):
            raise ValueError("Existing certificate row disagrees with its saved posterior or MC evidence")
    else:
        evaluation_config = {**config, "device": "cpu", "threads": threads,
                             "eval_draws": protocol["draws"], "eval_seed": seed}
        check_device(evaluation_config)
        torch.set_num_threads(threads)
        # No official test data or independent toy test sample is opened here.
        train_data, _, test_data, input_dim, classes, split = load_data(evaluation_config, include_test=False)
        if test_data is not None:
            raise ValueError("Bank-only certificate evaluation unexpectedly opened test data")
        for key in ("train_indices_hash", "validation_indices_hash"):
            if split[key] != metadata["split"][key]:
                raise ValueError(f"Recreated data split differs from checkpoint: {key}")
        model, bank = make_models(evaluation_config, input_dim, classes)
        regenerated_bank = digest({key: value.tolist() for key, value in bank.state_dict().items()})
        model.load_state_dict(state["model"], strict=True)
        bank.load_state_dict(state["bank"], strict=True)
        saved_bank = digest({key: value.tolist() for key, value in bank.state_dict().items()})
        if regenerated_bank != saved_bank or saved_bank != metadata["training_channel_bank_hash"]:
            raise ValueError("Saved and regenerated independent channel banks differ")
        empirical = evaluate(model, bank, make_loader(train_data, evaluation_config), evaluation_config,
                             draws=protocol["draws"], seed=seed, source="bank")
        validate_empirical(empirical, config, protocol["draws"])
        bounds = corrected_bounds(kl, config, empirical["bounded_loss"], confidence)
        result = {**full, **bounds, "schema_version": 2, "request": request,
                  "width": config["width"], "channel_name": entry["channel_name"],
                  "training_regime": "perfect" if config["train_channel"]["kind"] == "perfect" else "matched",
                  "empirical_bounded_risk": empirical["bounded_loss"],
                  "certificate_empirical_gibbs_mc": empirical,
                  "certificate_evaluation_config": evaluation_config,
                  "certificate_evaluation_seed": seed, "certificate_draws": protocol["draws"],
                  "full_test_draws": FULL_TEST_DRAWS, "confidence_allocation": confidence,
                  "full_test_empirical_bounded_risk": full["empirical_bounded_risk"],
                  "full_test_signed_gap_decomposition_mc": full_report["signed_gap_decomposition_mc"],
                  "source_fingerprints": before,
                  "verification": dict(training_sources=training_sources, sampler_source_matches=sampler_matches,
                      train_indices_hash=split["train_indices_hash"], validation_indices_hash=split["validation_indices_hash"],
                      saved_bank_hash=saved_bank, regenerated_bank_hash_matches=True, actual_saved_bank_used=True,
                      test_data_opened_for_certificate=False),
                  "certificate_report": str(row_path.resolve()), "certificate_evaluation": str(row_path.resolve()),
                  "diagnostic_scope": "Separate small-model bound demonstration; excluded from the main accuracy ranking and no nonvacuity guarantee"}
        if before != {name: fingerprint(path) for name, path in paths.items()}:
            raise ValueError("A frozen checkpoint or full-test source changed during fresh MC")
        if verify_training_sources(metadata["git_sha"]) != training_sources:
            raise ValueError("Training source files changed during fresh MC")
        save_json(row_path, result)
    if before != {name: fingerprint(path) for name, path in paths.items()}:
        raise ValueError("A frozen checkpoint or full-test source changed during certificate validation")
    return result


def aggregate(trials, root, output, protocol, family_hash, threads=2):
    rows = []
    for entry, row, _ in trials:
        path = Path(output) / "rows" / (entry["task_id"] + ".json")
        if not path.is_file():
            raise ValueError("Aggregate requires all 48 complete, validated certificate rows")
        result = evaluate_one(entry, row, root, output, protocol, family_hash, threads, resume=True)
        rows.append({**result, "certificate_report_fingerprint": fingerprint(path)})
    definitions = dict(
        family="Separate prespecified 48-checkpoint width-eight IID two-moons diagnostic; all eight matched/perfect Gaussian method labels and three paired seeds for each law.",
        risk="Expected normalized smooth bounded Gibbs loss -log(.01+.99*p_y)/log(100). Full-test accuracy is a twenty-draw estimate; fresh bank-only MC supplies the empirical risk for certificates.",
        confidence="Family delta=0.025: 80% split over two theorem/run events and 20% over empirical MC/run events. Independence across methods is not required. Full-loader draws are the MC sample size.",
        overall_budget="The main all-dataset bound family must separately use delta=0.025 for an overall union failure budget of at most 0.05.",
        bounds="Both theorem RHSs are unweighted, use float64 Gaussian KL, the family-specific fresh empirical MC upper radius, and the bounded-loss deployment shift. Uncapped RHSs remain visible.",
        assumptions="Conditional on the supplied theorem statements, IID data, independently sampled channel banks, and the fixed data-independent Gaussian prior; numerical code does not verify theorem proofs.",
        selection="No new training or test-based selection; no checkpoint or seed is omitted because of its accuracy or bound.",
        scope="Small-model demonstration is separate from the main accuracy target. These bounds concern smooth Gibbs loss, not classification error or unbounded cross entropy. Nonvacuity is not promised.")
    report = dict(schema_version=2, total_runs=FAMILY_RUNS, delta=protocol["delta"],
                  draws=protocol["draws"], family_sha256=family_hash,
                  family_protocol=protocol, definitions=definitions, rows=rows)
    save_json(Path(output) / "risk_bounds.json", report)
    fields = ["task_id", "channel_name", "training_regime", "method_label", "seed", "n", "m", "width", "accuracy",
              "test_bounded_risk", "empirical_bounded_risk", "empirical_bounded_risk_upper", "posterior_kl_float64",
              "channel_complexity", "sample_complexity", "joint_complexity", "mc_radius", "shift",
              "t1_rhs", "t1_bound", "t1_nonvacuous", "t2_rhs", "t2_bound", "t2_nonvacuous"]
    with (Path(output) / "risk_bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {**row, "mc_radius": row["confidence_allocation"]["empirical_mc_upper_radius"],
                    "shift": row["shift"]["channel_shift_complexity"]}
            for prefix, key in (("t1", "theorem_1"), ("t2", "theorem_2")):
                flat.update({prefix + "_rhs": row[key]["rhs_uncapped"], prefix + "_bound": row[key]["population_risk_upper"],
                             prefix + "_nonvacuous": row[key]["nonvacuous"]})
            writer.writerow(flat)
    return report


def run(manifest, root, output, draws=DEFAULT_DRAWS, delta=FAMILY_DELTA, threads=2,
        index=None, resume=False, aggregate_only=False):
    trials = declared_family(manifest, root)
    protocol = protocol_identity(manifest, root, trials, draws, delta)
    family_hash = retain_protocol(output, protocol)
    if aggregate_only:
        if index is not None:
            raise ValueError("--aggregate-only and --index are mutually exclusive")
        return aggregate(trials, root, output, protocol, family_hash, threads)
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(trials):
            raise ValueError("Certificate index is outside the declared 48-checkpoint family")
        entry, row, _ = trials[index]
        return evaluate_one(entry, row, root, output, protocol, family_hash, threads, resume)
    for entry, row, _ in trials:
        evaluate_one(entry, row, root, output, protocol, family_hash, threads, resume)
    return aggregate(trials, root, output, protocol, family_hash, threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Complete small-model training campaign root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--delta", type=float, default=FAMILY_DELTA)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--index", type=int, help="Evaluate one row; externally bound concurrency to four processes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Validate all saved rows without any new MC")
    args = parser.parse_args()
    result = run(args.manifest, args.root, args.output, args.draws, args.delta,
                 args.threads, args.index, args.resume, args.aggregate_only)
    print(json.dumps(dict(output=str(args.output.resolve()), task_id=result.get("task_id"),
                         total_runs=result.get("total_runs"), index=args.index,
                         draws=args.draws, delta=args.delta)))


if __name__ == "__main__":
    main()
