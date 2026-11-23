"""Rerun the selected MNIST bank sweep and two-moons fading certificates."""

import argparse
import csv
import json
import math
from pathlib import Path
import tarfile


SELECTED = Path(__file__).resolve().parents[1] / "results" / "final_chosen_results"
MNIST = SELECTED / "data" / "mnist_bank"
BOUNDS = SELECTED / "data" / "bounds"
BOUND_CASES = ("perfect", "t1_matched", "t2_matched", "t1_mismatch", "t2_mismatch")
STUDIES = ("mnist", "bounds")


def read_json(path):
    return json.loads(path.read_text())


def mnist_entries():
    with (MNIST / "manifest.jsonl").open() as source:
        return [json.loads(line) for line in source if line.strip()]


def case_record(case):
    return read_json(BOUNDS / "fading" / case / "bound.json")


def train_once(raw_config, checkpoint):
    import torch

    from .trainer import run_experiment as trainer

    config = trainer.normalize_config({**raw_config, "transmit_unit_power": True})
    metadata_path = checkpoint / "metadata.json"
    if metadata_path.exists() and (checkpoint / "weights.pt").exists():
        metadata = read_json(metadata_path)
        if metadata["config"] != config:
            raise ValueError(f"Training configuration differs in {checkpoint}")
        if "total_seconds" in metadata:
            return config, metadata
    trainer.check_device(config)
    torch.set_num_threads(config["threads"])
    trainer.train(config, checkpoint)
    return config, read_json(metadata_path)


def assert_unit_power(result):
    statistics = [result, *result["replicate_statistics"]]
    if any(not math.isclose(row["feature_power_per_coordinate"], 1., abs_tol=2e-5)
           for row in statistics):
        raise ValueError("Measured pre-channel feature power differs from one")


def assert_original_split(metadata, original):
    for key in ("train_indices_hash", "validation_indices_hash"):
        if metadata["split"][key] != original["split"][key]:
            raise ValueError(f"Data split changed: {key}")


def run_mnist(index, output_root):
    from .trainer import run_experiment as trainer

    entry = mnist_entries()[index]
    task_id = entry["task_id"]
    archived_path = f"runs/{task_id}"
    with tarfile.open(MNIST / "run_records.tar.gz", "r:gz") as records:
        original = json.load(records.extractfile(f"{archived_path}/test_full/evaluation.json"))
        original_metadata = json.load(records.extractfile(f"{archived_path}/metadata.json"))
    checkpoint = output_root / "mnist" / archived_path
    config, metadata = train_once(entry["config"], checkpoint)
    assert_original_split(metadata, original_metadata)
    evaluation_config = original["evaluation_config"]
    overrides = {key: evaluation_config[key] for key in ("eval_seed", "eval_draws")}
    overrides_path = checkpoint / "evaluation_overrides.json"
    trainer.save_json(overrides_path, overrides)
    evaluation = trainer.evaluate_checkpoint(checkpoint, str(overrides_path), checkpoint / "test_full", "test")
    for key in ("heldout_gibbs_mc", "empirical_gibbs_mc", "same_training_data_fresh_training_channel_gibbs_mc",
                "heldout_fresh_training_channel_gibbs_mc"):
        assert_unit_power(evaluation[key])
    if evaluation["evaluation_config"] != {**config, **overrides}:
        raise ValueError(f"Unexpected evaluation configuration for {task_id}")


def c1_shift(kl, width, radius, receiver_norm, epsilon):
    """C1(ii) with the iid Gaussian channel reference and Euclidean channel metric."""
    # The smooth bounded log-loss is sqrt(2)/-log(epsilon) Lipschitz in logits.
    # Cauchy-Schwarz bounds ||(H-H')z + (N-N')|| using ||z|| <= radius.
    lipschitz = math.sqrt(2.) * receiver_norm / -math.log(epsilon) * math.hypot(radius, 1.)
    dimension = 2 * width
    noncentrality_squared = width / 4  # iid gain mean 1/2 vs. perfect gain 1
    quadratic = lipschitz / math.sqrt(dimension + noncentrality_squared)
    if quadratic >= 1:
        raise ValueError("The C1(ii) Gaussian reference MGF bound requires a quadratic parameter below one")
    # Young's inequality and the noncentral chi-squared MGF give this upper bound
    # on log E_ref exp(L * distance(channel, perfect)).
    log_mgf = (lipschitz**2 / (2 * quadratic)
               - dimension / 2 * math.log1p(-quadratic)
               + quadratic * noncentrality_squared / (2 * (1 - quadratic)))
    return dict(method="C1 second", metric="sqrt(||H-H'||_F^2 + ||N-N'||_2^2)",
                reference="P_iid", reference_kl=kl, uniform_lipschitz=lipschitz,
                feature_norm_limit=radius, receiver_norm_limit=receiver_norm,
                quadratic_parameter=quadratic, noncentrality_squared=noncentrality_squared,
                reference_dimension=dimension, reference_log_mgf_upper=log_mgf,
                analytic=True, upper=kl + log_mgf)


def empirical_bound_risk(model, bank, train_data, config, empirical):
    import torch
    from torch.nn import functional as F

    from .trainer.run_experiment import bounded_losses_from_ce

    # Each replicate independently samples one example and one channel from the
    # realized training product, together with one posterior weight vector.
    count = empirical["iid_observations"]
    if empirical["examples_per_observation"] != 1 or empirical["total_evaluations"] != count:
        raise ValueError("Expected one example, channel and weight draw per empirical observation")
    seed = empirical["seed"]
    x, y = train_data.dataset.tensors
    device = config["device"]
    x, y = x[train_data.indices].to(device), y[train_data.indices].to(device)
    # The archive records the base seed only. Adjacent seeds separate the
    # example, posterior and channel streams in the reconstructed evaluator.
    data_rng = torch.Generator(device=device).manual_seed(seed)
    weight_rng = torch.Generator(device=device).manual_seed(seed + 1)
    channel_rng = torch.Generator(device=device).manual_seed(seed + 2)
    loss_sum = 0.
    model.eval()
    with torch.no_grad():
        for start in range(0, count, config["batch_size"]):
            batch = min(count - start, config["batch_size"])
            indices = torch.randint(len(train_data), (batch,), device=device, generator=data_rng)
            logits = model.forward_iid_weights(x[indices], bank, weight_rng, channel_rng)
            losses = F.cross_entropy(logits, y[indices], reduction="none")
            loss_sum += bounded_losses_from_ce(losses, config).double().sum().item()
    return {**empirical, "mean": loss_sum / count}


def run_bound(index, output_root):
    import torch

    from .trainer import run_experiment as trainer
    from .trainer.theorem1 import channel_shift_diagnostic, gaussian_kl, joint_complexity, theorem1_terms

    case = BOUND_CASES[index]
    original = case_record(case)
    directory = output_root / "bounds" / case
    checkpoint = directory / "checkpoint"
    config, metadata = train_once(original["config"], checkpoint)
    assert_original_split(metadata, read_json(BOUNDS / "fading" / case / "metadata.json"))
    trainer.check_device(config)
    torch.set_num_threads(config["threads"])
    train, _, test, input_dim, classes, _ = trainer.load_data(config, include_test=True)
    model, bank = trainer.make_models(config, input_dim, classes)
    state = torch.load(checkpoint / "weights.pt", map_location=config["device"], weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(trainer.checkpoint_bank_state(state, config, regenerated_bank=bank))
    bank_hash_algorithm = metadata["training_channel_bank_hash_algorithm"]
    bank_hash = trainer.channel_bank_fingerprint(bank, bank_hash_algorithm)
    if bank_hash != metadata["training_channel_bank_hash"]:
        raise ValueError(f"Training bank does not match the checkpoint metadata for {case}")
    empirical = empirical_bound_risk(model, bank, train, config, original["empirical"])
    # The archived test report stores its draw count and sample count. Its
    # selected config supplies the evaluation seed.
    test_seed = config["eval_seed"]
    test_result = trainer.evaluate(model, bank, trainer.make_loader(test, config), config,
                                   original["test"]["draws"], test_seed)
    assert_unit_power(test_result)
    if test_result["examples"] != original["test"]["examples"]:
        raise ValueError(f"Held-out sample size changed for {case}")

    with torch.no_grad():
        kl = gaussian_kl(model.mu.double(), model.rho.double(), config["prior_std"])
    delta = original["confidence"]["failure_each_theorem_and_mc"]
    mc_radius = math.sqrt(math.log(1 / delta) / (2 * empirical["iid_observations"]))
    components = {"empirical": empirical["mean"], "mc": mc_radius}
    certificate = original["certificate"]
    deployment, training, width = config["deployment_channel"], config["train_channel"], config["width"]
    if certificate == "C1":
        reference = {"kind": "diagonal_fading", "rho_c": 0.}
        ref_kl = channel_shift_diagnostic(deployment, reference, width)["channel_kl_deployment_to_training"]
        shift = c1_shift(ref_kl, width, model.transmit_feature_radius(bank),
                         config["bounded_receiver_norm"], config["loss_epsilon"])
    else:
        channel = channel_shift_diagnostic(deployment, training, width)
        shift = dict(method=f"{certificate} channel KL", analytic=True,
                     upper=channel["channel_shift_complexity"],
                     channel_kl=channel["channel_kl_deployment_to_training"])
    args = (kl, config["train_size"], config["m"], delta)
    if certificate == "T2":
        components["joint"] = joint_complexity(*args).item()
    else:
        terms = theorem1_terms(*args, perfect_training=certificate == "C1")
        components.update(sample=terms["sample_complexity"].item(),
                          channel=terms["channel_complexity"].item())
    components["shift"] = shift["upper"]
    uncapped = sum(components.values())
    report = {key: original[key] for key in ("dataset", "family", "regime", "certificate", "fallback",
                                             "theory", "confidence")}
    report.update(config=config, checkpoint_weights=trainer.file_fingerprint(checkpoint / "weights.pt"),
                  saved_bank_hash=bank_hash, selected_epoch=metadata["selected_epoch"],
                  empirical=empirical, test={**test_result, "seed": test_seed}, posterior_kl=kl.item(),
                  empirical_mc_radius=mc_radius, components=components, channel_shift=shift,
                  uncapped_risk_upper=uncapped, nonvacuous=uncapped < 1.)
    trainer.save_json(directory / "bound.json", report)


def collect(output_root, publish_root):
    from .fixed_studies_plots import plot_bound, plot_mnist

    observations = []
    for entry in mnist_entries():
        checkpoint = output_root / "mnist" / "runs" / entry["task_id"]
        metadata = read_json(checkpoint / "metadata.json")
        report = read_json(checkpoint / "test_full" / "evaluation.json")
        assert_unit_power(report["heldout_gibbs_mc"])
        heldout = report["heldout_gibbs_mc"]
        observations.append(dict(method=entry["method_label"], m=entry["sweep_value"], seed=entry["config"]["seed"],
                                 test_accuracy=heldout["accuracy"], generalization_error=report["bounded_gap_mc"],
                                 seconds=metadata["total_seconds"],
                                 feature_power_per_coordinate=heldout["feature_power_per_coordinate"],
                                 weights_sha256=report["checkpoint_source_fingerprints"]["weights"]["sha256"]))
    fading = [(case, read_json(output_root / "bounds" / case / "bound.json")) for case in BOUND_CASES]
    for _, report in fading:
        assert_unit_power(report["test"])
    erasure = [(case, read_json(BOUNDS / "erasure" / case / "bound.json")) for case in BOUND_CASES]
    bank_data = publish_root / "data" / "mnist_bank"
    bank_data.mkdir(parents=True, exist_ok=True)
    with (bank_data / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)
    for case, report in fading:
        destination = publish_root / "data" / "bounds" / "fading" / case
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bound.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_mnist(observations, publish_root / "mnist_analog_finite_channel_generalization")
    plot_bound(fading, erasure, publish_root / "bound_decomposition_d8_t1_t2_both_channels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count")
    count.add_argument("study", choices=STUDIES)
    run = subparsers.add_parser("run")
    run.add_argument("study", choices=STUDIES)
    run.add_argument("--index", required=True, type=int)
    run.add_argument("--output-root", required=True, type=Path)
    aggregate = subparsers.add_parser("collect")
    aggregate.add_argument("--output-root", required=True, type=Path)
    aggregate.add_argument("--publish-root", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output_root, args.publish_root or args.output_root / "fixed_results")
        return
    count = len(mnist_entries()) if args.study == "mnist" else len(BOUND_CASES)
    if args.command == "count":
        print(count)
        return
    if not 0 <= args.index < count:
        parser.error(f"{args.study} index must be within 0–{count - 1}")
    runner = run_mnist if args.study == "mnist" else run_bound
    runner(args.index, args.output_root)


if __name__ == "__main__":
    main()
