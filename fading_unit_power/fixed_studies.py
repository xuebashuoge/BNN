"""Rerun the selected MNIST bank sweep with unit transmit power."""

import argparse
import csv
import json
import math
from pathlib import Path
import tarfile


SELECTED = Path(__file__).resolve().parents[1] / "results" / "final_chosen_results"
MNIST = SELECTED / "data" / "mnist_bank"


def read_json(path):
    return json.loads(path.read_text())


def mnist_entries():
    with (MNIST / "manifest.jsonl").open() as source:
        return [json.loads(line) for line in source if line.strip()]


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


def collect(output_root, publish_root):
    from .fixed_studies_plots import plot_mnist

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
    bank_data = publish_root / "data" / "mnist_bank"
    bank_data.mkdir(parents=True, exist_ok=True)
    with (bank_data / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)
    plot_mnist(observations, publish_root / "mnist_analog_finite_channel_generalization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count")
    count.add_argument("study", choices=("mnist",))
    run = subparsers.add_parser("run")
    run.add_argument("study", choices=("mnist",))
    run.add_argument("--index", required=True, type=int)
    run.add_argument("--output-root", required=True, type=Path)
    aggregate = subparsers.add_parser("collect")
    aggregate.add_argument("--output-root", required=True, type=Path)
    aggregate.add_argument("--publish-root", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output_root, args.publish_root or args.output_root / "fixed_results")
        return
    count = len(mnist_entries())
    if args.command == "count":
        print(count)
        return
    if not 0 <= args.index < count:
        parser.error(f"{args.study} index must be within 0–{count - 1}")
    run_mnist(args.index, args.output_root)


if __name__ == "__main__":
    main()
