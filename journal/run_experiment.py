"""Isolated journal experiments; legacy scripts and checkpoints are unchanged.

Train selects a checkpoint using a held-out validation split and never opens
the official test set. Evaluate loads a saved checkpoint without a training
fallback. All reported risk estimates use Gibbs posterior draws fixed across
the complete evaluation loader. Bounded losses are diagnostics, not automatic
population-risk certificates; raw CE training is a tunable surrogate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

try:
    from .theorem1 import (ChannelBank, GaussianMLP, channel_shift_diagnostic, gaussian_kl, joint_complexity,
                           theorem1_terms, validate_channel)
except ImportError:
    from theorem1 import (ChannelBank, GaussianMLP, channel_shift_diagnostic, gaussian_kl, joint_complexity,
                          theorem1_terms, validate_channel)


DEFAULTS = dict(dataset="mnist", method="proposed", width=128, train_size=55000,
                val_size=5000, batch_size=256, epochs=30, lr=.001,
                eta_channel=.01, eta_data=.1, prior_std=1., posterior_std=.01,
                m=100, seed=0, data_seed=20261110, channel_seed=1, eval_seed=2,
                eval_draws=10, val_draws=1, clip_ce=10., delta=.05, l2=1e-4,
                device="cpu", threads=1, download=False, moon_noise=.3,
                train_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1},
                deployment_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1})
METHODS = {"erm", "l2", "stochastic_erm", "pac_bayes", "proposed", "joint"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_config(path: str | None) -> dict:
    supplied = {} if path is None else json.loads(Path(path).read_text())
    return normalize_config(supplied)


def normalize_config(supplied: dict) -> dict:
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    config["train_channel"] = validate_channel(config["train_channel"])
    config["deployment_channel"] = validate_channel(config["deployment_channel"])
    if config["method"] not in METHODS or config["dataset"] not in {"moons", "mnist", "fashionmnist"}:
        raise ValueError("Unknown method or dataset")
    for key in ("width", "batch_size", "epochs", "m", "threads", "eval_draws", "val_draws"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["train_size"] <= 1 or config["val_size"] <= 0:
        raise ValueError("Require train_size>1 and val_size>0")
    if min(config["clip_ce"], config["prior_std"], config["posterior_std"], config["lr"]) <= 0:
        raise ValueError("Loss cap, scales, and learning rate must be positive")
    if not 0 < config["delta"] < 1 or min(config["eta_channel"], config["eta_data"], config["l2"]) < 0:
        raise ValueError("Require 0<delta<1 and nonnegative regularization coefficients")
    if config["m"] <= 1 and config["train_channel"]["kind"] != "perfect":
        raise ValueError("Nonperfect training requires m>1")
    return config


def load_data(config: dict, include_test: bool = False):
    """Return train/validation/(optional test), shape and reproducible split IDs."""
    if config["dataset"] == "moons":
        from sklearn.datasets import make_moons
        count = config["train_size"] + config["val_size"]
        x, y = make_moons(n_samples=count, noise=config["moon_noise"],
                          random_state=config["data_seed"] % 2**32)
        full = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        test = None
        if include_test:
            x, y = make_moons(n_samples=10000, noise=config["moon_noise"],
                              random_state=(config["data_seed"] + 1) % 2**32)
            test = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        input_dim, classes = 2, 2
        transform_description = "float32 raw coordinates"
    else:
        from torchvision import datasets, transforms
        root = os.environ.get("DATASET")
        if not root:
            raise RuntimeError("DATASET must point to the dataset directory")
        dataset_class = datasets.MNIST if config["dataset"] == "mnist" else datasets.FashionMNIST
        # Fixed transforms avoid estimating normalization using the validation/test data.
        transform = transforms.ToTensor()
        full = dataset_class(root=root, train=True, transform=transform, download=config["download"])
        test = dataset_class(root=root, train=False, transform=transform, download=config["download"]) if include_test else None
        input_dim, classes = 784, 10
        transform_description = "ToTensor float32 [0,1]; no fitted normalization"
    if config["train_size"] + config["val_size"] > len(full):
        raise ValueError("train_size + val_size exceeds available training examples")
    permutation = torch.randperm(len(full), generator=torch.Generator().manual_seed(config["data_seed"]))
    val_indices = permutation[:config["val_size"]].tolist()
    train_indices = permutation[config["val_size"]:config["val_size"] + config["train_size"]].tolist()
    split = {"train_indices_hash": digest(train_indices), "validation_indices_hash": digest(val_indices),
             "data_seed": config["data_seed"], "transform": transform_description,
             "train_size": len(train_indices), "validation_size": len(val_indices),
             "test_policy": "official test never loaded during training; independent moons test"}
    return (Subset(full, train_indices), Subset(full, val_indices), test,
            input_dim, classes, split)


def make_loader(data, config: dict, shuffle: bool = False):
    return DataLoader(data, batch_size=config["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(config["seed"] + 71),
                      num_workers=0, pin_memory=config["device"].startswith("cuda"))


def make_models(config: dict, input_dim: int, classes: int):
    torch.manual_seed(config["seed"])
    stochastic = config["method"] not in {"erm", "l2"}
    model = GaussianMLP(input_dim, config["width"], classes, stochastic,
                        config["posterior_std"]).to(config["device"])
    bank = ChannelBank(config["width"], config["m"], config["train_channel"],
                       config["channel_seed"]).to(config["device"])
    return model, bank


def check_device(config: dict):
    device = torch.device(config["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use device='cpu' or 'cuda' for reproducible posterior generators")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable on this host")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device {device.index} is unavailable")


def complexities(model, config: dict) -> dict:
    if not model.stochastic:
        return {}
    kl = gaussian_kl(model.mu, model.rho, config["prior_std"])
    kwargs = dict(n=config["train_size"], m=config["m"], delta=config["delta"],
                  perfect_training=config["train_channel"]["kind"] == "perfect")
    terms = theorem1_terms(kl, **kwargs)
    return {"kl": kl, **terms, "joint_complexity": joint_complexity(kl, **kwargs)}


@torch.no_grad()
def evaluate(model, bank, loader, config: dict, draws: int, seed: int,
             source: str = "deployment") -> dict:
    """Independent full-loader Gibbs draws; do not resample W per minibatch."""
    model.eval()
    weight_generator = torch.Generator(device=config["device"]).manual_seed(seed)
    channel_generator = torch.Generator().manual_seed(seed + 1)
    replicates = []
    for _ in range(draws):
        theta = model.sample(weight_generator)
        loss_sum, bounded_sum, correct, count = 0., 0., 0, 0
        for x, y in loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            logits = model(x, theta, bank, source, config["deployment_channel"], channel_generator)
            losses = F.cross_entropy(logits, y, reduction="none")
            loss_sum += losses.sum().item()
            bounded_sum += losses.clamp(max=config["clip_ce"]).sum().item() / config["clip_ce"]
            correct += (logits.argmax(-1) == y).sum().item()
            count += len(y)
        replicates.append(dict(cross_entropy=loss_sum / count,
                               bounded_loss=bounded_sum / count, accuracy=correct / count))
    result = {key: float(np.mean([rep[key] for rep in replicates])) for key in replicates[0]}
    result.update(draws=draws, examples=count, channel_source=source,
                  bounded_loss_definition=f"min(CE,{config['clip_ce']})/{config['clip_ce']}",
                  posterior_rule="one weight draw fixed across entire loader per replicate",
                  replicate_statistics=replicates,
                  mc_note="Finite-data Gibbs Monte Carlo estimates, not population-risk certificates",
                  bounded_loss_mc_upper_radius_95pct=math.sqrt(math.log(20) / (2 * draws)))
    return result


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(config: dict, split: dict) -> dict:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          cwd=Path(__file__).resolve().parent).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                                 cwd=Path(__file__).resolve().parent).strip())
    except (OSError, subprocess.SubprocessError):
        git_sha, git_dirty = "unavailable", None
    return dict(config=config, config_hash=digest(config), split=split,
                git_sha=git_sha, git_dirty=git_dirty, torch_version=torch.__version__,
                python_version=platform.python_version(), platform=platform.platform(),
                prior="fixed N(0,prior_std^2 I), independent of all data and channel banks",
                training_objective="raw cross entropy + tuned complexity surrogate",
                theoretical_loss="min(CE,clip_ce)/clip_ce in [0,1]; sigma_data^2=sigma_channel^2=1/4",
                certificate_note="Complexity terms alone are not certificates. Risk estimates need MC/error control; perfect-to-noisy deployment needs a separately justified channel-shift term.")


def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if platform.system() == "Darwin" else 1024)


def train(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        old_config = json.loads((output / "metadata.json").read_text())["config"]
        if old_config != config:
            raise ValueError("Output contains a different experiment; use a fresh output directory")
    train_data, val_data, _, input_dim, classes, split = load_data(config, include_test=False)
    train_loader, val_loader = make_loader(train_data, config, True), make_loader(val_data, config)
    model, bank = make_models(config, input_dim, classes)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
    channel_generator = torch.Generator().manual_seed(config["channel_seed"] + 109)
    metadata = provenance(config, split)
    metadata.update(input_dim=input_dim, classes=classes, parameter_count=model.d,
                    training_channel_bank_hash=digest({"gain": bank.gain.cpu().tolist(),
                                                       "noise": bank.noise.cpu().tolist()}))
    save_json(output / "metadata.json", metadata)
    best, history = float("inf"), []
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        ce_sum, samples = 0., 0
        for x, y in train_loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            optimizer.zero_grad(set_to_none=True)
            theta = model.sample()
            logits = model(x, theta, bank, "bank", generator=channel_generator)
            ce = F.cross_entropy(logits, y)
            terms = complexities(model, config)
            penalty = ce.new_zeros(())
            if config["method"] == "l2":
                penalty = config["l2"] * model.mu.square().sum()
            elif config["method"] in {"proposed", "pac_bayes"}:
                penalty = config["eta_data"] * terms["sample_complexity"]
                if config["method"] == "proposed":
                    penalty = penalty + config["eta_channel"] * terms["channel_complexity"]
            elif config["method"] == "joint":
                penalty = config["eta_data"] * terms["joint_complexity"]
            loss = ce + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            ce_sum += ce.item() * len(y)
            samples += len(y)
        validation = evaluate(model, bank, val_loader, config, config["val_draws"],
                              config["eval_seed"] + epoch * 1009)
        if not math.isfinite(validation["cross_entropy"]):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch + 1}")
        record = dict(epoch=epoch + 1, training_cross_entropy=ce_sum / samples,
                      validation=validation, epoch_seconds=time.perf_counter() - epoch_start,
                      final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()})
        history.append(record)
        if validation["cross_entropy"] < best:
            best = validation["cross_entropy"]
            state = dict(model=model.state_dict(), bank=bank.state_dict())
            torch.save(state, output / "weights.pt.tmp")
            (output / "weights.pt.tmp").replace(output / "weights.pt")
            metadata["selected_epoch"] = epoch + 1
            metadata["selected_validation_cross_entropy"] = best
            save_json(output / "metadata.json", metadata)
        save_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch + 1, "validation_ce": validation["cross_entropy"],
                          "validation_accuracy": validation["accuracy"],
                          "epoch_seconds": record["epoch_seconds"]}), flush=True)
    metadata.update(total_seconds=time.perf_counter() - start, peak_rss_mb=peak_memory_mb(),
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated() / 1024**2
                                            if config["device"].startswith("cuda") else None))
    save_json(output / "metadata.json", metadata)
    print(json.dumps({"output": str(output), "selected_epoch": metadata["selected_epoch"],
                      "seconds": metadata["total_seconds"], "peak_rss_mb": metadata["peak_rss_mb"]}), flush=True)


def evaluate_checkpoint(checkpoint: Path, overrides_path: str | None, output: Path, split_name: str):
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"].copy()
    overrides = {} if overrides_path is None else json.loads(Path(overrides_path).read_text())
    allowed = {"deployment_channel", "eval_draws", "eval_seed", "device", "threads", "batch_size", "download"}
    if set(overrides) - allowed:
        raise ValueError(f"Evaluation overrides restricted to {sorted(allowed)}")
    config.update(overrides)
    config = normalize_config(config)
    check_device(config)
    torch.set_num_threads(config["threads"])
    data = load_data(config, include_test=split_name == "test")
    train_data, val_data, test_data, input_dim, classes, split = data
    if split["train_indices_hash"] != metadata["split"]["train_indices_hash"]:
        raise ValueError("Training data split no longer matches checkpoint")
    model, bank = make_models(config, input_dim, classes)
    state = torch.load(checkpoint / "weights.pt", map_location=config["device"], weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    selected = {"train": train_data, "validation": val_data, "test": test_data}[split_name]
    loader = make_loader(selected, config)
    result = evaluate(model, bank, loader, config, config["eval_draws"], config["eval_seed"])
    # Every quantity is evaluated at the same frozen posterior. Identical seeds
    # share posterior draws; fresh-channel comparisons on the same data also
    # share underlying standardized channel draws whenever channel kinds match.
    empirical = evaluate(model, bank, make_loader(train_data, config), config,
                         config["eval_draws"], config["eval_seed"], source="bank")
    training_config = {**config, "deployment_channel": config["train_channel"]}
    same_data_fresh_training = evaluate(model, bank, make_loader(train_data, config), training_config,
                                        config["eval_draws"], config["eval_seed"])
    heldout_fresh_training = evaluate(model, bank, loader, training_config,
                                      config["eval_draws"], config["eval_seed"])
    signed = {}
    for metric in ("cross_entropy", "bounded_loss"):
        channel = same_data_fresh_training[metric] - empirical[metric]
        data_gap = heldout_fresh_training[metric] - same_data_fresh_training[metric]
        shift = result[metric] - heldout_fresh_training[metric]
        signed[metric] = dict(channel_generalization=channel, data_generalization=data_gap,
                              channel_shift=shift, sum=channel + data_gap + shift,
                              observed_gap=result[metric] - empirical[metric])
    report = dict(checkpoint=str(checkpoint.resolve()), split=split_name, evaluation_config=config,
                  selected_epoch=metadata.get("selected_epoch"), empirical_gibbs_mc=empirical,
                  same_training_data_fresh_training_channel_gibbs_mc=same_data_fresh_training,
                  heldout_fresh_training_channel_gibbs_mc=heldout_fresh_training,
                  heldout_gibbs_mc=result,
                  signed_gap_decomposition_mc=signed,
                  channel_shift_diagnostic=channel_shift_diagnostic(config["deployment_channel"],
                                                                   config["train_channel"], config["width"]),
                  bounded_gap_mc=result["bounded_loss"] - empirical["bounded_loss"],
                  final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()},
                  certificate_note=metadata["certificate_note"])
    save_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "validate"], default="validate")
    parser.add_argument("--config", help="JSON config; evaluate accepts only evaluation overrides")
    parser.add_argument("--output", type=Path, default=Path("results/journal"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args()
    if args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate")
        evaluate_checkpoint(args.checkpoint, args.config, args.output, args.split)
    else:
        config = read_config(args.config)
        if args.mode == "validate":
            print(json.dumps(config, indent=2))
            return
        torch.set_num_threads(config["threads"])
        check_device(config)
        train(config, args.output)


if __name__ == "__main__":
    main()
