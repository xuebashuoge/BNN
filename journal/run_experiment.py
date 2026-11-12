"""Isolated journal experiments; legacy scripts and checkpoints are unchanged.

Train selects a checkpoint using a held-out validation split and never opens
the official test set. Evaluate loads a saved checkpoint without a training
fallback. All reported risk estimates use Gibbs posterior draws fixed across
the complete evaluation loader. Bounded losses are diagnostics, not automatic
population-risk certificates. Raw CE training is a tunable surrogate; the
optional bounded CE objective matches the bounded-loss theoretical empirical
term. Coefficients of one recover the separated bound's trainable objective.
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


DEFAULTS = dict(dataset="mnist", architecture="mlp", method="proposed", training_loss="raw_ce", loss_epsilon=.01,
                selection_metric="cross_entropy", validation_seed_policy="fixed", width=128, train_size=55000,
                val_size=5000, batch_size=256, epochs=30, lr=.001,
                eta_channel=.01, eta_data=.1, prior_std=1., posterior_std=.01,
                m=100, seed=0, data_seed=20261110, channel_seed=1, eval_seed=2,
                eval_draws=10, val_draws=1, clip_ce=10., delta=.05, l2=1e-4,
                device="cpu", threads=1, cpu_flush_denormal=False, rho_lr_scale=1., download=False, moon_noise=.3, moons_sampler="iid",
                train_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1},
                deployment_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1})
METHODS = {"erm", "l2", "stochastic_erm", "pac_bayes", "proposed", "joint"}
CNN_ARCHITECTURES = {"cnn", "cnn_mid", "cnn_mid_relu", "cnn_rxconv", "cnn_spatial3"}
_CPU_FLUSH_DENORMAL_SUPPORTED: bool | None = None


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
    if not isinstance(config["cpu_flush_denormal"], bool):
        raise ValueError("cpu_flush_denormal must be a boolean")
    if (not isinstance(config["rho_lr_scale"], (int, float)) or isinstance(config["rho_lr_scale"], bool)
            or not math.isfinite(config["rho_lr_scale"]) or config["rho_lr_scale"] < 0):
        raise ValueError("rho_lr_scale must be a finite nonnegative number")
    config["train_channel"] = validate_channel(config["train_channel"])
    config["deployment_channel"] = validate_channel(config["deployment_channel"])
    if config["method"] not in METHODS or config["dataset"] not in {"moons", "mnist", "fashionmnist", "cifar10"}:
        raise ValueError("Unknown method or dataset")
    if config["architecture"] not in {"mlp", *CNN_ARCHITECTURES}:
        raise ValueError("Unknown architecture")
    if config["architecture"] in CNN_ARCHITECTURES and config["dataset"] != "cifar10":
        raise ValueError("The compact CNN requires CIFAR-10 RGB inputs")
    if config["architecture"] in CNN_ARCHITECTURES - {"cnn"} and (
            not isinstance(config["width"], int) or isinstance(config["width"], bool)
            or config["width"] <= 0 or config["width"] % 16):
        raise ValueError("The middle CNN requires width to be a positive multiple of 16")
    if config["training_loss"] not in {"raw_ce", "bounded_ce", "bounded_ce_smooth"}:
        raise ValueError("training_loss must be raw_ce, bounded_ce, or bounded_ce_smooth")
    if not 0 < config["loss_epsilon"] < 1:
        raise ValueError("loss_epsilon must lie strictly between zero and one")
    if config["selection_metric"] not in {"cross_entropy", "accuracy"}:
        raise ValueError("selection_metric must be cross_entropy or accuracy")
    if config["validation_seed_policy"] not in {"fixed", "varying"}:
        raise ValueError("validation_seed_policy must be fixed or varying")
    if config["moons_sampler"] not in {"iid", "legacy_grid"}:
        raise ValueError("moons_sampler must be iid or legacy_grid")
    if config["moon_noise"] < 0:
        raise ValueError("moon_noise must be nonnegative")
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


def sample_moons_iid(n_samples: int, noise: float, seed: int):
    """IID two-arc population: Bernoulli labels, uniform angle, Gaussian noise.

    The second arc uses sklearn's conventional coordinates (1-cos(t),
    0.5-sin(t)). Unlike make_moons, neither class counts nor angles are fixed.
    """
    if n_samples <= 0 or noise < 0:
        raise ValueError("Require n_samples>0 and noise>=0")
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    angle = rng.uniform(0., math.pi, size=n_samples)
    x = np.where(labels == 0, np.cos(angle), 1. - np.cos(angle))
    y = np.where(labels == 0, np.sin(angle), .5 - np.sin(angle))
    coordinates = np.column_stack((x, y)) + rng.normal(0., noise, size=(n_samples, 2))
    return coordinates.astype(np.float32), labels


def load_data(config: dict, include_test: bool = False):
    """Return train/validation/(optional test), shape and reproducible split IDs."""
    if config["dataset"] == "moons":
        if config["moons_sampler"] == "iid":
            sampler = lambda count, seed: sample_moons_iid(count, config["moon_noise"], seed)
            sampler_description = "iid Bernoulli(1/2) label, Uniform[0,pi] angle, independent Gaussian coordinate noise"
        else:
            from sklearn.datasets import make_moons
            sampler = lambda count, seed: make_moons(n_samples=count, noise=config["moon_noise"],
                                                      random_state=seed % 2**32)
            sampler_description = "legacy sklearn make_moons fixed class counts and grid angles; not iid population draws"
        count = config["train_size"] + config["val_size"]
        x, y = sampler(count, config["data_seed"])
        full = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        test = None
        if include_test:
            x, y = sampler(10000, config["data_seed"] + 1)
            test = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        input_dim, classes = 2, 2
        transform_description = "float32 raw coordinates; " + sampler_description
    elif config["dataset"] == "cifar10":
        from torchvision import datasets
        root = os.environ.get("DATASET")
        if not root:
            raise RuntimeError("DATASET must point to the dataset directory")

        def tensor_dataset(training):
            raw = datasets.CIFAR10(root=root, train=training, download=config["download"])
            # Fixed, data-independent normalization. Materialize contiguous
            # tensors once instead of repeating PIL/ToTensor on every epoch.
            # Full training storage is ~586 MiB; no random augmentation or
            # additional training-distribution randomness is introduced.
            images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).to(
                dtype=torch.float32, memory_format=torch.contiguous_format)
            images.div_(127.5).sub_(1.)
            return TensorDataset(images, torch.tensor(raw.targets, dtype=torch.long))

        full = tensor_dataset(True)
        test = tensor_dataset(False) if include_test else None
        input_dim, classes = 3072, 10
        transform_description = "CIFAR RGB uint8 -> contiguous float32 CHW, pixel/127.5-1; no fitted normalization or augmentation"
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
    if config.get("architecture", "mlp") in CNN_ARCHITECTURES:
        from .cifar_models import GaussianMiddleSplitCNN, GaussianSplitCNN
        if config["architecture"] in {"cnn", "cnn_mid"}:
            model_class = GaussianMiddleSplitCNN if config["architecture"] == "cnn_mid" else GaussianSplitCNN
        else:
            from .cifar_search_models import CIFAR_SEARCH_MODELS
            model_class = CIFAR_SEARCH_MODELS[config["architecture"]]
        model = model_class(config["width"], classes, stochastic,
                            config["posterior_std"]).to(config["device"])
    else:
        model = GaussianMLP(input_dim, config["width"], classes, stochastic,
                            config["posterior_std"]).to(config["device"])
    if model.stochastic and config.get("rho_lr_scale", 1.) == 0:
        # A fixed positive Gaussian scale remains stochastic and has finite KL.
        model.rho.requires_grad_(False)
    bank = ChannelBank(config["width"], config["m"], config["train_channel"],
                       config["channel_seed"]).to(config["device"])
    return model, bank


def make_optimizer(model, config):
    """Separate mean/scale learning rates while preserving the legacy default."""
    scale = config.get("rho_lr_scale", 1.)
    if not model.stochastic or scale == 1.:
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
    groups = [dict(params=[model.mu], lr=config["lr"])]
    if model.rho.requires_grad and scale > 0:
        groups.append(dict(params=[model.rho], lr=config["lr"] * scale))
    return torch.optim.Adam(groups, lr=config["lr"])


def check_device(config: dict):
    global _CPU_FLUSH_DENORMAL_SUPPORTED
    device = torch.device(config["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use device='cpu' or 'cuda' for reproducible posterior generators")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable on this host")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device {device.index} is unavailable")
    if device.type == "cpu":
        # Set before creating intra-op workers: floating-point control state can
        # be thread-local. The opt-in avoids slow float32 subnormal arithmetic
        # when deterministic L2 weights decay toward zero.
        requested = config.get("cpu_flush_denormal", False)
        _CPU_FLUSH_DENORMAL_SUPPORTED = bool(torch.set_flush_denormal(requested))
        if requested and not _CPU_FLUSH_DENORMAL_SUPPORTED:
            raise RuntimeError("cpu_flush_denormal=True is unsupported on this CPU")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def complexities(model, config: dict) -> dict:
    if not model.stochastic:
        return {}
    kl = gaussian_kl(model.mu, model.rho, config["prior_std"])
    kwargs = dict(n=config["train_size"], m=config["m"], delta=config["delta"],
                  perfect_training=config["train_channel"]["kind"] == "perfect")
    terms = theorem1_terms(kl, **kwargs)
    return {"kl": kl, **terms, "joint_complexity": joint_complexity(kl, **kwargs)}


def bounded_losses_from_ce(losses: torch.Tensor, config: dict) -> torch.Tensor:
    """Per-example loss in [0,1], uniformly over data, channels and weights.

    Smooth mode places a floor under the true-class probability before taking
    its logarithm: -log(epsilon+(1-epsilon)*p_y)/(-log epsilon). It has no hard
    clipping threshold. logaddexp avoids underflow in its numerical evaluation.
    Raw-CE runs retain clipped CE as their separate theoretical diagnostic.
    """
    if config["training_loss"] == "bounded_ce_smooth":
        epsilon = config["loss_epsilon"]
        log_probability = torch.logaddexp(
            torch.full_like(losses, math.log(epsilon)), math.log1p(-epsilon) - losses)
        return (-log_probability / -math.log(epsilon)).clamp(min=0., max=1.)
    return losses.clamp(max=config["clip_ce"]) / config["clip_ce"]


def bounded_loss_description(config: dict) -> str:
    if config["training_loss"] == "bounded_ce_smooth":
        return f"-log(epsilon+(1-epsilon)*p_y)/(-log(epsilon)); epsilon={config['loss_epsilon']}"
    return f"min(CE,{config['clip_ce']})/{config['clip_ce']}"


def empirical_objective(logits: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Empirical loss optimized by training, distinct from raw CE reporting."""
    losses = F.cross_entropy(logits, targets, reduction="none")
    return losses.mean() if config["training_loss"] == "raw_ce" else bounded_losses_from_ce(losses, config).mean()


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
        loss_sum, bounded_sum, power_sum, correct, count = 0., 0., 0., 0, 0
        for x, y in loader:
            x, y = x.to(config["device"], non_blocking=True), y.to(config["device"], non_blocking=True)
            logits = model(x, theta, bank, source, config["deployment_channel"], channel_generator)
            losses = F.cross_entropy(logits, y, reduction="none")
            loss_sum += losses.sum().item()
            bounded_sum += bounded_losses_from_ce(losses, config).sum().item()
            power_sum += model.last_feature_power * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            count += len(y)
        replicates.append(dict(cross_entropy=loss_sum / count,
                               bounded_loss=bounded_sum / count, accuracy=correct / count,
                               feature_power_per_coordinate=power_sum / count))
    result = {key: float(np.mean([rep[key] for rep in replicates])) for key in replicates[0]}
    result.update(draws=draws, examples=count, channel_source=source,
                  bounded_loss_definition=bounded_loss_description(config),
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
                cpu_numerics=(dict(flush_denormal_requested=config.get("cpu_flush_denormal", False),
                                   flush_denormal_supported=_CPU_FLUSH_DENORMAL_SUPPORTED)
                              if torch.device(config["device"]).type == "cpu" else None),
                cpu_threading=(dict(
                    allowed_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    torch_intraop_threads=torch.get_num_threads(),
                    environment={key: os.environ.get(key) for key in (
                        "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
                        "KMP_AFFINITY", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST")})
                    if torch.device(config["device"]).type == "cpu" else None),
                cuda_policy=(dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
                                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
                             if config["device"].startswith("cuda") else None),
                prior="fixed N(0,prior_std^2 I), independent of all data and channel banks",
                training_objective=("raw cross entropy + tuned complexity surrogate"
                                    if config["training_loss"] == "raw_ce" else
                                    "normalized bounded empirical CE + configured regularization; proposed with eta_channel=eta_data=1 matches the separated bound's trainable objective"),
                theoretical_loss=bounded_loss_description(config) + " in [0,1]; sigma_data^2=sigma_channel^2=1/4",
                checkpoint_selection=("validation accuracy, with lower raw CE as tiebreak"
                                      if config["selection_metric"] == "accuracy" else
                                      "validation raw CE, with higher accuracy as tiebreak"),
                validation_randomness=config["validation_seed_policy"] + " posterior/channel seed across epochs",
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
    optimizer = make_optimizer(model, config)
    channel_generator = torch.Generator().manual_seed(config["channel_seed"] + 109)
    metadata = provenance(config, split)
    metadata.update(input_dim=input_dim, classes=classes, parameter_count=model.d,
                    training_channel_bank_hash=digest({"gain": bank.gain.cpu().tolist(),
                                                       "noise": bank.noise.cpu().tolist()}))
    if hasattr(model, "split_description"):
        metadata["network_split"] = dict(
            description=model.split_description,
            transmitted_shape=list(model.transmitted_shape),
            real_channel_uses=model.width,
            transmitter_parameter_count=model.transmitter_parameter_count,
            receiver_parameter_count=model.receiver_parameter_count)
    metadata["posterior_scale_optimization"] = dict(
        stochastic_predictor=model.stochastic, trainable=model.rho.requires_grad,
        initial_standard_deviation=config["posterior_std"],
        rho_learning_rate=(config["lr"] * config["rho_lr_scale"] if model.rho.requires_grad else 0.))
    save_json(output / "metadata.json", metadata)
    best, history = (float("inf"), float("inf")), []
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        ce_sum, empirical_sum, objective_sum, samples = 0., 0., 0., 0
        for x, y in train_loader:
            x, y = x.to(config["device"], non_blocking=True), y.to(config["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            theta = model.sample()
            logits = model(x, theta, bank, "bank", generator=channel_generator)
            ce = F.cross_entropy(logits, y)
            empirical = empirical_objective(logits, y, config)
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
            loss = empirical + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            ce_sum += ce.item() * len(y)
            empirical_sum += empirical.item() * len(y)
            objective_sum += loss.item() * len(y)
            samples += len(y)
        validation_seed = (config["eval_seed"] if config["validation_seed_policy"] == "fixed" else
                           config["eval_seed"] + epoch * 1009)
        validation = evaluate(model, bank, val_loader, config, config["val_draws"], validation_seed)
        if not math.isfinite(validation["cross_entropy"]):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch + 1}")
        record = dict(epoch=epoch + 1, training_cross_entropy=ce_sum / samples,
                      training_empirical_objective=empirical_sum / samples,
                      training_total_objective=objective_sum / samples,
                      validation=validation, epoch_seconds=time.perf_counter() - epoch_start,
                      final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()})
        history.append(record)
        selection_key = ((-validation["accuracy"], validation["cross_entropy"])
                         if config["selection_metric"] == "accuracy" else
                         (validation["cross_entropy"], -validation["accuracy"]))
        if selection_key < best:
            best = selection_key
            state = dict(model=model.state_dict(), bank=bank.state_dict())
            torch.save(state, output / "weights.pt.tmp")
            (output / "weights.pt.tmp").replace(output / "weights.pt")
            metadata["selected_epoch"] = epoch + 1
            metadata["selected_validation_cross_entropy"] = validation["cross_entropy"]
            metadata["selected_validation_accuracy"] = validation["accuracy"]
            metadata["selected_validation_bounded_loss"] = validation["bounded_loss"]
            save_json(output / "metadata.json", metadata)
        save_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch + 1, "validation_ce": validation["cross_entropy"],
                          "validation_accuracy": validation["accuracy"],
                          "epoch_seconds": record["epoch_seconds"]}), flush=True)
    metadata.update(total_seconds=time.perf_counter() - start, peak_rss_mb=peak_memory_mb(),
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated() / 1024**2
                                            if config["device"].startswith("cuda") else None),
                    cuda_peak_reserved_mb=(torch.cuda.max_memory_reserved() / 1024**2
                                           if config["device"].startswith("cuda") else None))
    save_json(output / "metadata.json", metadata)
    print(json.dumps({"output": str(output), "selected_epoch": metadata["selected_epoch"],
                      "seconds": metadata["total_seconds"], "peak_rss_mb": metadata["peak_rss_mb"]}), flush=True)


def evaluate_checkpoint(checkpoint: Path, overrides_path: str | None, output: Path, split_name: str,
                        risk_scope: str = "full"):
    if risk_scope not in {"deployment", "full"}:
        raise ValueError("risk_scope must be deployment or full")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"].copy()
    # Preserve data generated by checkpoints predating the IID sampler. Existing
    # pilot artifacts must not silently switch to a different population.
    if config["dataset"] == "moons" and "moons_sampler" not in config:
        config["moons_sampler"] = "legacy_grid"
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
    report = dict(checkpoint=str(checkpoint.resolve()), split=split_name, risk_scope=risk_scope,
                  evaluation_config=config, selected_epoch=metadata.get("selected_epoch"),
                  heldout_gibbs_mc=result,
                  channel_shift_diagnostic=channel_shift_diagnostic(config["deployment_channel"],
                                                                   config["train_channel"], config["width"]),
                  final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()},
                  certificate_note=metadata["certificate_note"])
    if risk_scope == "deployment":
        save_json(output / "evaluation.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
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
    report.update(empirical_gibbs_mc=empirical,
                  same_training_data_fresh_training_channel_gibbs_mc=same_data_fresh_training,
                  heldout_fresh_training_channel_gibbs_mc=heldout_fresh_training,
                  signed_gap_decomposition_mc=signed,
                  bounded_gap_mc=result["bounded_loss"] - empirical["bounded_loss"])
    save_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "validate"], default="validate")
    parser.add_argument("--config", help="JSON config; evaluate accepts only evaluation overrides")
    parser.add_argument("--output", type=Path, default=Path("results/journal"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--risk-scope", choices=["deployment", "full"], default="full",
                        help="deployment skips training-risk decomposition for inexpensive screening")
    args = parser.parse_args()
    if args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate")
        evaluate_checkpoint(args.checkpoint, args.config, args.output, args.split, args.risk_scope)
    else:
        config = read_config(args.config)
        if args.mode == "validate":
            print(json.dumps(config, indent=2))
            return
        check_device(config)
        torch.set_num_threads(config["threads"])
        train(config, args.output)


if __name__ == "__main__":
    main()
