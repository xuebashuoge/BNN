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
from collections.abc import Mapping
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
    from .full_fading import validate_fading_dimensions
    from .theorem1 import (ChannelBank, GaussianMLP, channel_shift_diagnostic, gaussian_kl, gaussian_kl_to_prior,
                           joint_complexity, theorem1_terms, validate_channel)
except ImportError:
    from full_fading import validate_fading_dimensions
    from theorem1 import (ChannelBank, GaussianMLP, channel_shift_diagnostic, gaussian_kl, gaussian_kl_to_prior,
                          joint_complexity, theorem1_terms, validate_channel)


DEFAULTS = dict(dataset="mnist", architecture="mlp", method="proposed", predictor="legacy", training_loss="raw_ce", loss_epsilon=.01,
                selection_metric="cross_entropy", validation_seed_policy="fixed", width=128, train_size=55000,
                val_size=5000, batch_size=256, epochs=30, lr=.001,
                eta_channel=.01, eta_data=.1, prior_std=1., posterior_std=.01,
                m=100, seed=0, data_seed=20261110, channel_seed=1, eval_seed=2,
                eval_draws=10, val_draws=1, clip_ce=10., delta=.05, l2=1e-4,
                device="cpu", threads=1, cpu_flush_denormal=False, transmit_unit_power=False,
                rho_lr_scale=1., download=False, moon_noise=.3, moons_sampler="iid",
                train_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1},
                deployment_channel={"kind": "gaussian", "gain_mean": 1., "gain_std": .1, "noise_std": .1})
BOUNDED_MODEL_DEFAULTS = dict(bounded_feature_norm=1., bounded_receiver_norm=.3, posterior_mean_clip=0.)
# Omitted settings keep the normalized configuration and hash of older runs.
OPTIONAL_OPTIMIZATION_DEFAULTS = dict(lr_decay=1., train_mc=1)
# Data-dependent channel prior Q_{W|S} = N(mu_S, channel_prior_std^2 I) for the
# channel complexity of Theorem 1. mu_S is the posterior mean of a frozen
# stage-1 checkpoint trained on the same S without the stage-2 bank U:
#   perfect           stage 1 over the noiseless channel (no channel samples);
#   independent_bank  stage 1 over an independently seeded bank U' ~ P_tr^m;
#   fresh             stage 1 with fresh P_tr draws for every minibatch.
# The sample complexity always keeps the fixed N(0, prior_std^2 I) prior.
# Omitted fields keep the configuration and hash of older runs.
OPTIONAL_PRIOR_DEFAULTS = dict(channel_prior="fixed", channel_prior_std=None, channel_prior_checkpoint=None,
                               channel_prior_count=1, init_from_prior=True, train_channel_source="bank",
                               bank_split=None, bank_part="all")
CHANNEL_PRIORS = {"fixed", "perfect", "independent_bank", "fresh", "split"}
# bank_part selects the training channels: all of U, U_1 = U[:bank_split] or U_2 = U[bank_split:].
BANK_PARTS = {"all", "first", "second"}
PRIOR_COMPATIBILITY_KEYS = ("dataset", "architecture", "width", "train_size", "val_size", "data_seed",
                            "moon_noise", "moons_sampler", "transmit_unit_power", "training_loss",
                            "loss_epsilon", "prior_std", *BOUNDED_MODEL_DEFAULTS)
MOONS_EVALUATION_OVERRIDES = ("moons_eval_data_seed", "moons_eval_size")
METHODS = {"erm", "l2", "stochastic_erm", "pac_bayes", "proposed", "joint"}
CNN_ARCHITECTURES = {"cnn", "cnn_mid", "cnn_mid_relu", "cnn_rxconv", "cnn_spatial3"}
_CPU_FLUSH_DENORMAL_SUPPORTED: bool | None = None
LEGACY_BANK_HASH_ALGORITHM = "legacy-json-sha256"
TENSOR_BANK_HASH_ALGORITHM = "tensor-bytes-sha256-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def channel_bank_hash_algorithm(bank_or_state) -> str:
    state = bank_or_state if isinstance(bank_or_state, Mapping) else bank_or_state.state_dict()
    diagonal_fading = getattr(bank_or_state, "config", {}).get("kind") == "diagonal_fading"
    return TENSOR_BANK_HASH_ALGORITHM if diagonal_fading or state["gain"].ndim == 3 else LEGACY_BANK_HASH_ALGORITHM


def channel_bank_fingerprint(bank_or_state, algorithm: str | None = None) -> str:
    """Hash actual bank buffers without materializing Python floats for full matrices."""
    state = bank_or_state if isinstance(bank_or_state, Mapping) else bank_or_state.state_dict()
    algorithm = channel_bank_hash_algorithm(bank_or_state) if algorithm is None else algorithm
    if algorithm == LEGACY_BANK_HASH_ALGORITHM:
        return digest({name: state[name].detach().cpu().tolist() for name in ("gain", "noise")})
    if algorithm != TENSOR_BANK_HASH_ALGORITHM:
        raise ValueError(f"Unknown channel-bank hash algorithm: {algorithm}")
    hasher = hashlib.sha256()
    hasher.update(algorithm.encode() + b"\0")
    for name in ("gain", "noise"):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = json.dumps(dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape),
                                     nbytes=tensor.numel() * tensor.element_size()),
                                sort_keys=True, separators=(",", ":")).encode()
        hasher.update(len(descriptor).to_bytes(8, "big"))
        hasher.update(descriptor)
        # CPU tensor bytes retain their dtype, shape and bit pattern.
        raw = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
        for start in range(0, len(raw), 8 * 1024**2):
            hasher.update(raw[start:start + 8 * 1024**2])
    return hasher.hexdigest()


def _bank_recipe(config):
    return dict(version=1, width=config["width"], m=config["m"],
                channel=validate_channel(config["train_channel"]),
                seed=config["channel_seed"], torch_version=str(torch.__version__))


def checkpoint_bank_state(state: Mapping, config: dict, regenerated_bank=None):
    """Restore legacy buffers or a verified recipe for a compact full-matrix checkpoint."""
    if "bank_recipe" not in state:
        return state["bank"]
    expected = _bank_recipe(config)
    if "bank" in state or state["bank_recipe"] != expected:
        raise ValueError("Channel-bank recipe differs from its declared training configuration/runtime")
    if regenerated_bank is None:
        regenerated_bank = ChannelBank(expected["width"], expected["m"], expected["channel"], expected["seed"])
    return regenerated_bank.state_dict()


def file_fingerprint(path: Path) -> dict:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for raw in iter(lambda: source.read(8 * 1024**2), b""):
            hasher.update(raw)
            size += len(raw)
    return dict(sha256=hasher.hexdigest(), bytes=size)


def read_config(path: str | None) -> dict:
    supplied = {} if path is None else json.loads(Path(path).read_text())
    return normalize_config(supplied)


def normalize_config(supplied: dict) -> dict:
    unknown = (supplied.keys() - DEFAULTS.keys() - BOUNDED_MODEL_DEFAULTS.keys()
               - OPTIONAL_OPTIMIZATION_DEFAULTS.keys() - OPTIONAL_PRIOR_DEFAULTS.keys())
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    if not isinstance(config["cpu_flush_denormal"], bool):
        raise ValueError("cpu_flush_denormal must be a boolean")
    if not isinstance(config["transmit_unit_power"], bool):
        raise ValueError("transmit_unit_power must be a boolean")
    if config["architecture"] == "bounded_mlp" and config["transmit_unit_power"]:
        raise ValueError("bounded_mlp requires transmit_unit_power=False to enforce its feature norm cap")
    if (not isinstance(config["rho_lr_scale"], (int, float)) or isinstance(config["rho_lr_scale"], bool)
            or not math.isfinite(config["rho_lr_scale"]) or config["rho_lr_scale"] < 0):
        raise ValueError("rho_lr_scale must be a finite nonnegative number")
    lr_decay = config.get("lr_decay", OPTIONAL_OPTIMIZATION_DEFAULTS["lr_decay"])
    train_mc = config.get("train_mc", OPTIONAL_OPTIMIZATION_DEFAULTS["train_mc"])
    if not isinstance(lr_decay, (int, float)) or isinstance(lr_decay, bool) or not 0 < lr_decay <= 1:
        raise ValueError("lr_decay must lie in (0, 1]")
    if not isinstance(train_mc, int) or isinstance(train_mc, bool) or train_mc < 1:
        raise ValueError("train_mc must be a positive integer")
    if config["architecture"] == "bounded_mlp" or BOUNDED_MODEL_DEFAULTS.keys() & supplied.keys():
        for key, default in BOUNDED_MODEL_DEFAULTS.items():
            value = config.setdefault(key, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                    or value < 0 or (key != "posterior_mean_clip" and value == 0)):
                raise ValueError(f"{key} requires a finite {'nonnegative' if key == 'posterior_mean_clip' else 'positive'} number")
    config["train_channel"] = validate_channel(config["train_channel"])
    config["deployment_channel"] = validate_channel(config["deployment_channel"])
    if config["method"] not in METHODS or config["dataset"] not in {"moons", "mnist", "fashionmnist", "cifar10"}:
        raise ValueError("Unknown method or dataset")
    if not isinstance(config["predictor"], str) or config["predictor"] not in {"legacy", "gaussian"}:
        raise ValueError("predictor must be legacy or gaussian")
    if config["architecture"] not in {"mlp", "bounded_mlp", *CNN_ARCHITECTURES}:
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
    for channel in (config["train_channel"], config["deployment_channel"]):
        if channel["kind"] == "full_fading":
            validate_fading_dimensions(channel, config["width"], config["width"])
    if config["train_size"] <= 1 or config["val_size"] <= 0:
        raise ValueError("Require train_size>1 and val_size>0")
    if min(config["clip_ce"], config["prior_std"], config["posterior_std"], config["lr"]) <= 0:
        raise ValueError("Loss cap, scales, and learning rate must be positive")
    if not 0 < config["delta"] < 1 or min(config["eta_channel"], config["eta_data"], config["l2"]) < 0:
        raise ValueError("Require 0<delta<1 and nonnegative regularization coefficients")
    if config["m"] <= 1 and config["train_channel"]["kind"] != "perfect":
        raise ValueError("Nonperfect training requires m>1")
    validate_prior_config(config)
    return config


def prior_setting(config: dict, key: str):
    return config.get(key, OPTIONAL_PRIOR_DEFAULTS[key])


def validate_prior_config(config: dict) -> None:
    kind = prior_setting(config, "channel_prior")
    if kind not in CHANNEL_PRIORS:
        raise ValueError(f"channel_prior must be one of {sorted(CHANNEL_PRIORS)}")
    if prior_setting(config, "train_channel_source") not in {"bank", "fresh"}:
        raise ValueError("train_channel_source must be bank or fresh")
    if not isinstance(prior_setting(config, "init_from_prior"), bool):
        raise ValueError("init_from_prior must be a boolean")
    count = prior_setting(config, "channel_prior_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("channel_prior_count must be a positive integer")
    part, cut = prior_setting(config, "bank_part"), prior_setting(config, "bank_split")
    if part not in BANK_PARTS:
        raise ValueError(f"bank_part must be one of {sorted(BANK_PARTS)}")
    if cut is not None:
        if not isinstance(cut, int) or isinstance(cut, bool) or not 2 <= cut <= config["m"] - 2:
            raise ValueError("bank_split must be an integer with 2 <= bank_split <= m - 2")
        if config["train_channel"]["kind"] == "perfect" or prior_setting(config, "train_channel_source") != "bank":
            raise ValueError("A bank split needs a nonperfect channel bank used for training")
    elif part != "all" or kind == "split":
        raise ValueError("bank_part other than 'all' and channel_prior='split' require bank_split")
    std, path = prior_setting(config, "channel_prior_std"), prior_setting(config, "channel_prior_checkpoint")
    if kind == "fixed":
        if std is not None or path is not None:
            raise ValueError("A fixed channel prior takes no channel_prior_std or checkpoint")
        return
    if (not isinstance(std, (int, float)) or isinstance(std, bool) or not math.isfinite(std) or std <= 0):
        raise ValueError("A data-dependent channel prior requires a positive channel_prior_std")
    if not isinstance(path, str) or not path:
        raise ValueError("A data-dependent channel prior requires channel_prior_checkpoint")
    if config["train_channel"]["kind"] == "perfect":
        raise ValueError("Perfect training has no channel complexity; use channel_prior='fixed'")
    if prior_setting(config, "train_channel_source") != "bank":
        raise ValueError("The stage-2 posterior must train on its channel bank U")
    if kind == "split" and part == "first":
        raise ValueError("A split-U posterior cannot train on U_1 alone")


def load_channel_prior(config: dict, split: dict | None = None):
    """Return the frozen prior mean mu_S and a verifiable record of its source.

    Validity of Theorem 1 needs Q_{W|S} independent of the bank U. Stage 1
    therefore must share S (same split hashes) and must not train on the
    stage-2 bank: it uses the perfect channel, a different bank seed, or fresh
    simulator draws, according to the declared construction. With "split", the
    bank is U = (U_1, U_2): stage 1 trains on U_1 = U[:bank_split] only, and the
    channel complexity is the conditional bound given (S, U_1) over U_2, with
    m_2 = m - bank_split channels.
    """
    kind = prior_setting(config, "channel_prior")
    if kind == "fixed":
        return None, None
    checkpoint = Path(prior_setting(config, "channel_prior_checkpoint"))
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    prior_config = normalize_config(metadata["config"])
    for key in PRIOR_COMPATIBILITY_KEYS:
        if prior_config.get(key) != config.get(key):
            raise ValueError(f"Channel-prior checkpoint differs from the posterior in {key}")
    if prior_setting(prior_config, "channel_prior") != "fixed":
        raise ValueError("A channel-prior checkpoint must itself use the fixed prior")
    source = prior_setting(prior_config, "train_channel_source")
    if kind == "perfect":
        valid = prior_config["train_channel"]["kind"] == "perfect"
    elif kind == "independent_bank":
        valid = (source == "bank" and prior_config["train_channel"] == config["train_channel"]
                 and prior_config["channel_seed"] != config["channel_seed"])
    elif kind == "split":
        valid = (source == "bank" and prior_config["train_channel"] == config["train_channel"]
                 and prior_config["channel_seed"] == config["channel_seed"] and prior_config["m"] == config["m"]
                 and prior_setting(prior_config, "bank_part") == "first"
                 and prior_setting(prior_config, "bank_split") == prior_setting(config, "bank_split"))
    else:
        valid = source == "fresh" and prior_config["train_channel"] == config["train_channel"]
    if not valid:
        raise ValueError(f"Channel-prior checkpoint does not implement the {kind} construction")
    if split is not None:
        for key in ("train_indices_hash", "validation_indices_hash"):
            if metadata["split"][key] != split[key]:
                raise ValueError(f"Channel-prior checkpoint uses a different data split: {key}")
    state = torch.load(checkpoint / "weights.pt", map_location=config["device"], weights_only=True)
    prior_mu = state["model"]["mu"].detach().clone()
    record = dict(construction=kind, checkpoint=str(checkpoint.resolve()),
                  weights=file_fingerprint(checkpoint / "weights.pt"),
                  metadata=file_fingerprint(checkpoint / "metadata.json"),
                  stage1_config_hash=digest(prior_config), stage1_selected_epoch=metadata.get("selected_epoch"),
                  stage1_train_channel=prior_config["train_channel"], stage1_channel_seed=prior_config["channel_seed"],
                  stage1_train_channel_source=source, prior_std=prior_setting(config, "channel_prior_std"),
                  stage1_bank_part=prior_setting(prior_config, "bank_part"),
                  bank_split=prior_setting(config, "bank_split"),
                  candidate_count=prior_setting(config, "channel_prior_count"))
    return prior_mu, record


def attach_channel_prior(model, config: dict, split: dict | None = None, expected: dict | None = None):
    prior_mu, record = load_channel_prior(config, split)
    if prior_mu is None:
        model.channel_prior_mu = None
        return None
    if prior_mu.shape != model.mu.shape:
        raise ValueError("Channel-prior mean has a different parameter count")
    if expected is not None and (record["weights"] != expected["weights"]
                                 or record["metadata"] != expected["metadata"]):
        raise ValueError("Channel-prior checkpoint changed after the posterior was trained")
    model.channel_prior_mu = prior_mu.to(model.mu)
    return record


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

        def image_dataset(training):
            raw = dataset_class(root=root, train=training, transform=transform, download=config["download"])
            if config["deployment_channel"]["kind"] == "diagonal_fading":
                # The diagonal sweep repeatedly evaluates the same images across
                # epochs and posterior draws. Materialize the exact ToTensor values once.
                pixels = raw.data.unsqueeze(1).to(torch.float32).div_(255)
                return TensorDataset(pixels, raw.targets)
            return raw

        full = image_dataset(True)
        test = image_dataset(False) if include_test else None
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


def uses_gaussian_predictor(config: dict) -> bool:
    """Explicit Gaussian baselines without changing saved deterministic runs.

    Legacy ERM/L2 checkpoints retain their Dirac predictor. Every algorithm in
    an all-Bayesian campaign sets predictor='gaussian'; the algorithm then
    determines only its regularization, not the predictor family.
    """
    return config.get("predictor", "legacy") == "gaussian" or config["method"] not in {"erm", "l2"}


def make_models(config: dict, input_dim: int, classes: int):
    torch.manual_seed(config["seed"])
    stochastic = uses_gaussian_predictor(config)
    if config.get("architecture", "mlp") in CNN_ARCHITECTURES:
        from .cifar_models import GaussianMiddleSplitCNN, GaussianSplitCNN
        if config["architecture"] in {"cnn", "cnn_mid"}:
            model_class = GaussianMiddleSplitCNN if config["architecture"] == "cnn_mid" else GaussianSplitCNN
        else:
            from .cifar_search_models import CIFAR_SEARCH_MODELS
            model_class = CIFAR_SEARCH_MODELS[config["architecture"]]
        model = model_class(config["width"], classes, stochastic,
                            config["posterior_std"]).to(config["device"])
    elif config["architecture"] == "bounded_mlp":
        from .bounded_mlp import BoundedGaussianMLP
        model = BoundedGaussianMLP(input_dim, config["width"], classes, stochastic, config["posterior_std"],
                                   config["bounded_feature_norm"], config["bounded_receiver_norm"]).to(config["device"])
    else:
        model = GaussianMLP(input_dim, config["width"], classes, stochastic,
                            config["posterior_std"]).to(config["device"])
    if model.stochastic and config.get("rho_lr_scale", 1.) == 0:
        # A fixed positive Gaussian scale remains stochastic and has finite KL.
        model.rho.requires_grad_(False)
    bank = ChannelBank(config["width"], config["m"], config["train_channel"],
                       config["channel_seed"],
                       transmit_unit_power=config["transmit_unit_power"]).to(config["device"])
    part, cut = prior_setting(config, "bank_part"), prior_setting(config, "bank_split")
    if part == "first":
        bank.restrict(0, cut)
    elif part == "second":
        bank.restrict(cut, config["m"])
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
    prior_mu = getattr(model, "channel_prior_mu", None)
    if prior_mu is None:
        terms = theorem1_terms(kl, **kwargs)
        return {"kl": kl, **terms, "joint_complexity": joint_complexity(kl, **kwargs)}
    channel_kl = gaussian_kl_to_prior(model.mu, model.rho, prior_mu, prior_setting(config, "channel_prior_std"))
    if prior_setting(config, "channel_prior") == "split":
        # Conditional on (S, U_1) the bound is over U_2 alone.
        kwargs["m"] = config["m"] - prior_setting(config, "bank_split")
    terms = theorem1_terms(kl, **kwargs, channel_kl=channel_kl,
                           channel_prior_count=prior_setting(config, "channel_prior_count"))
    return {"kl": kl, "channel_kl": channel_kl, **terms, "joint_complexity": joint_complexity(kl, **kwargs)}


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


def training_penalty(model, config: dict, terms: dict | None = None) -> torch.Tensor:
    """Algorithm-specific penalty for the configured common predictor family.

    Gaussian L2 regularizes E_q ||W||^2 exactly, including posterior variance.
    Its deterministic specialization is the historical ||mu||^2 objective.
    ERM optimizes the expected empirical loss with no regularization penalty.
    Bound evaluation remains unweighted and applies to any Gaussian method.
    """
    method = config["method"]
    if method == "l2":
        second_moment = model.mu.square().sum()
        if model.stochastic:
            second_moment = second_moment + F.softplus(model.rho).square().sum()
        return config["l2"] * second_moment
    if method in {"erm", "stochastic_erm"}:
        return model.mu.new_zeros(())
    if terms is None:
        terms = complexities(model, config)
    if method in {"proposed", "pac_bayes"}:
        penalty = config["eta_data"] * terms["sample_complexity"]
        if method == "proposed":
            penalty = penalty + config["eta_channel"] * terms["channel_complexity"]
        return penalty
    if method == "joint":
        return config["eta_data"] * terms["joint_complexity"]
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def evaluate(model, bank, loader, config: dict, draws: int, seed: int,
             source: str = "deployment") -> dict:
    """Independent full-loader Gibbs draws; do not resample W per minibatch."""
    model.eval()
    weight_generator = torch.Generator(device=config["device"]).manual_seed(seed)
    channel_kind = (config["deployment_channel"]["kind"] if source == "deployment"
                    else bank.config["kind"] if source == "bank" else "perfect")
    channel_device = config["device"] if channel_kind in {"full_fading", "diagonal_fading"} else "cpu"
    channel_generator = torch.Generator(device=channel_device).manual_seed(seed + 1)
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
                  posterior_rule=("one weight draw fixed across entire loader per replicate"
                                  if model.stochastic else "fixed deterministic posterior mean"),
                  predictor=dict(family="diagonal_gaussian" if model.stochastic else "deterministic_dirac",
                                 stochastic=model.stochastic),
                  replicate_statistics=replicates,
                  mc_note="Finite-data Gibbs Monte Carlo estimates, not population-risk certificates",
                  bounded_loss_mc_upper_radius_95pct=math.sqrt(math.log(20) / (2 * draws)))
    if channel_kind == "full_fading" and source == "deployment":
        result["channel_sampling"] = (
            "Exact conditional received-signal law per example; paired seeds reuse standardized "
            "Gaussian variates in received space.")
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
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", ".."], text=True,
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
                predictor=dict(family="diagonal_gaussian" if uses_gaussian_predictor(config) else "deterministic_dirac",
                               stochastic=uses_gaussian_predictor(config),
                               configuration=config["predictor"]),
                regularization=("l2 * E_q[||W||^2] = l2 * sum(mu^2 + softplus(rho)^2)"
                                if config["method"] == "l2" and uses_gaussian_predictor(config) else
                                "l2 * sum(mu^2)" if config["method"] == "l2" else
                                "none" if config["method"] in {"erm", "stochastic_erm"} else
                                "configured PAC-Bayes complexity penalty"),
                training_objective=(("raw cross entropy" if config["training_loss"] == "raw_ce" else
                                     "normalized bounded empirical CE") +
                                    ("; no regularization" if config["method"] in {"erm", "stochastic_erm"} else
                                     " + expected squared-weight L2 regularization" if config["method"] == "l2" else
                                     " + tuned complexity surrogate" if config["training_loss"] == "raw_ce" else
                                     " + configured PAC-Bayes penalty; proposed with eta_channel=eta_data=1 matches the separated bound's trainable objective")),
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
        if normalize_config(old_config) != config:
            raise ValueError("Output contains a different experiment; use a fresh output directory")
    train_data, val_data, _, input_dim, classes, split = load_data(config, include_test=False)
    train_loader, val_loader = make_loader(train_data, config, True), make_loader(val_data, config)
    model, bank = make_models(config, input_dim, classes)
    prior_record = attach_channel_prior(model, config, split)
    if prior_record is not None and prior_setting(config, "init_from_prior"):
        with torch.no_grad():
            model.mu.copy_(model.channel_prior_mu)
    fresh_training = prior_setting(config, "train_channel_source") == "fresh"
    mean_clip = config.get("posterior_mean_clip", 0.)
    if mean_clip:
        with torch.no_grad():
            model.mu.clamp_(-mean_clip, mean_clip)
    optimizer = make_optimizer(model, config)
    lr_decay = config.get("lr_decay", OPTIONAL_OPTIMIZATION_DEFAULTS["lr_decay"])
    train_mc = config.get("train_mc", OPTIONAL_OPTIMIZATION_DEFAULTS["train_mc"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay) if lr_decay < 1. else None
    channel_device = config["device"] if bank.config["kind"] in {"full_fading", "diagonal_fading"} else "cpu"
    channel_generator = torch.Generator(device=channel_device).manual_seed(config["channel_seed"] + 109)
    metadata = provenance(config, split)
    bank_hash_algorithm = channel_bank_hash_algorithm(bank)
    metadata.update(input_dim=input_dim, classes=classes, parameter_count=model.d,
                    training_channel_bank_hash=channel_bank_fingerprint(bank, bank_hash_algorithm),
                    training_channel_bank_hash_algorithm=bank_hash_algorithm)
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
    metadata["optimizer_schedule"] = dict(initial_learning_rates=[group["lr"] for group in optimizer.param_groups],
                                          epoch_multiplier=lr_decay)
    if prior_record is not None:
        metadata["prior"] = ("sample complexity: fixed N(0,prior_std^2 I); channel complexity: "
                             "data-dependent N(mu_S, channel_prior_std^2 I) independent of the bank U")
        metadata["channel_prior"] = dict(prior_record, initialized_posterior_mean=prior_setting(config, "init_from_prior"))
    if fresh_training:
        metadata["training_channel_source"] = "fresh draws from train_channel for every example and minibatch; bank unused"
    metadata["training_monte_carlo"] = dict(draws_per_minibatch=train_mc,
                                           empirical_reduction="mean of independent posterior and channel draws",
                                           penalty_applications_per_minibatch=1)
    save_json(output / "metadata.json", metadata)
    best, history = (float("inf"), float("inf")), []
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        model.train()
        ce_sum, empirical_sum, objective_sum, samples = 0., 0., 0., 0
        for x, y in train_loader:
            x, y = x.to(config["device"], non_blocking=True), y.to(config["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_ce, batch_empirical, batch_objective = 0., 0., 0.
            for draw in range(train_mc):
                theta = model.sample()
                if fresh_training:
                    logits = model(x, theta, bank, "deployment", config["train_channel"], channel_generator)
                else:
                    logits = model(x, theta, bank, "bank", generator=channel_generator)
                ce = F.cross_entropy(logits, y)
                empirical = empirical_objective(logits, y, config)
                loss = empirical if train_mc == 1 else empirical / train_mc
                if draw == train_mc - 1:
                    loss = loss + training_penalty(model, config, complexities(model, config))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite objective at epoch {epoch + 1}, draw {draw + 1}")
                # Backpropagate each draw immediately to keep the activation memory
                # at one draw. Apply Adam and the exact penalty once per minibatch.
                loss.backward()
                batch_ce += ce.item()
                batch_empirical += empirical.item()
                batch_objective += loss.item()
            optimizer.step()
            if mean_clip:
                with torch.no_grad():
                    model.mu.clamp_(-mean_clip, mean_clip)
            ce_sum += (batch_ce / train_mc) * len(y)
            empirical_sum += (batch_empirical / train_mc) * len(y)
            objective_sum += batch_objective * len(y)
            samples += len(y)
        validation_seed = (config["eval_seed"] if config["validation_seed_policy"] == "fixed" else
                           config["eval_seed"] + epoch * 1009)
        validation = evaluate(model, bank, val_loader, config, config["val_draws"], validation_seed)
        if not math.isfinite(validation["cross_entropy"]):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch + 1}")
        record = dict(epoch=epoch + 1, training_cross_entropy=ce_sum / samples,
                      training_empirical_objective=empirical_sum / samples,
                      training_total_objective=objective_sum / samples,
                      validation=validation, learning_rates=learning_rates,
                      epoch_seconds=time.perf_counter() - epoch_start,
                      final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()})
        history.append(record)
        selection_key = ((-validation["accuracy"], validation["cross_entropy"])
                         if config["selection_metric"] == "accuracy" else
                         (validation["cross_entropy"], -validation["accuracy"]))
        if selection_key < best:
            best = selection_key
            state = dict(model=model.state_dict())
            if bank.config["kind"] == "full_fading":
                state["bank_recipe"] = _bank_recipe(config)
            else:
                state["bank"] = bank.state_dict()
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
                          "learning_rates": learning_rates,
                          "epoch_seconds": record["epoch_seconds"]}), flush=True)
        if scheduler is not None:
            scheduler.step()
    metadata.update(total_seconds=time.perf_counter() - start, peak_rss_mb=peak_memory_mb(),
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated() / 1024**2
                                            if config["device"].startswith("cuda") else None),
                    cuda_peak_reserved_mb=(torch.cuda.max_memory_reserved() / 1024**2
                                           if config["device"].startswith("cuda") else None))
    save_json(output / "metadata.json", metadata)
    print(json.dumps({"output": str(output), "selected_epoch": metadata["selected_epoch"],
                      "seconds": metadata["total_seconds"], "peak_rss_mb": metadata["peak_rss_mb"]}), flush=True)


def validate_moons_evaluation(config: dict, overrides: dict, split_name: str, risk_scope: str) -> None:
    if set(overrides) != set(MOONS_EVALUATION_OVERRIDES):
        raise ValueError("moons_eval_data_seed and moons_eval_size must be provided together")
    if (config["dataset"] != "moons" or config["moons_sampler"] != "iid" or
            split_name not in {"validation", "test"} or risk_scope != "deployment"):
        raise ValueError("Custom Two moons samples require IID data and deployment validation or test evaluation")
    seed, size = (overrides[key] for key in MOONS_EVALUATION_OVERRIDES)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("moons_eval_data_seed must be a nonnegative signed 64-bit integer")
    if seed in (config["data_seed"], config["data_seed"] + 1):
        raise ValueError("moons_eval_data_seed must differ from the training pool and original test seeds")
    if type(size) is not int or size <= 0:
        raise ValueError("moons_eval_size must be a positive integer")


def evaluate_checkpoint(checkpoint: Path, overrides_path: str | None, output: Path, split_name: str,
                        risk_scope: str = "full"):
    if risk_scope not in {"deployment", "full"}:
        raise ValueError("risk_scope must be deployment or full")
    checkpoint_sources = {name: file_fingerprint(checkpoint / filename) for name, filename in
                          (("metadata", "metadata.json"), ("weights", "weights.pt"))}

    def verify_frozen_sources():
        current = {name: file_fingerprint(checkpoint / filename) for name, filename in
                   (("metadata", "metadata.json"), ("weights", "weights.pt"))}
        if current != checkpoint_sources:
            raise ValueError("Checkpoint metadata or weights changed during evaluation")

    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"].copy()
    # Preserve data generated by checkpoints predating the IID sampler. Existing
    # pilot artifacts must not silently switch to a different population.
    if config["dataset"] == "moons" and "moons_sampler" not in config:
        config["moons_sampler"] = "legacy_grid"
    overrides = {} if overrides_path is None else json.loads(Path(overrides_path).read_text())
    allowed = {"deployment_channel", "eval_draws", "eval_seed", "device", "threads", "batch_size", "download",
               *MOONS_EVALUATION_OVERRIDES}
    if set(overrides) - allowed:
        raise ValueError(f"Evaluation overrides restricted to {sorted(allowed)}")
    moons_evaluation = {key: overrides.pop(key) for key in MOONS_EVALUATION_OVERRIDES if key in overrides}
    config.update(overrides)
    config = normalize_config(config)
    if moons_evaluation:
        validate_moons_evaluation(config, moons_evaluation, split_name, risk_scope)
    check_device(config)
    torch.set_num_threads(config["threads"])
    data = load_data(config, include_test=split_name == "test" and not moons_evaluation)
    train_data, val_data, test_data, input_dim, classes, split = data
    split_matches = {}
    for key in ("train_indices_hash", "validation_indices_hash"):
        if key in metadata["split"]:
            split_matches[key] = split[key] == metadata["split"][key]
            if not split_matches[key]:
                raise ValueError(f"Data split no longer matches checkpoint: {key}")
    model, bank = make_models(config, input_dim, classes)
    attach_channel_prior(model, config, split, metadata.get("channel_prior"))
    state = torch.load(checkpoint / "weights.pt", map_location=config["device"], weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(checkpoint_bank_state(state, config, regenerated_bank=bank))
    bank_hash_algorithm = metadata.get("training_channel_bank_hash_algorithm", LEGACY_BANK_HASH_ALGORITHM)
    saved_bank_hash = channel_bank_fingerprint(bank, bank_hash_algorithm)
    recorded_bank_hash = metadata.get("training_channel_bank_hash")
    if recorded_bank_hash is not None and saved_bank_hash != recorded_bank_hash:
        raise ValueError("Saved channel-bank buffers differ from the recorded bank hash")
    selected = {"train": train_data, "validation": val_data, "test": test_data}[split_name]
    if moons_evaluation:
        x, y = sample_moons_iid(moons_evaluation["moons_eval_size"], config["moon_noise"],
                               moons_evaluation["moons_eval_data_seed"])
        selected = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = make_loader(selected, config)
    result = evaluate(model, bank, loader, config, config["eval_draws"], config["eval_seed"])
    report = dict(checkpoint=str(checkpoint.resolve()), split=split_name, risk_scope=risk_scope,
                  checkpoint_source_fingerprints=checkpoint_sources,
                  evaluation_config={**config, **moons_evaluation}, selected_epoch=metadata.get("selected_epoch"),
                  heldout_gibbs_mc=result,
                  channel_shift_diagnostic=channel_shift_diagnostic(config["deployment_channel"],
                                                                   config["train_channel"], config["width"]),
                  final_posterior_complexities={k: v.detach().item() for k, v in complexities(model, config).items()},
                  checkpoint_verification=dict(split_hash_matches=split_matches,
                      saved_channel_bank_hash=saved_bank_hash,
                      channel_bank_hash_algorithm=bank_hash_algorithm,
                      bank_hash_matches_metadata=(saved_bank_hash == recorded_bank_hash
                                                  if recorded_bank_hash is not None else None)),
                  certificate_note=metadata["certificate_note"])
    if risk_scope == "deployment":
        verify_frozen_sources()
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
    verify_frozen_sources()
    save_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "validate"], default="validate")
    parser.add_argument("--config", help="JSON config; evaluate accepts only evaluation overrides")
    parser.add_argument("--output", type=Path, default=Path("results/fading_unit_power"))
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
