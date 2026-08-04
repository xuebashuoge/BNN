"""MNIST version of the test12 simulation.

The experiment keeps the two-linear-layer classifier and places the stochastic
channel between its hidden activation and output layer, as in test12.  MNIST
requires a wider hidden representation, but adding another learned layer would
change the channel location assumed by the bound.

Examples
--------
Run the six non-sweep comparisons (the safe default)::

    python test17.py --mode single

Run the paired ERM/proposed training-channel sweep explicitly::

    python test17.py --mode sweep --device cpu

Only bounded smoke tests should be run locally, for example::

    python test17.py --mode single --single-scenarios erm-perfect \
        --n-samples 256 --epochs 1 --eval-repeats 1 --no-cache
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
from typing import Iterable

# Use a worker-safe writable Matplotlib cache on local and Slurm machines.
_MPL_CACHE_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test17-matplotlib")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join("results", "test17_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
DEVICE = torch.device("cpu")


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def configure_device(requested: str) -> torch.device:
    global DEVICE
    DEVICE = resolve_device(requested)
    return DEVICE


# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
DATASET_NAME = "MNIST"
N_SAMPLES = 60000
N_U_SETS = 10

# A width-64 hidden layer works for clean MNIST but loses substantial accuracy
# under the severe test12 channel.  Width 256 provides channel redundancy while
# retaining the original two-linear-layer architecture.
HIDDEN_DIM = 256
IN_DIM = 28 * 28
OUT_DIM = 10

BATCH_SIZE = 128
EPOCHS = 150
LR_BASE = 0.001
LR_DECAY_STEP = 75
LR_DECAY_GAMMA = 0.8
PRIOR_LAMBDA = 1.0
EPSILON = 0.025
SIGMA_SQ = 1.0
SIGMA_ART_SQ = 1.0
ALPHA_COEFF = 0.0
BETA_COEFF = 0.1
GAMMA_COEFF = 0.05
L2_COEFF = 1e-4
M_ARTIFICIAL_CHANNELS = 100
POSTERIOR_INIT_STD = 0.01

# test12 used 100 samples with a very small parameter vector.  MNIST has about
# 200k parameters per posterior component at width 256, so three samples give a
# practical stochastic estimate without creating multi-gigabyte tensors.
MI_MC_SAMPLES = 3
LIPSCHITZ_METHOD_PERFECT = "grad"

EVAL_REPEATS = 3
INFERENCE_WEIGHT_SAMPLES = 5
SEED = 616657849

# Inference/test channel: severe fading and low SNR, unchanged from test12.
MU_M_TE, STD_M_TE = 0.5, 1.0
MU_B_TE, STD_B_TE = 0.0, 1.0

# Artificial/training channel, unchanged from test12.
MU_M_TR, STD_M_TR = MU_M_TE, STD_M_TE
MU_B_TR, STD_B_TR = MU_B_TE, STD_B_TE


def kl_gaussian_1d(mu1: float, std1: float, mu2: float, std2: float) -> float:
    """Closed-form KL D(N_1 || N_2)."""
    if std1 <= 0 or std2 <= 0:
        raise ValueError("Gaussian standard deviations must be positive.")
    return float(
        np.log(std2 / std1)
        + (std1**2 + (mu1 - mu2) ** 2) / (2 * std2**2)
        - 0.5
    )


def channel_kl_total(hidden_dim: int) -> float:
    kl_ch_m = hidden_dim * kl_gaussian_1d(
        MU_M_TE, STD_M_TE, MU_M_TR, STD_M_TR
    )
    kl_ch_b = hidden_dim * kl_gaussian_1d(
        MU_B_TE, STD_B_TE, MU_B_TR, STD_B_TR
    )
    return kl_ch_m + kl_ch_b


def _normal_sum_of_squares(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
    mc_samples: int,
) -> np.ndarray:
    """Sample sums of squared iid normal variables without allocating them."""
    if std == 0:
        return np.full(mc_samples, count * mean**2, dtype=np.float64)
    noncentrality = count * (mean / std) ** 2
    return std**2 * rng.noncentral_chisquare(
        df=count,
        nonc=noncentrality,
        size=mc_samples,
    )


def estimate_expected_channel_norm(
    hidden_dim: int,
    mu_m_te: float,
    mu_b_te: float,
    std_m_te: float,
    std_b_te: float,
    mc_samples: int = 10_000,
) -> torch.Tensor:
    """Estimate the test12 Frobenius channel penalty memory-efficiently.

    This has the same distribution as explicitly sampling test12's
    ``[M - I, B]`` tensor, but samples its aggregate squared norm directly.
    """
    rng = np.random.default_rng(20260804 + int(hidden_dim))
    m_squared = _normal_sum_of_squares(
        rng,
        hidden_dim * hidden_dim,
        mu_m_te - 1.0,
        std_m_te,
        mc_samples,
    )
    b_squared = _normal_sum_of_squares(
        rng,
        hidden_dim,
        mu_b_te,
        std_b_te,
        mc_samples,
    )
    expected_norm = np.sqrt(m_squared + b_squared).mean()
    return torch.tensor(expected_norm, dtype=torch.float32)


CHANNEL_PENALTY_CACHE: dict[int, torch.Tensor] = {}


def get_channel_penalty(hidden_dim: int) -> torch.Tensor:
    if hidden_dim not in CHANNEL_PENALTY_CACHE:
        CHANNEL_PENALTY_CACHE[hidden_dim] = estimate_expected_channel_norm(
            hidden_dim,
            MU_M_TE,
            MU_B_TE,
            STD_M_TE,
            STD_B_TE,
        )
    return CHANNEL_PENALTY_CACHE[hidden_dim]


# ==========================================
# 3. MNIST DATA
# ==========================================
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def dataset_root() -> str:
    """Return the required dataset root from $DATASET."""
    root = os.environ.get("DATASET")
    if not root:
        raise RuntimeError(
            "DATASET is not set. Export DATASET to the directory where MNIST "
            "should be loaded or downloaded."
        )
    return os.path.abspath(os.path.expanduser(root))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def load_mnist_datasets(download: bool) -> tuple[datasets.MNIST, datasets.MNIST]:
    root = dataset_root()
    transform = mnist_transform()
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset


def balanced_subset_indices(
    targets: torch.Tensor,
    n_samples: int,
    seed: int,
) -> list[int]:
    """Select a reproducible approximately class-balanced MNIST subset."""
    if n_samples <= 0 or n_samples > len(targets):
        raise ValueError(
            f"n_samples must be in [1, {len(targets)}], got {n_samples}."
        )
    if n_samples == len(targets):
        return list(range(len(targets)))

    generator = torch.Generator().manual_seed(seed)
    per_class, remainder = divmod(n_samples, OUT_DIM)
    selected: list[int] = []
    for class_id in range(OUT_DIM):
        class_indices = torch.where(targets == class_id)[0]
        class_count = per_class + int(class_id < remainder)
        permutation = torch.randperm(len(class_indices), generator=generator)
        selected.extend(class_indices[permutation[:class_count]].tolist())

    order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[i] for i in order]


def get_dataloaders(
    n_samples: int,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """Load MNIST from $DATASET and create train-subset/test loaders."""
    train_dataset, test_dataset = load_mnist_datasets(download=download)
    indices = balanced_subset_indices(train_dataset.targets, n_samples, seed)
    train_data = (
        train_dataset
        if len(indices) == len(train_dataset)
        else Subset(train_dataset, indices)
    )

    shuffle_generator = torch.Generator().manual_seed(seed + 1)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, len(train_data)


# ==========================================
# 4. TWO-LINEAR-LAYER MODELS & CHANNEL
# ==========================================
class StochasticChannelLayer(nn.Module):
    def __init__(self, k: int, hidden_dim: int, num_artificial_channels: int):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_artificial_channels = num_artificial_channels
        self.train_generator = torch.Generator(device="cpu").manual_seed(1337)
        u_m = (
            torch.randn(
                k,
                num_artificial_channels,
                hidden_dim,
                generator=self.train_generator,
            )
            * STD_M_TR
            + MU_M_TR
        )
        u_b = (
            torch.randn(
                k,
                num_artificial_channels,
                hidden_dim,
                generator=self.train_generator,
            )
            * STD_B_TR
            + MU_B_TR
        )
        self.register_buffer("u_m", u_m)
        self.register_buffer("u_b", u_b)
        self.last_m: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str = "perfect") -> torch.Tensor:
        batch_size = x.shape[1]
        if mode == "perfect":
            m = torch.ones_like(x)
            b = torch.zeros_like(x)
        elif mode == "train":
            indices = torch.randint(
                self.num_artificial_channels,
                (self.k, batch_size),
                generator=self.train_generator,
                device="cpu",
            ).to(x.device)
            component_indices = torch.arange(self.k, device=x.device).unsqueeze(1)
            m = self.u_m[component_indices, indices]
            b = self.u_b[component_indices, indices]
        elif mode == "test":
            m = torch.randn_like(x) * STD_M_TE + MU_M_TE
            b = torch.randn_like(x) * STD_B_TE + MU_B_TE
        else:
            raise ValueError(f"Invalid channel mode: {mode!r}")

        self.last_m = m
        return x * m + b


class VectorizedBNNEnsemble(nn.Module):
    """K diagonal-Gaussian posterior components evaluated in parallel."""

    def __init__(
        self,
        k: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_artificial_channels: int,
    ):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_bnn = True

        self.d1 = in_dim * hidden_dim + hidden_dim
        self.d2 = hidden_dim * out_dim + out_dim
        self.d = self.d1 + self.d2

        self.mu = nn.Parameter(torch.empty(k, self.d))
        initial_rho = float(np.log(np.expm1(POSTERIOR_INIT_STD)))
        self.rho = nn.Parameter(torch.full((k, self.d), initial_rho))
        self.channel_layer = StochasticChannelLayer(
            k,
            hidden_dim,
            num_artificial_channels,
        )
        self.reset_parameters()

    @property
    def K(self) -> int:
        """Compatibility with the test12 notation."""
        return self.k

    def get_sigma(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def reset_parameters(self) -> None:
        """Use fan-in-aware means so 784 MNIST inputs do not inflate logits."""
        with torch.no_grad():
            w1, b1, w2, b2 = self._split_theta(self.mu)
            first_bound = 1.0 / np.sqrt(self.in_dim)
            second_bound = 1.0 / np.sqrt(self.hidden_dim)
            w1.uniform_(-first_bound, first_bound)
            b1.uniform_(-first_bound, first_bound)
            w2.uniform_(-second_bound, second_bound)
            b2.uniform_(-second_bound, second_bound)

    def sample_theta(self, num_samples: int = 1) -> torch.Tensor:
        sigma = self.get_sigma()
        eps = torch.randn(
            num_samples,
            self.k,
            self.d,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )
        return self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def _split_theta(
        self,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w1 = theta[:, : self.in_dim * self.hidden_dim].view(
            self.k,
            self.in_dim,
            self.hidden_dim,
        )
        b1 = theta[:, self.in_dim * self.hidden_dim : self.d1].view(
            self.k,
            self.hidden_dim,
        )
        w2 = theta[:, self.d1 : self.d1 + self.hidden_dim * self.out_dim].view(
            self.k,
            self.hidden_dim,
            self.out_dim,
        )
        b2 = theta[:, self.d1 + self.hidden_dim * self.out_dim :].view(
            self.k,
            self.out_dim,
        )
        return w1, b1, w2, b2

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        mode: str = "perfect",
    ) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        w1, b1, w2, b2 = self._split_theta(theta)
        batch_size = x.shape[0]
        x_k = x.unsqueeze(0).expand(self.k, batch_size, self.in_dim)
        hidden = F.relu(torch.bmm(x_k, w1) + b1.unsqueeze(1))
        hidden_channel = self.channel_layer(hidden, mode=mode)
        return torch.bmm(hidden_channel, w2) + b2.unsqueeze(1)

    def compute_analytical_lipschitz(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        mode: str = "perfect",
    ) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        w1, b1, w2, _ = self._split_theta(theta)
        if w2.is_mps:
            w2_spectral = torch.linalg.matrix_norm(w2.cpu(), ord=2).to(w2.device)
        else:
            w2_spectral = torch.linalg.matrix_norm(w2, ord=2)

        with torch.no_grad():
            batch_size = x.shape[0]
            x_k = x.unsqueeze(0).expand(self.k, batch_size, self.in_dim)
            hidden = F.relu(torch.bmm(x_k, w1) + b1.unsqueeze(1))
            hidden_channel = self.channel_layer(hidden, mode=mode)
            m = self.channel_layer.last_m

        x_norm = torch.norm(x, p=2, dim=1).max()
        hidden_norm = torch.norm(hidden_channel, p=2, dim=2).max(dim=1)[0]
        m_max = (
            torch.abs(m).max(dim=2)[0].max(dim=1)[0]
            if m is not None
            else torch.ones(self.k, device=x.device)
        )
        ce_lipschitz = 1.414
        result = ce_lipschitz * torch.sqrt(
            hidden_norm**2
            + 1.0
            + (x_norm * w2_spectral * m_max) ** 2
            + (w2_spectral * m_max) ** 2
        )
        return result.mean()


class DeterministicFC(nn.Module):
    """Two-linear-layer MNIST classifier with the same channel location."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_artificial_channels: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.channel_layer = StochasticChannelLayer(
            1,
            hidden_dim,
            num_artificial_channels,
        )
        self.is_bnn = False

    def forward(self, x: torch.Tensor, mode: str = "perfect") -> torch.Tensor:
        x = x.flatten(start_dim=1)
        hidden = F.relu(self.fc1(x))
        hidden_channel = self.channel_layer(
            hidden.unsqueeze(0),
            mode=mode,
        ).squeeze(0)
        return self.fc2(hidden_channel)


# ==========================================
# 5. INFORMATION-THEORETIC BOUND CALCULATIONS
# ==========================================
def log_prior(theta: torch.Tensor, prior_lambda: float) -> torch.Tensor:
    log_scale = 0.5 * np.log(2 * np.pi * prior_lambda)
    return -torch.sum(log_scale + theta**2 / (2 * prior_lambda), dim=-1)


def compute_component_expected_kl(model: VectorizedBNNEnsemble) -> torch.Tensor:
    variance = model.get_sigma() ** 2
    kl_components = 0.5 * torch.sum(
        variance / PRIOR_LAMBDA
        + model.mu**2 / PRIOR_LAMBDA
        - 1
        - torch.log(variance / PRIOR_LAMBDA),
        dim=-1,
    )
    return kl_components.mean()


def compute_mixture_kl_and_channel_overfit(
    model: VectorizedBNNEnsemble,
    num_samples: int = MI_MC_SAMPLES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte Carlo mixture KL without a [S, K, K, D] allocation.

    For a diagonal Gaussian, pairwise log densities can be expanded into three
    matrix products.  The result is algebraically equivalent to test12's
    broadcasted calculation while reducing peak intermediate memory from
    O(S*K*K*D) to O(S*K*D + S*K*K).
    """
    theta = model.sample_theta(num_samples)
    mu = model.mu
    variance = model.get_sigma() ** 2 + 1e-8
    inverse_variance = variance.reciprocal()

    flat_theta = theta.reshape(num_samples * model.k, model.d)
    quadratic = (
        flat_theta.square() @ inverse_variance.transpose(0, 1)
        - 2.0 * flat_theta @ (mu * inverse_variance).transpose(0, 1)
        + torch.sum(mu.square() * inverse_variance, dim=1).unsqueeze(0)
    )
    log_normalizer = 0.5 * torch.sum(
        torch.log(2 * np.pi * variance),
        dim=1,
    )
    log_q_all = (-0.5 * quadratic - log_normalizer.unsqueeze(0)).view(
        num_samples,
        model.k,
        model.k,
    )
    log_q_component = log_q_all.diagonal(dim1=1, dim2=2)
    log_q_mixture = torch.logsumexp(log_q_all, dim=2) - np.log(model.k)

    channel_overfit = (log_q_component - log_q_mixture).mean()
    mixture_kl = (log_q_mixture - log_prior(theta, PRIOR_LAMBDA)).mean()
    return mixture_kl, channel_overfit


# ==========================================
# 6. EVALUATION, PLOTTING, AND CACHING
# ==========================================
def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    mode: str = "train",
) -> tuple[float, float]:
    model.eval()
    device = model_device(model)
    total_loss = 0.0
    total_correct = 0.0
    total = 0

    with torch.no_grad():
        theta_mean = model.mu if getattr(model, "is_bnn", False) else None
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if getattr(model, "is_bnn", False):
                output = model(batch_x, theta_mean, mode=mode)
                expanded_y = batch_y.unsqueeze(0).expand(model.k, -1)
                loss = F.cross_entropy(
                    output.reshape(-1, model.out_dim),
                    expanded_y.reshape(-1),
                )
                predictions = output.argmax(dim=-1)
                total_correct += (predictions == expanded_y).sum().item() / model.k
            else:
                output = model(batch_x, mode=mode)
                loss = F.cross_entropy(output, batch_y)
                total_correct += (output.argmax(dim=-1) == batch_y).sum().item()
            total_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def scenario_label(scenario_name: str, mode: str, objective: str) -> str:
    if scenario_name == "erm":
        return f"{scenario_name}_{mode}"
    return f"{scenario_name}_{mode}_{objective}"


def plot_training_metrics(
    history: dict,
    scenario_name: str,
    mode: str,
    run_dir: str,
) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    fig.suptitle(f"{scenario_name.upper()} ({mode}) on MNIST")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "metrics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bound_decomposition(
    history: dict,
    scenario_name: str,
    mode: str,
    run_dir: str,
) -> None:
    if not history.get("bound_total"):
        return
    series = [
        ("Bound Total", history["bound_total"]),
        ("Channel Shifting", history["bound_term1"]),
        ("Channel Overfitting", history["bound_term2"]),
        ("Standard PAC-Bayes / L2", history["bound_term3"]),
        ("Channel Overfit KL", history["channel_overfit_kl"]),
        ("D(P_ch || P_art)", history["kl_ch_total"]),
        ("Mixture KL", history["mixture_kl"]),
        ("Component E[KL]", history["component_expected_kl"]),
        ("K_hat", history["k_hat"]),
        ("Channel Penalty", history["channel_penalty"]),
        ("K_hat * Channel Penalty", history["k_hat_channel_penalty"]),
    ]
    plotted = [
        (label, values)
        for label, values in series
        if values and any(value != 0.0 for value in values)
    ]
    if not plotted:
        return

    fig, axes = plt.subplots(
        len(plotted),
        1,
        figsize=(10, 2.2 * len(plotted)),
        sharex=True,
    )
    if len(plotted) == 1:
        axes = [axes]
    epochs = list(range(1, len(history["bound_total"]) + 1))
    for axis, (label, values) in zip(axes, plotted):
        axis.plot(epochs, values, label=label)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"Bound Decomposition: {scenario_name.upper()} ({mode})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "bound.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def build_param_config(
    scenario_name: str,
    mode: str,
    objective: str,
    n_samples: int,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    lr_decay_step: int,
    lr_decay_gamma: float,
    alpha_coeff: float,
    beta_coeff: float,
    gamma_coeff: float,
    hidden_dim: int,
    n_u_sets: int,
    m_artificial_channels: int,
    mi_mc_samples: int,
    lipschitz_method_perfect: str,
    is_bnn: bool,
) -> dict:
    return {
        "dataset": DATASET_NAME,
        "architecture": "784-hidden-10_two_linear_layers",
        "scenario_name": scenario_name,
        "mode": mode,
        "objective": "N/A" if scenario_name == "erm" else objective,
        "n_samples": n_samples,
        "seed": seed,
        "batch_size": batch_size,
        "is_bnn": is_bnn,
        "posterior_init_std": POSTERIOR_INIT_STD if is_bnn else "N/A",
        "training": {
            "epochs": epochs,
            "lr": lr,
            "lr_decay_step": lr_decay_step,
            "lr_decay_gamma": lr_decay_gamma,
            "alpha_coeff": alpha_coeff if scenario_name != "erm" else "N/A",
            "beta_coeff": beta_coeff if scenario_name != "erm" else "N/A",
            "gamma_coeff": gamma_coeff if scenario_name != "erm" else "N/A",
            "hidden_dim": hidden_dim,
            "n_u_sets": n_u_sets if scenario_name == "proposed" else "N/A",
            "m_artificial_channels": m_artificial_channels,
            "mi_mc_samples": mi_mc_samples,
            "lipschitz_method_perfect": lipschitz_method_perfect,
        },
    }


def get_run_dir(**param_config_kwargs) -> str:
    config = build_param_config(**param_config_kwargs)
    label = scenario_label(
        param_config_kwargs["scenario_name"],
        param_config_kwargs["mode"],
        param_config_kwargs["objective"],
    )
    run_dir = os.path.join(RESULTS_DIR, label, f"param_{config_hash(config)}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def is_converged_history(history: dict) -> bool:
    if not history.get("train_loss") or not history.get("train_acc"):
        return False
    finite = all(
        np.isfinite(value)
        for key in (
            "train_loss",
            "bound_total",
            "bound_term1",
            "bound_term2",
            "bound_term3",
        )
        for value in history.get(key, [])
    )
    return bool(
        finite
        and np.isfinite(history["train_loss"][-1])
        and history["train_loss"][-1] <= 5.0
        and history["train_acc"][-1] >= 0.50
    )


def save_training_history(
    run_dir: str,
    config: dict,
    history: dict,
    inference: dict | None = None,
) -> None:
    payload = {
        "config": config,
        "converged": is_converged_history(history),
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_bound_total": history["bound_total"][-1],
        "inference": inference,
        "history": history,
    }
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


# ==========================================
# 7. TRAINING
# ==========================================
def train_scenario(
    scenario_name: str,
    loader: DataLoader,
    n_samples: int,
    mode: str = "perfect",
    objective: str = "bound",
    *,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR_BASE,
    lr_decay_step: int = LR_DECAY_STEP,
    lr_decay_gamma: float = LR_DECAY_GAMMA,
    alpha_coeff: float = ALPHA_COEFF,
    beta_coeff: float = BETA_COEFF,
    gamma_coeff: float | None = None,
    mi_mc_samples: int = MI_MC_SAMPLES,
    seed: int = SEED,
    lipschitz_method_perfect: str = LIPSCHITZ_METHOD_PERFECT,
    hidden_dim: int = HIDDEN_DIM,
    n_u_sets: int = N_U_SETS,
    m_artificial_channels: int = M_ARTIFICIAL_CHANNELS,
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    if scenario_name not in {"erm", "l2", "proposed"}:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")
    if mode not in {"perfect", "train"}:
        raise ValueError(f"Unknown training channel mode: {mode!r}")
    if objective not in {"bound", "regularization", "heuristic"}:
        raise ValueError(f"Unknown objective: {objective!r}")
    if lipschitz_method_perfect not in {"grad", "analytical"}:
        raise ValueError("lipschitz_method_perfect must be 'grad' or 'analytical'.")
    if n_samples <= 1:
        raise ValueError("At least two training samples are required by the bound.")
    if mi_mc_samples <= 0:
        raise ValueError("mi_mc_samples must be positive.")

    if gamma_coeff is None:
        gamma_coeff = L2_COEFF if scenario_name == "l2" else GAMMA_COEFF

    set_seed(seed)
    if scenario_name in {"erm", "l2"}:
        model: nn.Module = DeterministicFC(
            IN_DIM,
            hidden_dim,
            OUT_DIM,
            m_artificial_channels,
        ).to(DEVICE)
    else:
        model = VectorizedBNNEnsemble(
            n_u_sets,
            IN_DIM,
            hidden_dim,
            OUT_DIM,
            m_artificial_channels,
        ).to(DEVICE)

    config = build_param_config(
        scenario_name=scenario_name,
        mode=mode,
        objective=objective,
        n_samples=n_samples,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lr_decay_step=lr_decay_step,
        lr_decay_gamma=lr_decay_gamma,
        alpha_coeff=alpha_coeff,
        beta_coeff=beta_coeff,
        gamma_coeff=gamma_coeff,
        hidden_dim=hidden_dim,
        n_u_sets=n_u_sets,
        m_artificial_channels=m_artificial_channels,
        mi_mc_samples=mi_mc_samples,
        lipschitz_method_perfect=lipschitz_method_perfect,
        is_bnn=model.is_bnn,
    )
    run_dir = None
    if use_cache:
        run_dir = get_run_dir(
            scenario_name=scenario_name,
            mode=mode,
            objective=objective,
            n_samples=n_samples,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            lr_decay_step=lr_decay_step,
            lr_decay_gamma=lr_decay_gamma,
            alpha_coeff=alpha_coeff,
            beta_coeff=beta_coeff,
            gamma_coeff=gamma_coeff,
            hidden_dim=hidden_dim,
            n_u_sets=n_u_sets,
            m_artificial_channels=m_artificial_channels,
            mi_mc_samples=mi_mc_samples,
            lipschitz_method_perfect=lipschitz_method_perfect,
            is_bnn=model.is_bnn,
        )
        weights_path = os.path.join(run_dir, "weights.pth")
        history_path = os.path.join(run_dir, "history.json")
        if os.path.exists(weights_path) and os.path.exists(history_path):
            print(f"Loading cached weights: {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            with open(history_path, "r", encoding="utf-8") as file:
                history = json.load(file)["history"]
            model.run_dir = run_dir
            return model, history

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_decay_step,
            gamma=lr_decay_gamma,
        )
        if lr_decay_step > 0
        else None
    )
    kl_ch_total = channel_kl_total(hidden_dim)
    channel_penalty = get_channel_penalty(hidden_dim).to(DEVICE)
    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)

    history = {
        "train_loss": [],
        "train_acc": [],
        "bound_total": [],
        "bound_term1": [],
        "bound_term2": [],
        "bound_term3": [],
        "channel_overfit_kl": [],
        "kl_ch_total": [],
        "mixture_kl": [],
        "component_expected_kl": [],
        "k_hat": [],
        "channel_penalty": [],
        "k_hat_channel_penalty": [],
        "applied_regularization": [],
        "nonfinite_loss": False,
    }

    stop_training = False
    for epoch in range(epochs):
        totals = {
            "ce": 0.0,
            "reg": 0.0,
            "bound": 0.0,
            "term1": 0.0,
            "term2": 0.0,
            "term3": 0.0,
            "channel_overfit_kl": 0.0,
            "kl_ch": 0.0,
            "mixture_kl": 0.0,
            "component_kl": 0.0,
            "k_hat": 0.0,
            "channel_penalty": 0.0,
            "k_hat_channel_penalty": 0.0,
        }
        processed_batches = 0
        model.train()

        for batch_x, batch_y in loader:
            processed_batches += 1
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if model.is_bnn:
                theta = model.sample_theta(1).squeeze(0)
                output = model(batch_x, theta, mode=mode)
                expanded_y = batch_y.unsqueeze(0).expand(model.k, -1)
                cross_entropy = F.cross_entropy(
                    output.reshape(-1, model.out_dim),
                    expanded_y.reshape(-1),
                )
                mixture_kl, channel_overfit_kl = (
                    compute_mixture_kl_and_channel_overfit(
                        model,
                        num_samples=mi_mc_samples,
                    )
                )
            else:
                output = model(batch_x, mode=mode)
                cross_entropy = F.cross_entropy(output, batch_y)
                mixture_kl = cross_entropy.new_tensor(0.0)
                channel_overfit_kl = cross_entropy.new_tensor(0.0)

            mixture_kl = torch.clamp(mixture_kl, min=0.0)
            channel_overfit_kl = torch.clamp(channel_overfit_kl, min=0.0)
            loss = cross_entropy
            regularizer = cross_entropy.new_tensor(0.0)
            bound = cross_entropy.new_tensor(0.0)
            channel_shift = cross_entropy.new_tensor(0.0)
            channel_overfit = cross_entropy.new_tensor(0.0)
            model_complexity = cross_entropy.new_tensor(0.0)
            component_expected_kl = cross_entropy.new_tensor(0.0)
            k_hat = cross_entropy.new_tensor(0.0)
            channel_penalty_value = cross_entropy.new_tensor(0.0)

            if mode == "perfect" and model.is_bnn:
                if lipschitz_method_perfect == "grad":
                    grad_theta = torch.autograd.grad(
                        cross_entropy,
                        theta,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    k_hat = torch.norm(grad_theta, p=2)
                else:
                    k_hat = model.compute_analytical_lipschitz(
                        batch_x,
                        theta,
                        mode=mode,
                    )
                channel_penalty_value = channel_penalty
                channel_shift = k_hat * channel_penalty_value
                model_complexity = torch.sqrt(
                    (2 * SIGMA_SQ / (n_samples - 1))
                    * (mixture_kl + complexity_term)
                )
                bound = channel_shift + model_complexity
            elif mode == "train":
                channel_shift = cross_entropy.new_tensor(
                    np.sqrt(2 * SIGMA_ART_SQ * kl_ch_total)
                )
                channel_overfit = torch.sqrt(
                    (2 * SIGMA_ART_SQ / m_artificial_channels)
                    * channel_overfit_kl
                )
                model_complexity = torch.sqrt(
                    (2 * SIGMA_SQ / (n_samples - 1))
                    * (mixture_kl + complexity_term)
                )
                bound = channel_shift + channel_overfit + model_complexity

            if scenario_name == "l2":
                l2_penalty = sum(
                    torch.sum(parameter**2) for parameter in model.parameters()
                )
                regularizer = l2_penalty
                if objective == "regularization":
                    loss = cross_entropy + l2_penalty
                elif objective == "heuristic":
                    loss = cross_entropy + gamma_coeff * l2_penalty
                logged_term3 = l2_penalty
            elif scenario_name == "proposed":
                regularizer = bound
                if mode == "train":
                    component_expected_kl = compute_component_expected_kl(model)
                if objective == "bound":
                    loss = cross_entropy + bound
                elif objective == "heuristic":
                    loss = cross_entropy + (
                        alpha_coeff * channel_shift
                        + beta_coeff * channel_overfit
                        + gamma_coeff * model_complexity
                    )
                logged_term3 = model_complexity
            else:
                logged_term3 = model_complexity

            if not torch.isfinite(loss):
                print(f"Stopping early: non-finite loss at epoch {epoch + 1}.")
                history["nonfinite_loss"] = True
                stop_training = True
                break

            loss.backward()
            optimizer.step()

            totals["ce"] += cross_entropy.item()
            totals["reg"] += regularizer.item()
            totals["bound"] += bound.item()
            totals["term1"] += channel_shift.item()
            totals["term2"] += channel_overfit.item()
            totals["term3"] += logged_term3.item()
            totals["channel_overfit_kl"] += channel_overfit_kl.item()
            totals["kl_ch"] += kl_ch_total if mode == "train" else 0.0
            totals["mixture_kl"] += mixture_kl.item()
            totals["component_kl"] += component_expected_kl.item()
            totals["k_hat"] += k_hat.item()
            totals["channel_penalty"] += channel_penalty_value.item()
            totals["k_hat_channel_penalty"] += (
                k_hat * channel_penalty_value
            ).item()

        divisor = max(processed_batches, 1)
        train_loss, train_acc = evaluate_model(model, loader, mode=mode)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["bound_total"].append(totals["bound"] / divisor)
        history["bound_term1"].append(totals["term1"] / divisor)
        history["bound_term2"].append(totals["term2"] / divisor)
        history["bound_term3"].append(totals["term3"] / divisor)
        history["channel_overfit_kl"].append(
            totals["channel_overfit_kl"] / divisor
        )
        history["kl_ch_total"].append(totals["kl_ch"] / divisor)
        history["mixture_kl"].append(totals["mixture_kl"] / divisor)
        history["component_expected_kl"].append(
            totals["component_kl"] / divisor
        )
        history["k_hat"].append(totals["k_hat"] / divisor)
        history["channel_penalty"].append(totals["channel_penalty"] / divisor)
        history["k_hat_channel_penalty"].append(
            totals["k_hat_channel_penalty"] / divisor
        )
        history["applied_regularization"].append(totals["reg"] / divisor)

        if scheduler is not None:
            scheduler.step()
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs):
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"CE {totals['ce'] / divisor:.4f} | "
                f"Reg {totals['reg'] / divisor:.4f} | "
                f"Bound {totals['bound'] / divisor:.4f} | "
                f"Train Acc {train_acc * 100:.2f}%"
            )
        if stop_training:
            break

    if use_cache and run_dir is not None:
        torch.save(model.state_dict(), os.path.join(run_dir, "weights.pth"))
        plot_training_metrics(history, scenario_name, mode, run_dir)
        plot_bound_decomposition(history, scenario_name, mode, run_dir)
        save_training_history(run_dir, config, history)
    model.run_dir = run_dir
    model.training_history = history
    model.converged = is_converged_history(history)
    return model, history


# ==========================================
# 8. INFERENCE CHANNEL EVALUATION
# ==========================================
def evaluate_inference(
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    repeats: int = EVAL_REPEATS,
    weight_mc_samples: int = INFERENCE_WEIGHT_SAMPLES,
) -> tuple[float, float]:
    if repeats <= 0 or weight_mc_samples <= 0:
        raise ValueError("Evaluation repeat/sample counts must be positive.")
    model.eval()
    device = model_device(model)
    loss_runs: list[float] = []
    accuracy_runs: list[float] = []
    set_seed(seed)

    with torch.no_grad():
        for _ in range(repeats):
            total_loss = 0.0
            total_correct = 0.0
            total = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                if getattr(model, "is_bnn", False):
                    batch_loss = 0.0
                    batch_correct = 0.0
                    for _ in range(weight_mc_samples):
                        theta = model.sample_theta(1).squeeze(0)
                        output = model(batch_x, theta, mode="test")
                        expanded_y = batch_y.unsqueeze(0).expand(model.k, -1)
                        batch_loss += F.cross_entropy(
                            output.reshape(-1, model.out_dim),
                            expanded_y.reshape(-1),
                        ).item()
                        batch_correct += (
                            (output.argmax(dim=-1) == expanded_y).sum().item()
                            / model.k
                        )
                    average_loss = batch_loss / weight_mc_samples
                    average_correct = batch_correct / weight_mc_samples
                else:
                    output = model(batch_x, mode="test")
                    average_loss = F.cross_entropy(output, batch_y).item()
                    average_correct = (
                        output.argmax(dim=-1) == batch_y
                    ).sum().item()
                total_loss += average_loss * batch_y.size(0)
                total_correct += average_correct
                total += batch_y.size(0)
            loss_runs.append(total_loss / max(total, 1))
            accuracy_runs.append(total_correct / max(total, 1))
    return float(np.mean(loss_runs)), float(np.mean(accuracy_runs))


def free_memory(model: nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ==========================================
# 9. NON-SWEEP EXPERIMENT
# ==========================================
SINGLE_SCENARIOS = {
    "erm-perfect": ("erm", "perfect", "bound"),
    "erm-train": ("erm", "train", "bound"),
    "l2-perfect": ("l2", "perfect", "heuristic"),
    "l2-train": ("l2", "train", "heuristic"),
    "proposed-perfect": ("proposed", "perfect", "heuristic"),
    "proposed-train": ("proposed", "train", "heuristic"),
}


def parse_name_list(raw: str, allowed: Iterable[str], argument_name: str) -> list[str]:
    allowed_set = set(allowed)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - allowed_set)
    if unknown:
        raise ValueError(
            f"Unknown {argument_name} value(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed_set))}."
        )
    if not values:
        raise ValueError(f"At least one {argument_name} value is required.")
    return values


def run_single_experiment(args: argparse.Namespace) -> None:
    selected_names = (
        list(SINGLE_SCENARIOS)
        if args.single_scenarios == "all"
        else parse_name_list(
            args.single_scenarios,
            SINGLE_SCENARIOS,
            "single scenario",
        )
    )
    train_loader, test_loader, n_train = get_dataloaders(
        args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        download=args.download,
    )
    results = []
    for name in selected_names:
        scenario_name, mode, objective = SINGLE_SCENARIOS[name]
        gamma_coeff = L2_COEFF if scenario_name == "l2" else GAMMA_COEFF
        print(f"\nRunning {name} ...")
        model, history = train_scenario(
            scenario_name,
            train_loader,
            n_train,
            mode=mode,
            objective=objective,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lr_decay_step=args.lr_decay_step,
            lr_decay_gamma=args.lr_decay_gamma,
            gamma_coeff=gamma_coeff,
            mi_mc_samples=args.mi_mc_samples,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            n_u_sets=args.n_u_sets,
            m_artificial_channels=args.m_artificial_channels,
            use_cache=args.cache,
            verbose=True,
        )
        population_loss, population_accuracy = evaluate_inference(
            model,
            test_loader,
            seed=args.seed,
            repeats=args.eval_repeats,
            weight_mc_samples=args.inference_weight_samples,
        )
        result = {
            "name": name,
            "scenario": scenario_name,
            "mode": mode,
            "objective": objective,
            "population_loss": population_loss,
            "population_accuracy": population_accuracy,
            "empirical_loss": history["train_loss"][-1],
            "empirical_accuracy": history["train_acc"][-1],
            "bound": history["bound_total"][-1],
        }
        results.append(result)

        if args.cache and model.run_dir is not None:
            history_path = os.path.join(model.run_dir, "history.json")
            with open(history_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            payload["inference"] = {
                "loss": population_loss,
                "accuracy": population_accuracy,
                "repeats": args.eval_repeats,
                "weight_mc_samples": args.inference_weight_samples,
            }
            with open(history_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
        free_memory(model)

    print("\nFINAL MNIST INFERENCE RESULTS (P_ch)")
    print("=" * 94)
    print(
        f"{'Method':<25} {'Pop. Loss':>12} {'Pop. Acc':>12} "
        f"{'Emp. Loss':>12} {'Emp. Acc':>12} {'Bound':>12}"
    )
    for result in results:
        print(
            f"{result['name']:<25} "
            f"{result['population_loss']:>12.4f} "
            f"{result['population_accuracy']:>12.4f} "
            f"{result['empirical_loss']:>12.4f} "
            f"{result['empirical_accuracy']:>12.4f} "
            f"{result['bound']:>12.4f}"
        )
    print("=" * 94)

    if args.cache:
        summary_path = os.path.join(RESULTS_DIR, "single_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"Single-run summary saved to {summary_path}")


# ==========================================
# 10. PAIRED ERM/PROPOSED PARAMETER SWEEP
# ==========================================
SWEEP_SCENARIOS = ("erm", "proposed")

# These settings must be identical within every ERM/proposed comparison pair.
SHARED_SWEEP_FIELDS = (
    "dataset",
    "seed",
    "hidden_dim",
    "batch_size",
    "n_samples",
    "epochs",
    "m_artificial_channels",
    "lr",
    "lr_decay_step",
    "lr_decay_gamma",
    "mode",
)

# These identify a proposed-method parameter setting when results are aggregated
# across random seeds.  Shared training settings are included so future grid
# expansions cannot accidentally mix unlike configurations.
PROPOSED_PARAMETER_FIELDS = (
    "hidden_dim",
    "batch_size",
    "n_samples",
    "epochs",
    "n_u_sets",
    "m_artificial_channels",
    "lr",
    "lr_decay_step",
    "lr_decay_gamma",
    "mi_mc_samples",
    "alpha_coeff",
    "beta_coeff",
    "gamma_coeff",
)


def run_sweep_task(
    task_config: dict,
    repeats: int = EVAL_REPEATS,
    weight_mc_samples: int = INFERENCE_WEIGHT_SAMPLES,
) -> dict:
    """Execute one sweep configuration in an isolated CPU worker."""
    configure_device(task_config["device"])
    torch.set_num_threads(1)
    set_seed(task_config["seed"])
    train_loader, test_loader, n_train = get_dataloaders(
        task_config["n_samples"],
        batch_size=task_config["batch_size"],
        seed=task_config["seed"],
        download=False,
    )
    model, history = train_scenario(
        task_config["scenario"],
        train_loader,
        n_train,
        mode=task_config["mode"],
        objective=task_config["objective"],
        batch_size=task_config["batch_size"],
        epochs=task_config["epochs"],
        lr=task_config["lr"],
        lr_decay_step=task_config["lr_decay_step"],
        lr_decay_gamma=task_config["lr_decay_gamma"],
        alpha_coeff=task_config.get("alpha_coeff", ALPHA_COEFF),
        beta_coeff=task_config.get("beta_coeff", BETA_COEFF),
        gamma_coeff=task_config.get("gamma_coeff"),
        mi_mc_samples=task_config["mi_mc_samples"],
        seed=task_config["seed"],
        hidden_dim=task_config["hidden_dim"],
        n_u_sets=task_config["n_u_sets"],
        m_artificial_channels=task_config["m_artificial_channels"],
        use_cache=task_config["use_cache"],
        verbose=task_config["verbose"],
    )
    population_loss, population_accuracy = evaluate_inference(
        model,
        test_loader,
        seed=task_config["seed"],
        repeats=repeats,
        weight_mc_samples=weight_mc_samples,
    )
    result = {
        **task_config,
        "device": str(DEVICE),
        "population_loss": population_loss,
        "population_accuracy": population_accuracy,
        "empirical_loss": history["train_loss"][-1],
        "empirical_accuracy": history["train_acc"][-1],
        "bound": history["bound_total"][-1],
        "converged": is_converged_history(history),
    }
    free_memory(model)
    return result


def build_sweep_tasks(args: argparse.Namespace) -> list[dict]:
    master_rng = random.Random(args.master_seed)
    seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(args.num_seeds)]

    hidden_dim_grid = [args.hidden_dim]
    n_u_sets_grid = [args.n_u_sets]
    m_artificial_channels_grid = [args.m_artificial_channels]
    lr_grid = [args.lr]
    lr_decay_step_grid = [args.lr_decay_step]
    lr_decay_gamma_grid = [args.lr_decay_gamma]
    epochs_grid = [args.epochs]
    beta_coeff_grid = [BETA_COEFF]
    proposed_gamma_grid = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    tasks: list[dict] = []
    base_grid = itertools.product(
        seeds,
        hidden_dim_grid,
        m_artificial_channels_grid,
        epochs_grid,
        lr_grid,
        lr_decay_step_grid,
        lr_decay_gamma_grid,
    )
    for (
        seed,
        hidden_dim,
        m_artificial_channels,
        epochs,
        lr,
        lr_decay_step,
        lr_decay_gamma,
    ) in base_grid:
        shared_hyperparameters = {
            "dataset": DATASET_NAME,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "batch_size": args.batch_size,
            "n_samples": args.n_samples,
            "epochs": epochs,
            "m_artificial_channels": m_artificial_channels,
            "lr": lr,
            "lr_decay_step": lr_decay_step,
            "lr_decay_gamma": lr_decay_gamma,
            "mode": "train",
        }
        comparison_id = config_hash(shared_hyperparameters)
        base = {
            **shared_hyperparameters,
            "comparison_id": comparison_id,
            "mi_mc_samples": args.mi_mc_samples,
            "device": str(DEVICE),
            "use_cache": args.sweep_cache,
            "verbose": False,
        }
        # Exactly one ERM baseline is generated for this shared configuration.
        tasks.append(
            {
                **base,
                "scenario": "erm",
                "objective": "bound",
                "n_u_sets": args.n_u_sets,
                "alpha_coeff": None,
                "beta_coeff": None,
                "gamma_coeff": None,
            }
        )

        # Every proposed task shares all optimization/data/channel settings with
        # the ERM task above.  Only method-specific parameters are swept.
        for n_u_sets in n_u_sets_grid:
            for beta_coeff in beta_coeff_grid:
                for gamma_coeff in proposed_gamma_grid:
                    tasks.append(
                        {
                            **base,
                            "scenario": "proposed",
                            "objective": "heuristic",
                            "n_u_sets": n_u_sets,
                            "alpha_coeff": ALPHA_COEFF,
                            "beta_coeff": beta_coeff,
                            "gamma_coeff": gamma_coeff,
                        }
                    )
    return tasks


def _result_metrics(result: dict) -> dict:
    return {
        "population_loss": result["population_loss"],
        "population_accuracy": result["population_accuracy"],
        "empirical_loss": result["empirical_loss"],
        "empirical_accuracy": result["empirical_accuracy"],
        "bound": result["bound"],
        "converged": result["converged"],
    }


def pair_erm_and_proposed_results(
    results: list[dict],
) -> tuple[list[dict], dict]:
    """Pair each proposed run with ERM using the full shared configuration."""
    erm_by_comparison_id: dict[str, dict] = {}
    duplicate_erm_ids: list[str] = []
    for result in results:
        if result["scenario"] != "erm":
            continue
        comparison_id = result["comparison_id"]
        if comparison_id in erm_by_comparison_id:
            duplicate_erm_ids.append(comparison_id)
        else:
            erm_by_comparison_id[comparison_id] = result

    comparisons: list[dict] = []
    unmatched_proposed: list[dict] = []
    matched_erm_ids: set[str] = set()
    for proposed in results:
        if proposed["scenario"] != "proposed":
            continue
        comparison_id = proposed["comparison_id"]
        erm = erm_by_comparison_id.get(comparison_id)
        if erm is None:
            unmatched_proposed.append(
                {
                    "comparison_id": comparison_id,
                    "seed": proposed["seed"],
                    "proposed_hyperparameters": {
                        field: proposed[field] for field in PROPOSED_PARAMETER_FIELDS
                    },
                }
            )
            continue

        mismatched_fields = [
            field
            for field in SHARED_SWEEP_FIELDS
            if proposed[field] != erm[field]
        ]
        if mismatched_fields:
            raise RuntimeError(
                f"Comparison {comparison_id} has mismatched shared fields: "
                f"{mismatched_fields}"
            )

        matched_erm_ids.add(comparison_id)
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "seed": proposed["seed"],
                "shared_hyperparameters": {
                    field: proposed[field] for field in SHARED_SWEEP_FIELDS
                },
                "proposed_hyperparameters": {
                    field: proposed[field] for field in PROPOSED_PARAMETER_FIELDS
                },
                "erm": _result_metrics(erm),
                "proposed": _result_metrics(proposed),
                "difference": {
                    "population_accuracy": (
                        proposed["population_accuracy"] - erm["population_accuracy"]
                    ),
                    "empirical_accuracy": (
                        proposed["empirical_accuracy"] - erm["empirical_accuracy"]
                    ),
                    "population_loss_reduction": (
                        erm["population_loss"] - proposed["population_loss"]
                    ),
                    "empirical_loss_reduction": (
                        erm["empirical_loss"] - proposed["empirical_loss"]
                    ),
                },
            }
        )

    unmatched_erm_ids = sorted(set(erm_by_comparison_id) - matched_erm_ids)
    pairing_status = {
        "duplicate_erm_comparison_ids": sorted(set(duplicate_erm_ids)),
        "unmatched_erm_comparison_ids": unmatched_erm_ids,
        "unmatched_proposed_runs": unmatched_proposed,
        "all_completed_proposed_runs_are_paired": not unmatched_proposed,
    }
    return comparisons, pairing_status


def summarize_proposed_parameters(comparisons: list[dict]) -> list[dict]:
    """Aggregate each proposed parameter setting across its paired seeds."""
    grouped: dict[str, list[dict]] = {}
    parameter_configs: dict[str, dict] = {}
    for comparison in comparisons:
        parameters = comparison["proposed_hyperparameters"]
        parameter_id = config_hash(parameters)
        grouped.setdefault(parameter_id, []).append(comparison)
        parameter_configs[parameter_id] = parameters

    summaries: list[dict] = []
    for parameter_id, group in grouped.items():
        proposed_accuracies = [item["proposed"]["population_accuracy"] for item in group]
        erm_accuracies = [item["erm"]["population_accuracy"] for item in group]
        accuracy_differences = [
            item["difference"]["population_accuracy"] for item in group
        ]
        proposed_losses = [item["proposed"]["population_loss"] for item in group]
        erm_losses = [item["erm"]["population_loss"] for item in group]
        loss_reductions = [
            item["difference"]["population_loss_reduction"] for item in group
        ]
        summaries.append(
            {
                "parameter_id": parameter_id,
                "proposed_hyperparameters": parameter_configs[parameter_id],
                "num_paired_seeds": len(group),
                "seeds": sorted(item["seed"] for item in group),
                "mean_proposed_population_accuracy": float(
                    np.mean(proposed_accuracies)
                ),
                "std_proposed_population_accuracy": float(
                    np.std(proposed_accuracies)
                ),
                "mean_erm_population_accuracy": float(np.mean(erm_accuracies)),
                "mean_population_accuracy_difference": float(
                    np.mean(accuracy_differences)
                ),
                "std_population_accuracy_difference": float(
                    np.std(accuracy_differences)
                ),
                "proposed_win_rate": float(
                    np.mean([difference > 0 for difference in accuracy_differences])
                ),
                "mean_proposed_population_loss": float(np.mean(proposed_losses)),
                "mean_erm_population_loss": float(np.mean(erm_losses)),
                "mean_population_loss_reduction": float(np.mean(loss_reductions)),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["mean_proposed_population_accuracy"],
            item["mean_population_accuracy_difference"],
        ),
        reverse=True,
    )
    return summaries


def summarize_sweep(
    results: list[dict],
    failed: int,
) -> tuple[dict, list[dict], list[dict]]:
    comparisons, pairing_status = pair_erm_and_proposed_results(results)
    parameter_summaries = summarize_proposed_parameters(comparisons)
    accuracy_wins = [
        comparison
        for comparison in comparisons
        if comparison["difference"]["population_accuracy"] > 0
    ]
    summary = {
        "comparison": "proposed_vs_erm_on_training_channel",
        "shared_hyperparameter_fields": list(SHARED_SWEEP_FIELDS),
        "completed_tasks": len(results),
        "failed_tasks": failed,
        "completed_erm_tasks": sum(
            result["scenario"] == "erm" for result in results
        ),
        "completed_proposed_tasks": sum(
            result["scenario"] == "proposed" for result in results
        ),
        "paired_proposed_results": len(comparisons),
        "pairing_status": pairing_status,
        "proposed_beats_erm_anywhere": bool(accuracy_wins),
        "paired_runs_where_proposed_beats_erm": len(accuracy_wins),
        "best_paired_run_by_accuracy_margin": (
            max(
                accuracy_wins,
                key=lambda item: item["difference"]["population_accuracy"],
            )
            if accuracy_wins
            else None
        ),
        "best_proposed_parameters_by_mean_accuracy": (
            parameter_summaries[0] if parameter_summaries else None
        ),
        "best_proposed_parameters_by_mean_improvement": (
            max(
                parameter_summaries,
                key=lambda item: item["mean_population_accuracy_difference"],
            )
            if parameter_summaries
            else None
        ),
    }
    return summary, comparisons, parameter_summaries


def run_parameter_sweep(args: argparse.Namespace) -> None:
    tasks = build_sweep_tasks(args)
    print(f"Total sweep tasks: {len(tasks)}")
    if args.dry_run:
        scenario_counts = {
            scenario: sum(task["scenario"] == scenario for task in tasks)
            for scenario in sorted({task["scenario"] for task in tasks})
        }
        print(f"Dry run only; task counts: {scenario_counts}")
        return

    # Download/check MNIST once before workers are spawned.  Workers only read it.
    load_mnist_datasets(download=args.download)
    results_path = os.path.join(RESULTS_DIR, "sweep_results.jsonl")
    errors_path = os.path.join(RESULTS_DIR, "sweep_errors.jsonl")
    for path in (results_path, errors_path):
        with open(path, "w", encoding="utf-8"):
            pass

    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    max_workers = args.max_workers or allocated_cpus
    max_workers = max(1, min(max_workers, allocated_cpus, len(tasks)))
    if DEVICE.type != "cpu" and max_workers > 1:
        print("A shared accelerator uses one sweep worker to avoid device OOM.")
        max_workers = 1
    print(f"Starting sweep with {max_workers} worker(s) on {DEVICE}.")

    results: list[dict] = []
    failed = 0
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as executor:
        future_to_task = {
            executor.submit(
                run_sweep_task,
                task,
                args.eval_repeats,
                args.inference_weight_samples,
            ): task
            for task in tasks
        }
        progress_interval = max(1, len(tasks) // 20)
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_task),
            start=1,
        ):
            try:
                result = future.result()
                results.append(result)
                with open(results_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(result) + "\n")
            except Exception as exception:
                failed += 1
                failure = {
                    "task": future_to_task[future],
                    "error": repr(exception),
                }
                with open(errors_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(failure) + "\n")
                print(f"Sweep task failed: {failure}")
            if completed % progress_interval == 0 or completed == len(tasks):
                print(f"Progress: {completed}/{len(tasks)} tasks completed.")

    summary, comparisons, parameter_summaries = summarize_sweep(results, failed)
    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
    comparisons_path = os.path.join(RESULTS_DIR, "sweep_comparisons.jsonl")
    parameter_summary_path = os.path.join(
        RESULTS_DIR,
        "sweep_parameter_summary.json",
    )
    with open(comparisons_path, "w", encoding="utf-8") as file:
        for comparison in comparisons:
            file.write(json.dumps(comparison) + "\n")
    with open(parameter_summary_path, "w", encoding="utf-8") as file:
        json.dump(parameter_summaries, file, indent=2)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Sweep results saved to {results_path}")
    print(f"Paired ERM/proposed comparisons saved to {comparisons_path}")
    print(f"Proposed parameter summary saved to {parameter_summary_path}")
    print(f"Sweep summary saved to {summary_path}")
    if failed:
        raise RuntimeError(f"{failed} sweep task(s) failed; see {errors_path}.")


# ==========================================
# 11. COMMAND-LINE ENTRY POINT
# ==========================================
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the test12 simulation on MNIST.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "sweep"),
        default="single",
        help="single is the safe default; sweep must be requested explicitly.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR_BASE)
    parser.add_argument("--lr-decay-step", type=int, default=LR_DECAY_STEP)
    parser.add_argument("--lr-decay-gamma", type=float, default=LR_DECAY_GAMMA)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--n-u-sets", type=int, default=N_U_SETS)
    parser.add_argument(
        "--m-artificial-channels",
        type=int,
        default=M_ARTIFICIAL_CHANNELS,
    )
    parser.add_argument("--mi-mc-samples", type=int, default=MI_MC_SAMPLES)
    parser.add_argument("--eval-repeats", type=int, default=EVAL_REPEATS)
    parser.add_argument(
        "--inference-weight-samples",
        type=int,
        default=INFERENCE_WEIGHT_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--single-scenarios",
        default="all",
        help=(
            "Comma-separated subset of: "
            + ", ".join(SINGLE_SCENARIOS)
            + "; default: all."
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report sweep tasks without training any model.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow torchvision to download missing MNIST files into $DATASET.",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache non-sweep weights and plots.",
    )
    parser.add_argument(
        "--sweep-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache every sweep model (off by default to limit server storage).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "n_samples",
        "batch_size",
        "epochs",
        "hidden_dim",
        "n_u_sets",
        "m_artificial_channels",
        "mi_mc_samples",
        "eval_repeats",
        "inference_weight_samples",
        "num_seeds",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0 < args.lr_decay_gamma <= 1:
        raise ValueError("--lr-decay-gamma must be in (0, 1].")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    configure_device(args.device)
    print(f"Using device: {DEVICE}")
    print(f"MNIST root: {dataset_root()}")
    print(f"D(P_ch || P_art): {channel_kl_total(args.hidden_dim):.4f}")
    print(
        f"Architecture: {IN_DIM} -> {args.hidden_dim} -> {OUT_DIM} "
        "with the channel after ReLU"
    )

    if args.mode == "single":
        run_single_experiment(args)
    else:
        run_parameter_sweep(args)


if __name__ == "__main__":
    main()
