"""Single-posterior models and the journal's conditioned PAC-Bayes terms.

The constants 1/4 apply to a loss in [0, 1], by Hoeffding's lemma.  They do
not certify unbounded cross entropy.  A raw-cross-entropy training objective
using these penalties is a tunable surrogate, not a numerical certificate.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_kl(mu: torch.Tensor, rho: torch.Tensor, prior_std: float = 1.0) -> torch.Tensor:
    """Exact KL(diagonal Gaussian || fixed zero-mean isotropic Gaussian)."""
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")
    variance_ratio = (F.softplus(rho) / prior_std).square()
    return .5 * (variance_ratio + (mu / prior_std).square() - 1 - variance_ratio.log()).sum()


def theorem1_terms(kl: torch.Tensor, n: int, m: int, delta: float = .05,
                   sigma_data_sq: float = .25, sigma_channel_sq: float = .25,
                   perfect_training: bool = False) -> dict[str, torch.Tensor]:
    """Separated conditioned terms, using a union-bound confidence allocation.

    Perfect training has a deterministic channel bank: its channel complexity
    is exactly zero.  A deployment-shift term is a separate question.
    """
    if n <= 1 or (m <= 1 and not perfect_training) or not 0 < delta < 1:
        raise ValueError("Require n>1, m>1 (unless perfect), and 0<delta<1")
    if min(sigma_data_sq, sigma_channel_sq) < 0:
        raise ValueError("Sub-Gaussian variance proxies must be nonnegative")
    eps_data = delta if perfect_training else delta / 2
    sample = torch.sqrt(2 * sigma_data_sq / (n - 1) *
                        (kl + math.log(math.sqrt(n) / eps_data)))
    channel = kl.new_zeros(()) if perfect_training else torch.sqrt(
        2 * sigma_channel_sq / (m - 1) *
        (kl + math.log(math.sqrt(m) / (delta / 2))))
    return {"channel_complexity": channel, "sample_complexity": sample}


def joint_complexity(kl: torch.Tensor, n: int, m: int, delta: float = .05,
                     sigma_data_sq: float = .25, sigma_channel_sq: float = .25,
                     perfect_training: bool = False) -> torch.Tensor:
    if min(sigma_data_sq, sigma_channel_sq) < 0:
        raise ValueError("Sub-Gaussian variance proxies must be nonnegative")
    if perfect_training:
        return theorem1_terms(kl, n, m, delta, sigma_data_sq,
                              sigma_channel_sq, True)["sample_complexity"]
    if n <= 1 or m <= 1 or not 0 < delta < 1:
        raise ValueError("Require n>1, m>1, and 0<delta<1")
    return torch.sqrt((2 * sigma_data_sq / (n - 1) +
                       2 * sigma_channel_sq / (m - 1)) *
                      (kl + math.log(math.sqrt(n * m) / delta)))


def validate_channel(config: Mapping) -> dict:
    config = dict(config)
    kind = config.get("kind", "gaussian")
    if kind not in {"gaussian", "outage", "perfect"}:
        raise ValueError(f"Unknown channel kind: {kind}")
    if kind == "gaussian":
        config = {"kind": kind, "gain_mean": float(config.get("gain_mean", 1.)),
                  "gain_std": float(config.get("gain_std", .1)),
                  "noise_std": float(config.get("noise_std", .1))}
        if min(config["gain_std"], config["noise_std"]) < 0:
            raise ValueError("Channel standard deviations must be nonnegative")
    elif kind == "outage":
        config = {"kind": kind, "outage_prob": float(config.get("outage_prob", .1))}
        if not 0 <= config["outage_prob"] <= 1:
            raise ValueError("outage_prob must lie in [0,1]")
    else:
        config = {"kind": "perfect"}
    return config


def channel_shift_diagnostic(deployment: Mapping, training: Mapping, width: int) -> dict:
    """KL in deployment||training direction, for independent feature channels.

    The shift sqrt(KL/2) uses the normalized bounded loss's variance proxy
    1/4.  Singular-support mismatch is unavailable here; a Lipschitz argument
    would require its own justified channel metric and constant.
    """
    deployment, training = validate_channel(deployment), validate_channel(training)

    def normal_kl(mean_p, std_p, mean_q, std_q):
        if std_p == 0 or std_q == 0:
            return 0. if std_p == std_q == 0 and mean_p == mean_q else math.inf
        return math.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q)**2) / (2 * std_q**2) - .5

    if deployment == training:
        divergence = 0.
    elif {deployment["kind"], training["kind"]} <= {"perfect", "gaussian"}:
        perfect = {"kind": "gaussian", "gain_mean": 1., "gain_std": 0., "noise_std": 0.}
        p = perfect if deployment["kind"] == "perfect" else deployment
        q = perfect if training["kind"] == "perfect" else training
        divergence = width * (normal_kl(p["gain_mean"], p["gain_std"], q["gain_mean"], q["gain_std"]) +
                              normal_kl(0., p["noise_std"], 0., q["noise_std"]))
    elif {deployment["kind"], training["kind"]} <= {"perfect", "outage"}:
        p = deployment.get("outage_prob", 0.)
        q = training.get("outage_prob", 0.)
        divergence = 0.
        for prob_p, prob_q in ((p, q), (1-p, 1-q)):
            if prob_p:
                divergence += math.inf if prob_q == 0 else prob_p * math.log(prob_p / prob_q)
        divergence *= width
    else:
        divergence = math.inf
    finite = math.isfinite(divergence)
    return dict(channel_kl_deployment_to_training=max(divergence, 0.) if finite else None,
                channel_shift_complexity=math.sqrt(max(divergence, 0.) / 2) if finite else None,
                status="finite KL for bounded loss" if finite else "KL unavailable: singular or unsupported channel supports",
                feature_coordinates=width)


class GaussianMLP(nn.Module):
    """One diagonal-Gaussian posterior, with the channel after hidden ReLU."""
    def __init__(self, input_dim: int, width: int, classes: int,
                 stochastic: bool = True, posterior_std: float = .01):
        super().__init__()
        self.input_dim, self.width, self.classes = input_dim, width, classes
        self.stochastic = stochastic
        self.d1 = input_dim * width + width
        self.d = self.d1 + width * classes + classes
        self.mu = nn.Parameter(torch.empty(self.d))
        self.rho = nn.Parameter(torch.full((self.d,), math.log(math.expm1(posterior_std))),
                                requires_grad=stochastic)
        with torch.no_grad():
            w1, b1, w2, b2 = self.split(self.mu)
            w1.uniform_(-1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))
            b1.zero_()
            w2.uniform_(-1 / math.sqrt(width), 1 / math.sqrt(width))
            b2.zero_()

    def split(self, theta: torch.Tensor):
        a = self.input_dim * self.width
        b = self.d1
        c = b + self.width * self.classes
        return (theta[:a].view(self.input_dim, self.width), theta[a:b],
                theta[b:c].view(self.width, self.classes), theta[c:])

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if not self.stochastic:
            return self.mu
        noise = torch.randn(self.mu.shape, device=self.mu.device,
                            dtype=self.mu.dtype, generator=generator)
        return self.mu + F.softplus(self.rho) * noise

    def forward(self, x: torch.Tensor, theta: torch.Tensor,
                bank: "ChannelBank", source: str = "bank", channel: Mapping | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        w1, b1, w2, b2 = self.split(theta)
        hidden = F.relu(x.flatten(1) @ w1 + b1)
        if not self.training:
            self.last_feature_power = hidden.detach().square().mean().item()
        return bank.apply(hidden, source, channel, generator) @ w2 + b2


class ChannelBank(nn.Module):
    """A fixed independently seeded bank reused throughout training.

    Minibatches sample an independent bank index for each example.  This gives
    an unbiased stochastic estimator of the Cartesian data-by-channel risk.
    A bank entry is a complete diagonal gain/noise vector (or outage mask).
    """
    def __init__(self, width: int, m: int, config: Mapping, seed: int):
        super().__init__()
        self.width, self.m = width, m
        self.config = validate_channel(config)
        generator = torch.Generator().manual_seed(seed)
        gain, noise = self.draw(m, self.config, generator)
        self.register_buffer("gain", gain)
        self.register_buffer("noise", noise)

    def draw(self, count: int, config: Mapping, generator: torch.Generator):
        config = validate_channel(config)
        shape = (count, self.width)
        if config["kind"] == "perfect":
            return torch.ones(shape), torch.zeros(shape)
        if config["kind"] == "outage":
            gain = (torch.rand(shape, generator=generator) >= config["outage_prob"]).float()
            return gain, torch.zeros(shape)
        gain = torch.randn(shape, generator=generator) * config["gain_std"] + config["gain_mean"]
        noise = torch.randn(shape, generator=generator) * config["noise_std"]
        return gain, noise

    def apply(self, hidden: torch.Tensor, source: str = "bank",
              config: Mapping | None = None, generator: torch.Generator | None = None):
        if source == "perfect":
            return hidden
        generator = generator or torch.default_generator
        if source == "bank":
            indices = torch.randint(self.m, (len(hidden),), generator=generator).to(hidden.device)
            gain, noise = self.gain[indices], self.noise[indices]
        elif source == "deployment":
            gain, noise = self.draw(len(hidden), config or self.config, generator)
            gain, noise = gain.to(hidden.device), noise.to(hidden.device)
        else:
            raise ValueError(f"Unknown source: {source}")
        return hidden * gain + noise
