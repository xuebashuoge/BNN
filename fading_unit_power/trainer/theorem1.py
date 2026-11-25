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

try:
    from .diagonal_fading import diagonal_gain_kl, draw_diagonal_fading, normalize_diagonal_fading
    from .full_fading import (correlation_factor, draw_full_fading, fading_gain_kl,
                              normalize_full_fading, sample_received, validate_fading_dimensions)
except ImportError:
    from diagonal_fading import diagonal_gain_kl, draw_diagonal_fading, normalize_diagonal_fading
    from full_fading import (correlation_factor, draw_full_fading, fading_gain_kl,
                             normalize_full_fading, sample_received, validate_fading_dimensions)


def gaussian_kl(mu: torch.Tensor, rho: torch.Tensor, prior_std: float = 1.0) -> torch.Tensor:
    """Exact KL(diagonal Gaussian || fixed zero-mean isotropic Gaussian)."""
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")
    variance_ratio = (F.softplus(rho) / prior_std).square()
    return .5 * (variance_ratio + (mu / prior_std).square() - 1 - variance_ratio.log()).sum()


def gaussian_kl_to_prior(mu: torch.Tensor, rho: torch.Tensor, prior_mu: torch.Tensor,
                         prior_std: float) -> torch.Tensor:
    """Exact KL(N(mu, softplus(rho)^2) || N(prior_mu, prior_std^2 I)).

    The prior mean is a frozen tensor; gradients flow only to (mu, rho).
    """
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")
    variance_ratio = (F.softplus(rho) / prior_std).square()
    shift = ((mu - prior_mu.detach()) / prior_std).square()
    return .5 * (variance_ratio + shift - 1 - variance_ratio.log()).sum()


def theorem1_terms(kl: torch.Tensor, n: int, m: int, delta: float = .05,
                   sigma_data_sq: float = .25, sigma_channel_sq: float = .25,
                   perfect_training: bool = False, channel_kl: torch.Tensor | None = None,
                   channel_prior_count: int = 1) -> dict[str, torch.Tensor]:
    """Separated conditioned terms of Theorem 1, eps_1 = eps_2 = delta/2.

    ``kl`` is D(P_{W|US} || Q_W) for a prior independent of (U, S), used by the
    sample complexity.  ``channel_kl`` is D(P_{W|US} || Q_{W|S}) for a prior
    that may depend on S but not on U; it defaults to ``kl`` (Q_{W|S} = Q_W).
    ``channel_prior_count`` is the size of a finite family of candidate
    Q_{W|S} fixed before U is observed; a union bound replaces eps_1 by
    eps_1 / count so the minimum over the family remains valid.

    Perfect training has a deterministic channel bank: its channel complexity
    is exactly zero.  A deployment-shift term is a separate question.
    """
    if n <= 1 or (m <= 1 and not perfect_training) or not 0 < delta < 1:
        raise ValueError("Require n>1, m>1 (unless perfect), and 0<delta<1")
    if min(sigma_data_sq, sigma_channel_sq) < 0:
        raise ValueError("Sub-Gaussian variance proxies must be nonnegative")
    if channel_prior_count < 1:
        raise ValueError("channel_prior_count must be positive")
    channel_kl = kl if channel_kl is None else channel_kl
    eps_data = delta if perfect_training else delta / 2
    sample = torch.sqrt(2 * sigma_data_sq / (n - 1) *
                        (kl + math.log(math.sqrt(n) / eps_data)))
    channel = kl.new_zeros(()) if perfect_training else torch.sqrt(
        2 * sigma_channel_sq / (m - 1) *
        (channel_kl + math.log(math.sqrt(m) * channel_prior_count / (delta / 2))))
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
    if kind not in {"gaussian", "diagonal_fading", "full_fading", "outage", "perfect"}:
        raise ValueError(f"Unknown channel kind: {kind}")
    if kind == "gaussian":
        config = {"kind": kind, "gain_mean": float(config.get("gain_mean", 1.)),
                  "gain_std": float(config.get("gain_std", .1)),
                  "noise_std": float(config.get("noise_std", .1))}
        if min(config["gain_std"], config["noise_std"]) < 0:
            raise ValueError("Channel standard deviations must be nonnegative")
    elif kind == "diagonal_fading":
        config = normalize_diagonal_fading(config)
    elif kind == "full_fading":
        config = normalize_full_fading(config)
    elif kind == "outage":
        config = {"kind": kind, "outage_prob": float(config.get("outage_prob", .1))}
        if not 0 <= config["outage_prob"] <= 1:
            raise ValueError("outage_prob must lie in [0,1]")
    else:
        config = {"kind": "perfect"}
    return config


def channel_shift_diagnostic(deployment: Mapping, training: Mapping, width: int) -> dict:
    """Uniform shift bound for a loss in [0,1] over the channel law.

    The bounded-loss inequality |E_P loss-E_Q loss| <= TV(P,Q) applies without
    a channel Lipschitz assumption. For independent feature-erasure masks, the
    exact TV is a sum over the number of erased coordinates; grouping the
    2**width masks reduces it to width+1 binomial-probability differences.
    Perfect transmission is the erasure law with probability zero.

    Where KL(deployment||training) is finite, Pinsker gives sqrt(KL/2).
    Full fading uses the KL between matrix-normal gains and independent noise
    vectors. Its correlated gain covariance is handled via the two factors.
    Unsupported or singular continuous pairs retain the universal bound one.
    These bounded-loss consequences supplement, rather than instantiate, the
    draft's Wasserstein theorem. They do not supply a channel Lipschitz K.
    """
    if width <= 0 or int(width) != width:
        raise ValueError("width must be a positive integer")
    width = int(width)
    deployment, training = validate_channel(deployment), validate_channel(training)
    # The historical iid analog law is precisely the rho_c=0 endpoint.
    if "diagonal_fading" in (deployment["kind"], training["kind"]):
        iid = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)
        if deployment == iid:
            deployment = dict(kind="diagonal_fading", rho_c=0.)
        if training == iid:
            training = dict(kind="diagonal_fading", rho_c=0.)
    for config in (deployment, training):
        if config["kind"] == "full_fading":
            validate_fading_dimensions(config, width, width)

    def normal_kl(mean_p, std_p, mean_q, std_q):
        if std_p == 0 or std_q == 0:
            return 0. if std_p == std_q == 0 and mean_p == mean_q else math.inf
        return math.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q)**2) / (2 * std_q**2) - .5

    erasure_pair = {deployment["kind"], training["kind"]} <= {"perfect", "outage"}
    exact_tv = None
    if erasure_pair:
        p = deployment.get("outage_prob", 0.)
        q = training.get("outage_prob", 0.)
        divergence = 0.
        for prob_p, prob_q in ((p, q), (1-p, 1-q)):
            if prob_p:
                divergence += math.inf if prob_q == 0 else prob_p * math.log(prob_p / prob_q)
        divergence *= width

        def binomial_probability(k, probability):
            # Explicit endpoints avoid 0*log(0); lgamma avoids huge binomial
            # coefficients while exponential underflow safely gives zero mass.
            if probability == 0.:
                return float(k == 0)
            if probability == 1.:
                return float(k == width)
            log_mass = (math.lgamma(width + 1) - math.lgamma(k + 1) -
                        math.lgamma(width - k + 1) + k * math.log(probability) +
                        (width - k) * math.log1p(-probability))
            return math.exp(log_mass)

        exact_tv = min(1., .5 * math.fsum(
            abs(binomial_probability(k, p) - binomial_probability(k, q))
            for k in range(width + 1)))
    elif deployment == training:
        divergence = 0.
    elif deployment["kind"] == training["kind"] == "diagonal_fading":
        divergence = diagonal_gain_kl(deployment, training, width)
    elif deployment["kind"] == training["kind"] == "full_fading":
        divergence = fading_gain_kl(deployment, training, width, width)
        divergence += width * normal_kl(0., deployment["noise_std"], 0., training["noise_std"])
    elif {deployment["kind"], training["kind"]} <= {"perfect", "gaussian"}:
        perfect = {"kind": "gaussian", "gain_mean": 1., "gain_std": 0., "noise_std": 0.}
        p = perfect if deployment["kind"] == "perfect" else deployment
        q = perfect if training["kind"] == "perfect" else training
        divergence = width * (normal_kl(p["gain_mean"], p["gain_std"], q["gain_mean"], q["gain_std"]) +
                              normal_kl(0., p["noise_std"], 0., q["noise_std"]))
    else:
        divergence = math.inf
    finite = math.isfinite(divergence)
    pinsker = math.sqrt(max(divergence, 0.) / 2) if finite else None
    if erasure_pair:
        shift = min(1., exact_tv, pinsker if pinsker is not None else 1.)
        method = "exact_total_variation_feature_erasure"
        status = "exact feature-erasure total variation for bounded loss"
        if not finite:
            status += "; KL infinite because deployment is not absolutely continuous w.r.t. training"
    elif finite:
        shift = min(1., pinsker)
        method = "bounded_loss_pinsker_kl"
        status = "finite KL bounded-loss shift, capped at the universal bound one"
    else:
        shift = 1.
        method = "universal_bounded_loss_shift"
        status = "universal trivial bounded-loss shift; KL singular or channel-family pair unsupported; no Lipschitz claim"
    return dict(channel_kl_deployment_to_training=max(divergence, 0.) if finite else None,
                channel_shift_complexity=shift,
                channel_total_variation_exact=exact_tv,
                channel_shift_pinsker_bound=pinsker,
                shift_method=method, status=status,
                theory_basis="Uniform [0,1] bounded-loss total-variation inequality; additional to the draft's Wasserstein formulation",
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
        hidden = bank.transmit(F.relu(x.flatten(1) @ w1 + b1))
        if not self.training:
            self.last_feature_power = hidden.detach().square().mean().item()
        return bank.apply(hidden, source, channel, generator) @ w2 + b2


class ChannelBank(nn.Module):
    """A fixed independently seeded bank reused throughout training.

    Minibatches sample an independent bank index for each example.  This gives
    an unbiased stochastic estimator of the Cartesian data-by-channel risk.
    A bank entry is a complete full fading matrix and noise vector, diagonal
    gain and noise vector, or outage mask, depending on the channel.
    """
    def __init__(self, width: int, m: int, config: Mapping, seed: int, *,
                 transmit_unit_power: bool = False):
        super().__init__()
        self.width, self.m = width, m
        self.config = validate_channel(config)
        self.transmit_unit_power = transmit_unit_power
        self._fading_factors = {}
        generator = torch.Generator().manual_seed(seed)
        gain, noise = self.draw(m, self.config, generator)
        self.register_buffer("gain", gain)
        self.register_buffer("noise", noise)

    def transmit(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize each vector before any channel gain or receiver noise."""
        if not self.transmit_unit_power:
            return features
        norm = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        nonzero = norm > 0
        safe_norm = torch.where(nonzero, norm, 1.)
        unit_power = (features / safe_norm) * math.sqrt(self.width)
        # An inactive ReLU encoder can produce an all-zero vector.
        return torch.where(nonzero, unit_power, 1.)

    def _factors(self, config, reference):
        key = (tuple(tuple(row) for row in config["rx_correlation"])
               if isinstance(config["rx_correlation"], list) else config["rx_correlation"],
               tuple(tuple(row) for row in config["tx_correlation"])
               if isinstance(config["tx_correlation"], list) else config["tx_correlation"],
               reference.device, reference.dtype)
        if key not in self._fading_factors:
            self._fading_factors[key] = tuple(correlation_factor(spec, self.width).to(reference)
                                              for spec in key[:2])
        return self._fading_factors[key]

    def draw(self, count: int, config: Mapping, generator: torch.Generator):
        config = validate_channel(config)
        if config["kind"] == "diagonal_fading":
            rho = config["rho_c"]
            key = ("diagonal", rho, generator.device, torch.get_default_dtype())
            if key not in self._fading_factors:
                self._fading_factors[key] = (correlation_factor(rho, self.width).to(
                    device=generator.device, dtype=torch.get_default_dtype()) if rho else None)
            return draw_diagonal_fading(count, self.width, self._fading_factors[key], generator)
        if config["kind"] == "full_fading":
            reference = torch.empty(0, device=generator.device)
            rx, tx = self._factors(config, reference)
            return draw_full_fading(count, rx, tx, config["noise_std"], generator)
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
            indices = torch.randint(self.m, (len(hidden),), generator=generator,
                                    device=generator.device).to(hidden.device)
            gain, noise = self.gain[indices], self.noise[indices]
        elif source == "deployment":
            deployment = validate_channel(config or self.config)
            if deployment["kind"] == "full_fading":
                rx, tx = self._factors(deployment, hidden)
                return sample_received(hidden, rx, tx, deployment["noise_std"], generator)
            gain, noise = self.draw(len(hidden), deployment, generator)
            gain, noise = gain.to(hidden.device), noise.to(hidden.device)
        else:
            raise ValueError(f"Unknown source: {source}")
        if gain.ndim == 3:
            return torch.bmm(gain.to(hidden.dtype), hidden.unsqueeze(-1)).squeeze(-1) + noise
        return hidden * gain + noise
