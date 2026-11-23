"""Compact Gaussian two-layer predictor with uniform channel sensitivity."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .theorem1 import GaussianMLP


class BoundedGaussianMLP(GaussianMLP):
    """Project transmitted activations and the sampled receiver weights.

    The diagonal Gaussian prior and posterior are over the raw weights. Both
    projections are fixed parts of the predictor and hold for every raw draw
    and input, including the unbounded two-moons covariates.
    """

    def __init__(self, input_dim, width, classes, stochastic, posterior_std,
                 feature_norm, receiver_norm):
        super().__init__(input_dim, width, classes, stochastic, posterior_std)
        self.feature_norm = feature_norm
        self.receiver_norm = receiver_norm

    def channel_lipschitz_constant(self, loss_epsilon: float) -> float:
        """Smooth bounded loss constant for R||ΔH||_F + ||ΔN||_2."""
        return math.sqrt(2.) * self.receiver_norm / -math.log(loss_epsilon)

    def transmit_feature_radius(self, bank) -> float:
        return math.sqrt(self.width) if bank.transmit_unit_power else self.feature_norm

    def project_features(self, hidden):
        return hidden * (self.feature_norm / hidden.norm(dim=-1, keepdim=True)
                         .clamp_min(self.feature_norm))

    def project_receiver(self, weight, bias):
        norm = weight.norm(dim=(-2, -1), keepdim=True)
        scale = self.receiver_norm / norm.clamp_min(self.receiver_norm)
        return weight * scale, bias * scale.squeeze(-1)

    def forward(self, x, theta, bank, source="bank", channel=None, generator=None):
        w1, b1, w2, b2 = self.split(theta)
        hidden = bank.transmit(self.project_features(F.relu(x.flatten(1) @ w1 + b1)))
        if not self.training:
            self.last_feature_power = hidden.detach().square().mean().item()
        w2, b2 = self.project_receiver(w2, b2)
        return bank.apply(hidden, source, channel, generator) @ w2 + b2

    def forward_iid_weights(self, x, bank, weight_generator, channel_generator):
        raw = self.mu + F.softplus(self.rho) * torch.randn(
            (len(x), self.d), device=self.mu.device, dtype=self.mu.dtype,
            generator=weight_generator)
        a, b, c = self.input_dim * self.width, self.d1, self.d1 + self.width * self.classes
        w1 = raw[:, :a].reshape(-1, self.input_dim, self.width)
        w2 = raw[:, b:c].reshape(-1, self.width, self.classes)
        hidden = torch.bmm(x.flatten(1).unsqueeze(1), w1).squeeze(1) + raw[:, a:b]
        hidden = bank.transmit(self.project_features(F.relu(hidden)))
        received = bank.apply(hidden, "bank", generator=channel_generator)
        w2, b2 = self.project_receiver(w2, raw[:, c:])
        return torch.bmm(received.unsqueeze(1), w2).squeeze(1) + b2
