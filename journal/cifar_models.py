"""Compact split CNN with one flat diagonal-Gaussian weight posterior.

The architecture is shared by deterministic ERM/L2 and stochastic T1/T2.
The transmitter is Conv(3,16)-ReLU-Pool, Conv(16,32)-ReLU-Pool,
Conv(32,64)-ReLU, adaptive 4x4 pooling, then a width-dimensional affine
projection and ReLU. Transmission acts on that complete feature vector;
the receiver is a single affine ten-class classifier. No normalization,
dropout or batch-dependent state changes predictor semantics between modes.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .theorem1 import ChannelBank


class GaussianSplitCNN(nn.Module):
    """CIFAR-sized split network compatible with :class:`GaussianMLP`.

    ``mu`` and ``rho`` parameterize a single posterior over *all* convolutional
    and affine weights and biases. ``split`` returns storage-sharing views, so
    drawing a posterior does not construct or copy per-layer modules. The KL
    can be calculated directly by the existing ``gaussian_kl(mu,rho,prior)``.
    The default width 128 has 156,074 weights; width 64 has 89,834.
    """

    def __init__(self, width: int = 128, classes: int = 10,
                 stochastic: bool = True, posterior_std: float = .01):
        super().__init__()
        if not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(classes, int) or classes <= 1:
            raise ValueError("classes must be an integer greater than one")
        if not math.isfinite(posterior_std) or posterior_std <= 0:
            raise ValueError("posterior_std must be finite and positive")
        self.input_dim, self.width, self.classes = 3 * 32 * 32, width, classes
        self.stochastic = stochastic
        self.parameter_shapes = (
            (16, 3, 3, 3), (16,),
            (32, 16, 3, 3), (32,),
            (64, 32, 3, 3), (64,),
            (width, 64 * 4 * 4), (width,),
            (classes, width), (classes,),
        )
        offsets = []
        cursor = 0
        for shape in self.parameter_shapes:
            count = math.prod(shape)
            offsets.append((cursor, count))
            cursor += count
        self._offsets = tuple(offsets)
        self.d = cursor
        # Stable inverse of softplus for all positive scales.
        initial_rho = posterior_std + math.log(-math.expm1(-posterior_std))
        self.mu = nn.Parameter(torch.empty(self.d))
        self.rho = nn.Parameter(torch.full((self.d,), initial_rho),
                                requires_grad=stochastic)
        with torch.no_grad():
            tensors = self.split(self.mu)
            for index in range(0, len(tensors), 2):
                weight, bias = tensors[index:index + 2]
                fan_in = math.prod(weight.shape[1:])
                bound = 1 / math.sqrt(fan_in)
                weight.uniform_(-bound, bound)
                bias.zero_()

    def split(self, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return layer parameters as views of a complete weight realization."""
        if theta.ndim != 1 or theta.numel() != self.d:
            raise ValueError(f"theta must have shape ({self.d},)")
        return tuple(theta.narrow(0, start, count).view(shape)
                     for (start, count), shape in zip(self._offsets, self.parameter_shapes))

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if not self.stochastic:
            return self.mu
        noise = torch.randn(self.mu.shape, device=self.mu.device,
                            dtype=self.mu.dtype, generator=generator)
        return self.mu + F.softplus(self.rho) * noise

    def forward(self, x: torch.Tensor, theta: torch.Tensor,
                bank: ChannelBank, source: str = "bank", channel: Mapping | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError("CIFAR CNN input must have shape [batch,3,32,32]")
        if bank.width != self.width:
            raise ValueError("Channel-bank width must equal the CNN feature width")
        w1, b1, w2, b2, w3, b3, wf, bf, wo, bo = self.split(theta)
        hidden = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
        hidden = F.max_pool2d(F.relu(F.conv2d(hidden, w2, b2, padding=1)), 2)
        hidden = F.relu(F.conv2d(hidden, w3, b3, padding=1))
        hidden = F.adaptive_avg_pool2d(hidden, (4, 4)).flatten(1)
        hidden = F.relu(F.linear(hidden, wf, bf))
        if not self.training:
            self.last_feature_power = hidden.detach().square().mean().item()
        recovered = bank.apply(hidden, source, channel, generator)
        return F.linear(recovered, wo, bo)
