"""Additional split-CNN candidates for explicit CIFAR architecture searches.

These variants transmit ``width`` features and retain a convolutional receiver.
They share one flat diagonal-Gaussian posterior over all weights and biases,
with the same sampling/channel interface as the existing CIFAR models. No
normalization, dropout or batch-dependent state is introduced.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .cifar_models import GaussianMiddleSplitCNN
from .theorem1 import ChannelBank


class _GaussianSearchCNN(nn.Module):
    """Shared implementation; subclasses specify the bottleneck and receiver."""

    receiver_hidden_width = 128

    def __init__(self, width: int = 128, classes: int = 10,
                 stochastic: bool = True, posterior_std: float = .01):
        super().__init__()
        if not isinstance(width, int) or width <= 0 or width % 16:
            raise ValueError("width must be a positive integer multiple of 16")
        if not isinstance(classes, int) or classes <= 1:
            raise ValueError("classes must be an integer greater than one")
        if not math.isfinite(posterior_std) or posterior_std <= 0:
            raise ValueError("posterior_std must be finite and positive")
        self.input_dim, self.width, self.classes = 3 * 32 * 32, width, classes
        self.stochastic = stochastic
        self.transmitted_shape = (width // 16, 4, 4)
        latent_channels = self.transmitted_shape[0]
        bottleneck_shapes = (((width, 64 * 4 * 4), (width,))
                             if self.bottleneck_kind == "affine_relu" else
                             ((latent_channels, 64, 1, 1), (latent_channels,)))
        self.parameter_shapes = (
            (16, 3, 3, 3), (16,),
            (32, 16, 3, 3), (32,),
            (64, 32, 3, 3), (64,),
            *bottleneck_shapes,
            (self.receiver_channels, latent_channels, 3, 3), (self.receiver_channels,),
            (self.receiver_hidden_width, self.receiver_channels * 4 * 4), (self.receiver_hidden_width,),
            (classes, self.receiver_hidden_width), (classes,),
        )
        offsets, cursor = [], 0
        for shape in self.parameter_shapes:
            count = math.prod(shape)
            offsets.append((cursor, count))
            cursor += count
        self._offsets = tuple(offsets)
        self.d = cursor
        self.transmitter_parameter_count = sum(math.prod(shape) for shape in self.parameter_shapes[:8])
        self.receiver_parameter_count = self.d - self.transmitter_parameter_count
        bottleneck_description = (
            f"flatten1024; Affine1024->{width}/ReLU; learned vector packing"
            if self.bottleneck_kind == "affine_relu" else
            f"Conv64->{latent_channels}(1x1)/ReLU; spatial projection")
        self.split_description = (
            "Tx: Conv3->16(3x3,pad1)/ReLU/MaxPool2; "
            "Conv16->32(3x3,pad1)/ReLU/MaxPool2; "
            "Conv32->64(3x3,pad1)/ReLU; AdaptiveAvgPool4x4; "
            f"{bottleneck_description}; {width} channel symbols. "
            f"Rx: reshape {self.transmitted_shape}; "
            f"Conv{latent_channels}->{self.receiver_channels}(3x3,pad1)/ReLU; "
            f"flatten{self.receiver_channels * 16}; "
            f"Affine{self.receiver_channels * 16}->{self.receiver_hidden_width}/ReLU; "
            f"Affine{self.receiver_hidden_width}->{classes}."
        )
        initial_rho = posterior_std + math.log(-math.expm1(-posterior_std))
        self.mu = nn.Parameter(torch.empty(self.d))
        self.rho = nn.Parameter(torch.full((self.d,), initial_rho), requires_grad=stochastic)
        with torch.no_grad():
            parts = self.split(self.mu)
            for weight, bias in zip(parts[::2], parts[1::2]):
                bound = 1 / math.sqrt(math.prod(weight.shape[1:]))
                weight.uniform_(-bound, bound)
                bias.zero_()

    def split(self, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return transmitter then receiver parameters without copying storage."""
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
            raise ValueError("CIFAR search CNN input must have shape [batch,3,32,32]")
        if bank.width != self.width:
            raise ValueError("Channel-bank width must equal the CNN feature width")
        w1, b1, w2, b2, w3, b3, wl, bl, wr, br, wf, bf, wo, bo = self.split(theta)
        hidden = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
        hidden = F.max_pool2d(F.relu(F.conv2d(hidden, w2, b2, padding=1)), 2)
        hidden = F.adaptive_avg_pool2d(F.relu(F.conv2d(hidden, w3, b3, padding=1)), (4, 4))
        if self.bottleneck_kind == "affine_relu":
            transmitted = F.relu(F.linear(hidden.flatten(1), wl, bl))
        else:
            transmitted = F.relu(F.conv2d(hidden, wl, bl)).flatten(1)
        if not self.training:
            self.last_feature_power = transmitted.detach().square().mean().item()
        recovered = bank.apply(transmitted, source, channel, generator)
        recovered = recovered.reshape(x.shape[0], *self.transmitted_shape)
        hidden = F.relu(F.conv2d(recovered, wr, br, padding=1)).flatten(1)
        hidden = F.relu(F.linear(hidden, wf, bf))
        return F.linear(hidden, wo, bo)


class GaussianReceiverConvCNN(_GaussianSearchCNN):
    """Original strong affine-bottleneck encoder with a convolutional decoder.

    At width 128/classes 10, transmitter/receiver counts are 154,784/69,290,
    totaling 224,074 scalar weights and biases in one realization. The learned
    vector packing does not assume spatial correspondence with input pixels.
    """

    architecture = "cnn_rxconv"
    bottleneck_kind = "affine_relu"
    receiver_channels = 32


class GaussianSpatialThreeConvCNN(_GaussianSearchCNN):
    """Three transmitter convolutions and a spatial 1x1/ReLU bottleneck.

    At width 128/classes 10, transmitter/receiver counts are 24,104/137,162,
    totaling 161,266 scalar weights and biases in one realization.
    """

    architecture = "cnn_spatial3"
    bottleneck_kind = "spatial1x1_relu"
    receiver_channels = 64


class GaussianMiddleSplitReLUCNN(GaussianMiddleSplitCNN):
    """Controlled middle-split variant with a nonnegative transmitted latent.

    Parameter shapes, initialization, sampling and receiver match the original
    middle-split model. Only ReLU after the 1x1 projection is added. Width 128
    still has 5,352 transmitter and 137,162 receiver weights/biases (142,514).
    """

    architecture = "cnn_mid_relu"
    bottleneck_kind = "spatial1x1_relu"
    nonnegative_transmitted_features = True

    def __init__(self, width: int = 128, classes: int = 10,
                 stochastic: bool = True, posterior_std: float = .01):
        super().__init__(width, classes, stochastic, posterior_std)
        self.split_description = (
            "Tx: Conv3->16(3x3,pad1)/ReLU/MaxPool2; "
            "Conv16->32(3x3,pad1)/ReLU/MaxPool2; AdaptiveAvgPool4x4; "
            f"Conv32->{self.transmitted_shape[0]}(1x1)/ReLU; "
            f"flatten {self.transmitted_shape} to {width} nonnegative channel symbols. "
            f"Rx: reshape {self.transmitted_shape}; "
            f"Conv{self.transmitted_shape[0]}->64(3x3,pad1)/ReLU; flatten1024; "
            f"Affine1024->{width}/ReLU; Affine{width}->{classes}."
        )

    def forward(self, x: torch.Tensor, theta: torch.Tensor,
                bank: ChannelBank, source: str = "bank", channel: Mapping | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError("CIFAR middle CNN input must have shape [batch,3,32,32]")
        if bank.width != self.width:
            raise ValueError("Channel-bank width must equal the CNN feature width")
        w1, b1, w2, b2, wl, bl, wr, br, wf, bf, wo, bo = self.split(theta)
        hidden = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
        hidden = F.max_pool2d(F.relu(F.conv2d(hidden, w2, b2, padding=1)), 2)
        hidden = F.adaptive_avg_pool2d(hidden, (4, 4))
        transmitted = F.relu(F.conv2d(hidden, wl, bl)).flatten(1)
        if not self.training:
            self.last_feature_power = transmitted.detach().square().mean().item()
        recovered = bank.apply(transmitted, source, channel, generator)
        recovered = recovered.reshape(x.shape[0], *self.transmitted_shape)
        hidden = F.relu(F.conv2d(recovered, wr, br, padding=1)).flatten(1)
        hidden = F.relu(F.linear(hidden, wf, bf))
        return F.linear(hidden, wo, bo)


CIFAR_SEARCH_MODELS = {
    GaussianReceiverConvCNN.architecture: GaussianReceiverConvCNN,
    GaussianSpatialThreeConvCNN.architecture: GaussianSpatialThreeConvCNN,
    GaussianMiddleSplitReLUCNN.architecture: GaussianMiddleSplitReLUCNN,
}
