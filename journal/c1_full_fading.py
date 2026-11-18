"""Analytic Corollary 1 shifts for the correlated Gaussian channel."""
from __future__ import annotations

import math

import torch

from .c1_reference import erasure_first_shift
from .full_fading import correlation_matrix


def log_gaussian_norm_moment_upper(t: float, mean: torch.Tensor,
                                   variances: torch.Tensor) -> float:
    """Bound log E exp(t||X||₂) for independent Gaussian coordinates.

    Young's inequality gives t||X|| ≤ t²/(2a) + a||X||²/2. Integrating
    the resulting quadratic exponential against a Gaussian gives the
    expression below for any 0 < a < 1 / max(variances).
    """
    if t == 0:
        return 0.
    rms = (mean.square() + variances).sum().sqrt().item()
    if rms == 0:
        return 0.
    largest_variance = variances.max().item()
    a = t / rms
    if largest_variance:
        a = min(a, .5 / largest_variance)
    scaled_variance = a * variances
    log_quadratic_moment = .5 * (
        -torch.log1p(-scaled_variance) + a * mean.square() / (1 - scaled_variance)
    ).sum().item()
    return t * t / (2 * a) + log_quadratic_moment


def gaussian_second_shift(deployment: dict, width: int, feature_norm: float,
                          loss_lipschitz: float) -> dict:
    """Use Q_C=P_ch and an analytic upper bound on its Gaussian log-CGF.

    The metric is R||H-H'||_F + ||N-N'||₂, where ||f(x)||≤R. The
    orthonormal receive/transmit eigenbases preserve the Frobenius norm
    and diagonalize the covariance of H. The noise is independent of H.
    """
    rx_values, rx_vectors = torch.linalg.eigh(
        correlation_matrix(deployment["rx_correlation"], width))
    tx_values, tx_vectors = torch.linalg.eigh(
        correlation_matrix(deployment["tx_correlation"], width))
    mean_delta = .5 * torch.ones((width, width), dtype=torch.float64) - torch.eye(
        width, dtype=torch.float64)
    h_moment = log_gaussian_norm_moment_upper(
        loss_lipschitz * feature_norm, rx_vectors.T @ mean_delta @ tx_vectors,
        rx_values[:, None] * tx_values[None, :])
    n_moment = log_gaussian_norm_moment_upper(
        loss_lipschitz, torch.zeros(width, dtype=torch.float64),
        torch.full((width,), deployment["noise_std"]**2, dtype=torch.float64))
    return dict(method="C1 second", metric="R * Frobenius gain distance + Euclidean noise distance",
                analytic=True, upper=h_moment + n_moment, reference="Q_C = P_ch",
                reference_kl=0., gain_log_mgf_upper=h_moment, noise_log_mgf_upper=n_moment,
                uniform_lipschitz=loss_lipschitz, feature_norm_limit=feature_norm)


def perfect_shift(deployment: dict, width: int, model, loss_epsilon: float) -> dict:
    feature_norm = getattr(model, "feature_norm", None)
    if deployment["kind"] == "outage":
        probability = deployment["outage_prob"]
        if feature_norm is None:
            distance = erasure_first_shift(width, probability)
            return dict(method="C1 first", metric="discrete complete-mask distance",
                        analytic=True, expectation=distance, upper=distance, uniform_lipschitz=1.)
        distance = math.fsum(math.sqrt(k) * math.comb(width, k) * probability**k
                             * (1 - probability)**(width - k) for k in range(1, width + 1))
        lipschitz = feature_norm * model.channel_lipschitz_constant(loss_epsilon)
        return dict(method="C1 first", metric="Frobenius mask distance",
                    analytic=True, expectation=distance, upper=lipschitz * distance,
                    uniform_lipschitz=lipschitz)
    if feature_norm is None:
        # Raw Gaussian weights in the existing models do not imply a finite
        # uniform Lipschitz constant in the Frobenius gain metric. Bounded loss
        # instead gives K=1 for the discrete metric on (H,N), and Q_C=P_ch.
        # A nondegenerate Gaussian H equals the identity with probability zero.
        return dict(method="C1 second", metric="discrete complete-channel distance",
                    analytic=True, upper=1., uniform_lipschitz=1.,
                    reference="Q_C = P_ch", reference_kl=0., log_mgf=1.)
    return gaussian_second_shift(deployment, width, feature_norm,
                                 model.channel_lipschitz_constant(loss_epsilon))
