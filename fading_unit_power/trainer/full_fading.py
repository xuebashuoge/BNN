"""Matrix-normal fading with fixed mean 0.5 and unit marginal variance."""
from __future__ import annotations

from functools import lru_cache
import math

import torch


GAIN_MEAN = .5
DEFAULT_CORRELATION = .25
DEFAULT_NOISE_STD = 1.


def correlation_key(spec):
    """Normalize an AR(1) coefficient or an explicit correlation matrix."""
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        if not math.isfinite(spec) or not -1 < spec < 1:
            raise ValueError("AR(1) correlation must be finite and in (-1, 1)")
        return float(spec)
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError("Expected an AR(1) coefficient or a square correlation matrix")
    if any(not isinstance(row, (tuple, list)) or len(row) != len(spec) for row in spec):
        raise ValueError("Correlation matrix must be square")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for row in spec for value in row):
        raise ValueError("Correlation matrix entries must be numeric")
    key = tuple(tuple(float(value) for value in row) for row in spec)
    _correlation_and_factor(key, len(key))
    return key


@lru_cache(maxsize=32)
def _correlation_and_factor(spec, width):
    if isinstance(spec, float):
        indices = torch.arange(width)
        matrix = spec ** (indices[:, None] - indices).abs().to(torch.float64)
    else:
        if len(spec) != width:
            raise ValueError(f"Correlation matrix size must match channel width {width}")
        matrix = torch.tensor(spec, dtype=torch.float64)
    if (not torch.isfinite(matrix).all() or not torch.equal(matrix, matrix.T)
            or not torch.all(matrix.diag() == 1.)):
        raise ValueError("Correlation matrix must be finite, symmetric, and have unit diagonal")
    factor, info = torch.linalg.cholesky_ex(matrix)
    if info.item():
        raise ValueError("Correlation matrix must be positive definite")
    return matrix, factor


def correlation_matrix(spec, width: int) -> torch.Tensor:
    """Materialize a unit-diagonal SPD correlation matrix in float64."""
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("Correlation width must be a positive integer")
    matrix, _ = _correlation_and_factor(correlation_key(spec), width)
    return matrix.clone()


def correlation_factor(spec, width: int) -> torch.Tensor:
    """Return a lower factor L with L Lᵀ equal to the correlation matrix."""
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("Correlation width must be a positive integer")
    return _correlation_and_factor(correlation_key(spec), width)[1].clone()


def normalize_full_fading(config):
    unknown = set(config) - {"kind", "rx_correlation", "tx_correlation", "noise_std"}
    if unknown:
        raise ValueError(f"Unknown full fading fields: {sorted(unknown)}")
    noise_std = config.get("noise_std", DEFAULT_NOISE_STD)
    if (not isinstance(noise_std, (int, float)) or isinstance(noise_std, bool)
            or not math.isfinite(noise_std) or noise_std < 0):
        raise ValueError("Full fading noise_std must be finite and nonnegative")
    result = {"kind": "full_fading", "noise_std": float(noise_std)}
    for name in ("rx_correlation", "tx_correlation"):
        key = correlation_key(config.get(name, DEFAULT_CORRELATION))
        result[name] = [list(row) for row in key] if isinstance(key, tuple) else key
    return result


def validate_fading_dimensions(config, receive_width: int, transmit_width: int):
    """Check a channel config against the feature dimensions."""
    correlation_factor(config["rx_correlation"], receive_width)
    correlation_factor(config["tx_correlation"], transmit_width)


def draw_full_fading(count: int, rx_factor: torch.Tensor, tx_factor: torch.Tensor,
                     noise_std: float, generator: torch.Generator):
    """Sample actual H and independent N, allowing rectangular gains.

    L_r G L_tᵀ has exactly the law of R_r^{1/2} G R_t^{1/2} since G is
    invariant under left and right orthogonal multiplication.
    """
    receive_width, transmit_width = rx_factor.shape[0], tx_factor.shape[0]
    shape = (count, receive_width, transmit_width)
    gain = torch.randn(shape, generator=generator, device=generator.device, dtype=rx_factor.dtype)
    # Transform the bank in place in blocks to bound intermediate allocations.
    chunk_size = max(1, 4_000_000 // (receive_width * transmit_width))
    for chunk in gain.split(chunk_size):
        left = torch.bmm(rx_factor.expand(len(chunk), -1, -1), chunk)
        torch.bmm(left, tx_factor.T.expand(len(chunk), -1, -1), out=chunk)
    gain.add_(GAIN_MEAN)
    noise_shape = (count, receive_width)
    if noise_std:
        noise = torch.randn(noise_shape, generator=generator, device=gain.device, dtype=gain.dtype)
        noise.mul_(noise_std)
    else:
        noise = gain.new_zeros(noise_shape)
    return gain, noise


def sample_received(hidden: torch.Tensor, rx_factor: torch.Tensor, tx_factor: torch.Tensor,
                    noise_std: float, generator: torch.Generator) -> torch.Tensor:
    """Exactly sample the per-example law of Hf+N for fresh independent H.

    The projected white Gaussian is reused across matched evaluation seeds.
    This is a common-random-number coupling of received signals, rather than
    the same realized H applied to distinct transmitters.
    """
    shape = (len(hidden), rx_factor.shape[0])
    white = torch.randn(shape, generator=generator, device=generator.device, dtype=hidden.dtype).to(hidden.device)
    scale = (hidden @ tx_factor).norm(dim=-1, keepdim=True)
    received = GAIN_MEAN * hidden.sum(dim=-1, keepdim=True) + scale * (white @ rx_factor.T)
    if noise_std:
        noise = torch.randn(shape, generator=generator, device=generator.device, dtype=hidden.dtype).to(hidden.device)
        received = received + noise_std * noise
    return received


def fading_gain_kl(deployment, training, receive_width: int, transmit_width: int) -> float:
    """Exact KL between gain laws using Kronecker traces and log determinants."""
    def factor_terms(name, width):
        matrix_p, chol_p = _correlation_and_factor(correlation_key(deployment[name]), width)
        _, chol_q = _correlation_and_factor(correlation_key(training[name]), width)
        trace = torch.cholesky_solve(matrix_p, chol_q).trace().item()
        logdet_ratio = 2 * (chol_q.diag().log().sum() - chol_p.diag().log().sum()).item()
        return trace, logdet_ratio

    rx_trace, rx_logdet_ratio = factor_terms("rx_correlation", receive_width)
    tx_trace, tx_logdet_ratio = factor_terms("tx_correlation", transmit_width)
    dimension = receive_width * transmit_width
    return max(0., .5 * math.fsum((rx_trace * tx_trace, -dimension,
                                  transmit_width * rx_logdet_ratio, receive_width * tx_logdet_ratio)))
