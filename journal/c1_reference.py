"""Analytic C1 channel terms for frozen, perfect-trained Bayesian networks.

The transmitter sends its two output logits; the receiver applies softmax.
For ell=-log(epsilon+(1-epsilon)*p_y)/log(1/epsilon), differentiation gives

    grad ell = a(p_y) * (p-e_y) / log(1/epsilon),  0 <= a(p_y) <= 1.

Since ||p-e_y||_2 <= sqrt(2), K=sqrt(2)/log(1/epsilon) is uniform over
inputs, labels, logits and network weights, including Gaussian posteriors.
This proves a logit-channel constant, not a hidden-feature-channel constant.

Independent centered, standardized lognormal coordinates have no simple
known closed form for their two-dimensional mean Euclidean norm. C1's
second inequality instead uses a fixed moment-matched Gaussian reference,
whose exact KL and Rayleigh exponential moment are analytic. Numerical
distance diagnostics below never enter that bound. The deployment law's
own positive exponential moment is infinite and is not used here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr, logsumexp


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return float(value)


def _shape_square(shape: float) -> float:
    shape = _positive(shape, "shape")
    square = shape * shape
    if not math.isfinite(square) or square == 0:
        raise ValueError("shape squared must be positive and finite")
    return square


def _log_expm1(value: float) -> float:
    return (math.log(math.expm1(value)) if value < math.log(2.) else
            value + math.log1p(-math.exp(-value)))


def _lognormal_shape_kl(shape: float) -> float:
    """Two-coordinate KL to the Gaussian with the same mean and variance."""
    square = _shape_square(shape)
    if square < 1e-3:
        # log(expm1(x)/x); the series preserves the Gaussian limit at tiny x.
        ratio_log = square / 2 + square**2 / 24 - square**4 / 2880
    else:
        ratio_log = _log_expm1(square) - math.log(square)
    result = square + ratio_log
    if not math.isfinite(result):
        raise ValueError("Lognormal shape KL is outside the finite numeric range")
    return result


def smooth_loss_lipschitz(epsilon: float = .01) -> float:
    """Uniform Euclidean logit Lipschitz constant for the smooth bounded loss."""
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and in (0,1)")
    return math.sqrt(2.) / -math.log(epsilon)


def log_rayleigh_mgf(t: float) -> float:
    """log E exp(t R), R~Rayleigh(1), evaluated without exponent overflow.

    M(t)=1+t*sqrt(2*pi)*exp(t**2/2)*Phi(t), for t>=0.
    """
    t = _nonnegative(t, "t")
    if t == 0:
        return 0.
    log_second = math.log(t) + .5 * math.log(2 * math.pi) + .5 * t * t + float(log_ndtr(t))
    result = float(logsumexp([0., log_second]))
    if not math.isfinite(result):
        raise ValueError("Rayleigh log moment is outside the finite numeric range")
    return result


def erasure_first_shift(width: int, probability: float) -> float:
    """Exact C1 first-distance term under the discrete complete-mask metric."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    probability = _nonnegative(probability, "probability")
    if probability > 1:
        raise ValueError("probability must lie in [0,1]")
    return 1. if probability == 1 else -math.expm1(width * math.log1p(-probability))


def centered_lognormal_kl(sigma: float, shape: float = .1,
                          reference_std: float | None = None) -> float:
    """Exact KL of two independent centered-lognormal coordinates to N(0,s²I).

    N_i=sigma*(exp(shape*Z_i-shape²/2)-1)/sqrt(expm1(shape²)), Z_i~N(0,1).
    The shifted density is absolutely continuous with respect to the Gaussian.
    Entropy and E[N_i²]=sigma² give J+2*log(s/sigma)+sigma²/s²-1.
    This helper requires positive noise; the zero-noise Dirac case is handled
    separately by lognormal_logit_channel_terms.
    """
    sigma = _positive(sigma, "sigma")
    s = sigma if reference_std is None else _positive(reference_std, "reference_std")
    shape_kl = _lognormal_shape_kl(shape)
    log_ratio = math.log(sigma) - math.log(s)
    try:
        mismatch = math.expm1(2 * log_ratio) - 2 * log_ratio
    except OverflowError as error:
        raise ValueError("Gaussian reference variance mismatch is outside the numeric range") from error
    result = shape_kl + mismatch
    if not math.isfinite(result):
        raise ValueError("KL is outside the finite numeric range")
    return result


def lognormal_logit_channel_terms(sigma: float, shape: float = .1,
                                 epsilon: float = .01) -> dict:
    """C1's analytic second-distance upper using fixed Q=N(0,sigma² I_2).

    No direct-distance estimate is accepted or used. The Jensen comparison
    K*sigma*sqrt(2) is also analytic; neither bound uniformly dominates.
    At sigma=0 the actual and reference laws are the same Dirac mass and the
    shift is exactly zero. This is a separate endpoint, not the positive-noise
    Gaussian/lognormal KL formula's limit, which retains a shape penalty.
    """
    sigma = _nonnegative(sigma, "sigma")
    _shape_square(shape)
    k = smooth_loss_lipschitz(epsilon)
    divergence = centered_lognormal_kl(sigma, shape) if sigma else 0.
    log_moment = log_rayleigh_mgf(k * sigma)
    return dict(sigma=sigma, shape=float(shape), lipschitz=k, reference_std=sigma,
                log_exponential_moment=log_moment, channel_kl=divergence,
                c1_second_shift_upper=log_moment + divergence,
                jensen_shift_upper=k * sigma * math.sqrt(2.),
                kl_to_perfect_training="infinite" if sigma else "zero")


def sample_centered_lognormal(rng: np.random.Generator, size,
                              sigma: float = 1., shape: float = .1) -> np.ndarray:
    """Sample the fully specified channel; size includes the coordinate axis."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    dimensions = (size,) if isinstance(size, int) else size
    if (not isinstance(dimensions, tuple) or not dimensions or
            any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in dimensions)):
        raise ValueError("size must be a positive integer or tuple of positive integers")
    sigma = _nonnegative(sigma, "sigma")
    square = _shape_square(shape)
    if sigma == 0:
        return np.zeros(dimensions, dtype=np.float64)
    z = rng.standard_normal(dimensions)
    log_scale = math.log(sigma) - .5 * _log_expm1(square)
    # exp(log_scale)*expm1(...) is accurate around zero for the small shape
    # used in the experiment. Explicit errors reject numerically invalid draws.
    try:
        with np.errstate(over="raise", invalid="raise"):
            noise = math.exp(log_scale) * np.expm1(shape * z - square / 2)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Noise samples are outside the finite numeric range") from error
    return noise


def lognormal_distance_benchmark(shape: float = .1, samples: int = 2_000_000,
                                seed: int = 879513, batch_size: int = 100_000) -> dict:
    """Independent MC diagnostic of E||N|| for sigma=1, never a certificate.

    The standard error is descriptive, not a rigorous concentration radius
    for the unbounded lognormal distance. Scaling this mean by K*sigma is
    solely a comparison with the analytic second-inequality bound.
    """
    _shape_square(shape)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < minimum
           for x, minimum in ((samples, 2), (batch_size, 1))):
        raise ValueError("samples must be an integer >=2 and batch_size a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    count, mean, sum_square_deviations = 0, 0., 0.
    while count < samples:
        size = min(batch_size, samples - count)
        norms = np.linalg.norm(sample_centered_lognormal(rng, (size, 2), shape=shape), axis=1)
        batch_mean = float(norms.mean())
        batch_m2 = float(np.square(norms - batch_mean).sum())
        total = count + size
        difference = batch_mean - mean
        sum_square_deviations += batch_m2 + difference**2 * count * size / total
        mean += difference * size / total
        count = total
    return dict(unit_scale_mean_norm=mean,
                unit_scale_mean_norm_standard_error=math.sqrt(sum_square_deviations / (samples - 1) / samples),
                samples=samples, seed=seed, shape=float(shape),
                status="Independent diagnostic MC only; not a certified first-inequality bound")
