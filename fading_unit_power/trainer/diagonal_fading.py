"""Spatially correlated diagonal gains with unit marginal variance and independent AWGN."""
import math

import torch


DEPLOYMENT_RHO = .35


def normalize_diagonal_fading(config):
    unknown = set(config) - {"kind", "rho_c"}
    if unknown:
        raise ValueError(f"Unknown diagonal fading fields: {sorted(unknown)}")
    rho = config.get("rho_c", DEPLOYMENT_RHO)
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 <= rho < 1:
        raise ValueError("rho_c must lie in [0, 1)")
    return {"kind": "diagonal_fading", "rho_c": float(rho)}


def draw_diagonal_fading(count, width, factor, generator):
    """Each (gain, noise) row is one channel variable. A null factor denotes iid gains."""
    shape = (count, width)
    innovations = torch.randn(shape, device=generator.device, generator=generator)
    centered = innovations if factor is None else innovations @ factor.T
    noise = torch.randn(shape, device=generator.device, generator=generator)
    return centered.add_(.5), noise


def diagonal_gain_kl(deployment, training, width):
    """KL(deployment || training) via the conditional normals of a stationary AR(1)."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Channel width must be a positive integer")
    p = normalize_diagonal_fading(deployment)["rho_c"]
    q = normalize_diagonal_fading(training)["rho_c"]
    # Initial coordinates and additive noise have identical laws.
    p_var, q_var = (1 - p) * (1 + p), (1 - q) * (1 + q)
    variance_term = (p_var - q_var) / q_var - (math.log(p_var) - math.log(q_var))
    return .5 * (width - 1) * max(0., variance_term + (p - q)**2 / q_var)
