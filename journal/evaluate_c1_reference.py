"""Evaluate two C1 expressions on the same frozen perfect-trained posteriors.

Erasure uses an exact expected distance. Fully specified centered-lognormal
output noise has a non-elementary mean Euclidean distance; the independent
MC distance benchmark is diagnostic only. Its bound uses fixed Gaussian
Q=N(0,sigma**2 I), whose radial chi exponential moment and entropy-based
KL(P||Q) are closed form. No distance MC or reference optimization enters
that bound. Network weights and perfect-channel training remain unchanged.

The smooth bounded loss has a uniform output-logit Lipschitz constant
sqrt(2)/log(1/epsilon), independent of weights and inputs. This is an output
transmission example, not a Lipschitz claim for hidden-feature Gaussian noise.
Gaussian helpers remain as independently tested analytical diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ellipe, gammaln, hyp1f1
import torch
from torch.nn import functional as F

from .certify_checkpoint import bank_hash, verify_sampler_source
from .run_experiment import bounded_losses_from_ce, load_data, make_models, normalize_config
from .theorem1 import gaussian_kl, theorem1_terms


def smooth_loss_lipschitz(epsilon: float) -> float:
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Require a finite loss epsilon in (0,1)")
    return math.sqrt(2.) / math.log(1. / epsilon)


def log_chi_mgf(t: float, dimension: int) -> float:
    """Exact log E exp(t R), R chi_dimension, for t>=0.

    Splitting the exponential series into even and odd powers gives these
    two confluent hypergeometric terms. All terms are positive for t>=0.
    """
    if not math.isfinite(t) or t < 0 or isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Require finite t>=0 and positive integer dimension")
    z = t * t / 2
    even = hyp1f1(dimension / 2, .5, z)
    odd = (math.sqrt(2.) * t * math.exp(gammaln((dimension+1)/2) - gammaln(dimension/2)) *
           hyp1f1((dimension+1)/2, 1.5, z))
    moment = float(even + odd)
    if not math.isfinite(moment):
        raise ValueError("Chi MGF overflow; reduce t or use a log-domain quadrature")
    return math.log(moment)


def gaussian_reference_terms(covariance, reference_std: float, lipschitz: float) -> dict:
    covariance = np.asarray(covariance, dtype=np.float64)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or
            covariance.shape[0] < 1 or not np.isfinite(covariance).all() or
            not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("Require a finite symmetric covariance matrix")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Require positive definite deployment covariance")
    if not math.isfinite(reference_std) or reference_std <= 0 or not math.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("Reference std and Lipschitz constant must be positive and finite")
    dimension = len(covariance)
    variance = reference_std**2
    logdet = np.linalg.slogdet(covariance)[1]
    divergence = .5 * (np.trace(covariance) / variance - dimension + dimension * math.log(variance) - logdet)
    log_moment = log_chi_mgf(lipschitz * reference_std, dimension)
    return dict(reference_std=reference_std, log_exponential_moment=log_moment,
                channel_kl=float(divergence), c1_second_shift_upper=log_moment + float(divergence))


def gaussian_logit_channel_terms(sigma: float, correlation: float, epsilon: float = .01) -> dict:
    """Two output logits, equal marginal variance and fixed correlation."""
    if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(correlation) or abs(correlation) >= 1:
        raise ValueError("Require sigma>=0 and |correlation|<1")
    k = smooth_loss_lipschitz(epsilon)
    if sigma == 0:
        return dict(sigma=0., correlation=correlation, lipschitz=k, covariance=[[0., 0.], [0., 0.]],
                    direct_first_shift_term=0., reference_std=0., log_exponential_moment=0., channel_kl=0.,
                    c1_second_shift_upper=0., reference_std_equal_to_sigma_upper=0.,
                    reference_slack=0., kl_to_perfect_training="zero")
    covariance = sigma**2 * np.array([[1., correlation], [correlation, 1.]])
    # MGF increases with s; the KL alone is minimized at s=sigma. Thus the
    # optimum is below sigma. Search a broad interval in log scale, using
    # channel law and epsilon only, never the posterior or observed risks.
    result = minimize_scalar(
        lambda log_ratio: gaussian_reference_terms(covariance, sigma * math.exp(log_ratio), k)["c1_second_shift_upper"],
        bounds=(-7., 0.), method="bounded", options={"xatol": 1e-12})
    if not result.success:
        raise RuntimeError("Analytic reference-scale minimization failed")
    terms = gaussian_reference_terms(covariance, sigma * math.exp(result.x), k)
    # Exact E||N(0,Sigma)|| in 2D: angular integration gives a complete
    # elliptic integral of the second kind. This is a benchmark for C1's
    # first inequality, not an input to the second-inequality calculation.
    largest = sigma**2 * (1 + abs(correlation))
    smallest = sigma**2 * (1 - abs(correlation))
    direct = k * math.sqrt(2 / math.pi) * math.sqrt(largest) * float(ellipe(1 - smallest/largest))
    return dict(sigma=sigma, correlation=correlation, lipschitz=k, covariance=covariance.tolist(),
                direct_first_shift_term=direct, **terms,
                reference_std_equal_to_sigma_upper=gaussian_reference_terms(covariance, sigma, k)["c1_second_shift_upper"],
                reference_slack=terms["c1_second_shift_upper"] - direct,
                kl_to_perfect_training="infinite")


def erasure_first_shift(width: int, probability: float) -> float:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1 or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Require positive integer width and erasure probability in [0,1]")
    return 1. if probability == 1 else -math.expm1(width * math.log1p(-probability))


def centered_lognormal_reference_terms(sigma: float, shape: float,
                                     reference_std: float, lipschitz: float,
                                     dimension: int = 2) -> dict:
    """Exact entropy-based KL for known standardized centered lognormal noise.

    N_i=sigma*(exp(shape*Z_i-shape**2/2)-1)/sqrt(expm1(shape**2)),
    with independent standard normals Z_i. Q=N(0,reference_std**2 I).
    Every N_i has mean zero, variance sigma**2, and a shifted lognormal
    density; absolute continuity P<<Q holds despite the unequal supports.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (sigma, shape, reference_std, lipschitz)):
        raise ValueError("Require positive finite scale, shape, reference std, and Lipschitz constant")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Require a positive integer dimension")
    shape_sq = shape * shape
    entropy_deficit = .5 * (math.log(math.expm1(shape_sq) / shape_sq) + shape_sq)
    divergence = dimension * (math.log(reference_std / sigma) +
                              sigma**2 / (2 * reference_std**2) - .5 + entropy_deficit)
    log_moment = log_chi_mgf(lipschitz * reference_std, dimension)
    return dict(reference_std=reference_std, log_exponential_moment=log_moment,
                channel_kl=divergence, c1_second_shift_upper=log_moment + divergence)


def lognormal_distance_benchmark(shape: float, samples: int = 2_000_000,
                                seed: int = 879513) -> dict:
    """Independent MC diagnostic of E||N|| at sigma=1, not a certificate.

    The reported standard error is descriptive, not a rigorous confidence
    radius for an unbounded lognormal distance. No result from this function
    is used in computing the analytic second-inequality bound.
    """
    if not math.isfinite(shape) or shape <= 0 or samples < 2:
        raise ValueError("Require positive finite shape and at least two samples")
    rng = np.random.default_rng(seed)
    variance = math.expm1(shape * shape)
    count, total, squares = 0, 0., 0.
    while count < samples:
        size = min(100_000, samples-count)
        z = rng.standard_normal((size, 2))
        noise = np.expm1(shape*z - shape*shape/2) / math.sqrt(variance)
        norms = np.linalg.norm(noise, axis=1)
        total += float(norms.sum())
        squares += float(np.dot(norms, norms))
        count += size
    mean = total/count
    variance_estimate = (squares-count*mean*mean)/(count-1)
    return dict(unit_scale_mean_norm=mean,
                unit_scale_mean_norm_standard_error=math.sqrt(variance_estimate/count),
                samples=samples, seed=seed,
                status="Independent diagnostic MC only; not a certified first-inequality bound")


def lognormal_logit_channel_terms(sigma: float, shape: float = .1,
                                 epsilon: float = .01, benchmark: dict | None = None) -> dict:
    if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(shape) or shape <= 0:
        raise ValueError("Require finite sigma>=0 and shape>0")
    k = smooth_loss_lipschitz(epsilon)
    if sigma == 0:
        return dict(sigma=0., shape=shape, lipschitz=k, direct_first_shift_term=0.,
                    direct_first_shift_mc_standard_error=0., reference_std=0.,
                    log_exponential_moment=0., channel_kl=0., c1_second_shift_upper=0.,
                    reference_std_equal_to_sigma_upper=0., jensen_shift_upper=0., kl_to_perfect_training="zero")
    # Fixed, moment-matched reference: the entire second expression is closed
    # form, including the choice s=sigma. No numerical optimization is needed.
    terms = centered_lognormal_reference_terms(sigma, shape, sigma, k)
    return dict(sigma=sigma, shape=shape, lipschitz=k, **terms,
                direct_first_shift_term=None if benchmark is None else k*sigma*benchmark["unit_scale_mean_norm"],
                direct_first_shift_mc_standard_error=None if benchmark is None else
                    k*sigma*benchmark["unit_scale_mean_norm_standard_error"],
                reference_std_equal_to_sigma_upper=centered_lognormal_reference_terms(
                    sigma, shape, sigma, k)["c1_second_shift_upper"],
                jensen_shift_upper=k*sigma*math.sqrt(2.),
                kl_to_perfect_training="infinite")


def _metric(logits, labels, config):
    losses = F.cross_entropy(logits, labels, reduction="none")
    return dict(bounded_loss=float(bounded_losses_from_ce(losses, config).mean()),
                accuracy=float((logits.argmax(-1) == labels).double().mean()),
                cross_entropy=float(losses.mean()))


def _mean_metrics(replicates):
    return {name: statistics.fmean(row[name] for row in replicates)
            for name in ("bounded_loss", "accuracy", "cross_entropy")}


@torch.no_grad()
def evaluate_frozen(checkpoint: Path, draws: int, seed: int, shape: float,
                    erasure_probabilities: list[float], noise_sigmas: list[float],
                    data_setups: int, benchmark: dict, mc_setups: int = 3) -> dict:
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = normalize_config({**metadata["config"], "device": "cpu", "threads": 2})
    if (config["dataset"] != "moons" or config["moons_sampler"] != "iid" or
            config["train_channel"] != {"kind": "perfect"} or config["training_loss"] != "bounded_ce_smooth"):
        raise ValueError("Require an IID two-moons, smooth-bounded-loss, perfect-trained checkpoint")
    source_verified = verify_sampler_source(metadata["git_sha"])
    train_data, _, test_data, input_dim, classes, split = load_data(config, include_test=True)
    if classes != 2:
        raise ValueError("This declared output channel has exactly two output logits")
    for name in ("train_indices_hash", "validation_indices_hash"):
        if split[name] != metadata["split"][name]:
            raise ValueError("Recreated data split differs from the saved checkpoint")
    model, bank = make_models(config, input_dim, classes)
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    if bank_hash(bank) != metadata["training_channel_bank_hash"]:
        raise ValueError("Loaded channel bank does not match the recorded training bank")
    model.eval()
    x_train, y_train = train_data[:]
    x_test, y_test = test_data[:]
    empirical_replicates = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(draws):
        theta = model.sample(generator)
        w1, b1, w2, b2 = model.split(theta)
        logits = F.relu(x_train @ w1 + b1) @ w2 + b2
        empirical_replicates.append(_metric(logits, y_train, config))
    empirical = _mean_metrics(empirical_replicates)
    kl = gaussian_kl(model.mu.detach().double(), model.rho.detach().double(), config["prior_std"])
    if min(data_setups, mc_setups) < 1:
        raise ValueError("Data and MC setup counts must be positive")
    data_failure, mc_failure = .04 / data_setups, .01 / mc_setups
    data_term = theorem1_terms(kl, config["train_size"], config["m"], delta=data_failure,
                               perfect_training=True)["sample_complexity"].item()
    mc_radius = math.sqrt(math.log(1 / mc_failure) / (2 * draws))
    baseline = empirical["bounded_loss"] + mc_radius + data_term
    lognormal_terms = [lognormal_logit_channel_terms(sigma, shape, config["loss_epsilon"], benchmark) for sigma in noise_sigmas]
    erasure_replicates = [[] for _ in erasure_probabilities]
    lognormal_replicates = [[] for _ in noise_sigmas]
    generator = torch.Generator().manual_seed(seed + 100000)
    channel_generator = torch.Generator().manual_seed(seed + 200000)
    for _ in range(draws):
        theta = model.sample(generator)
        w1, b1, w2, b2 = model.split(theta)
        hidden = F.relu(x_test @ w1 + b1)
        logits = hidden @ w2 + b2
        uniforms = torch.rand(hidden.shape, generator=channel_generator)
        z = torch.randn(logits.shape, generator=channel_generator)
        standard_noise = torch.expm1(shape*z-shape*shape/2) / math.sqrt(math.expm1(shape*shape))
        for probability, replicates in zip(erasure_probabilities, erasure_replicates):
            erased_logits = hidden * (uniforms >= probability) @ w2 + b2
            replicates.append(_metric(erased_logits, y_test, config))
        for sigma, replicates in zip(noise_sigmas, lognormal_replicates):
            replicates.append(_metric(logits + sigma * standard_noise, y_test, config))
    identity_loss = _mean_metrics(erasure_replicates[erasure_probabilities.index(0.)])["bounded_loss"]
    erasure_rows, lognormal_rows = [], []
    for probability, replicates in zip(erasure_probabilities, erasure_replicates):
        shift = erasure_first_shift(config["width"], probability)
        metrics = _mean_metrics(replicates)
        erasure_rows.append(dict(channel_family="hidden_feature_erasure", severity=probability,
            **metrics, observed_shift=metrics["bounded_loss"] - identity_loss,
            direct_first_shift_term=shift, direct_first_shift_mc_standard_error=0., jensen_shift_upper=None,
            log_exponential_moment=None, channel_kl=None,
            c1_second_shift_upper=None, reference_std=None,
            first_risk_upper_uncapped=baseline + shift, second_risk_upper_uncapped=None))
    for terms, replicates in zip(lognormal_terms, lognormal_replicates):
        metrics = _mean_metrics(replicates)
        lognormal_rows.append(dict(channel_family="additive_output_logit_centered_lognormal", severity=terms["sigma"],
            **metrics, observed_shift=metrics["bounded_loss"] - identity_loss,
            direct_first_shift_term=terms["direct_first_shift_term"],
            direct_first_shift_mc_standard_error=terms["direct_first_shift_mc_standard_error"],
            jensen_shift_upper=terms["jensen_shift_upper"],
            log_exponential_moment=terms["log_exponential_moment"], channel_kl=terms["channel_kl"],
            c1_second_shift_upper=terms["c1_second_shift_upper"], reference_std=terms["reference_std"],
            first_risk_upper_uncapped=None,
            second_risk_upper_uncapped=baseline + terms["c1_second_shift_upper"]))
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_git_sha=metadata["git_sha"],
                weights_sha256=hashlib.sha256((checkpoint / "weights.pt").read_bytes()).hexdigest(),
                seed=config["seed"], inference_seed=seed, draws=draws, test_examples=len(test_data),
                training_size=config["train_size"], width=config["width"], epsilon=config["loss_epsilon"],
                posterior_kl=kl.item(), empirical=empirical, empirical_replicates=empirical_replicates,
                empirical_mc_radius=mc_radius, data_complexity=data_term,
                shared_baseline_upper_uncapped=baseline, data_failure=data_failure, mc_failure=mc_failure,
                sampler_source_verification=source_verified, training_channel_bank_verified=True,
                erasure=erasure_rows, lognormal=lognormal_rows, lognormal_analytic_terms=lognormal_terms,
                deployment_replicates=dict(erasure=erasure_replicates, lognormal=lognormal_replicates))


def _aggregate(reports, family):
    rows = []
    for i, template in enumerate(reports[0][family]):
        row = dict(channel_family=template["channel_family"], severity=template["severity"], seeds=len(reports))
        for key in template:
            if key in {"channel_family", "severity"}:
                continue
            values = [report[family][i][key] for report in reports]
            row[key] = None if values[0] is None else statistics.fmean(values)
            row[key + "_sd"] = None if values[0] is None else statistics.stdev(values) if len(values) > 1 else 0.
        for key, getter in (("empirical_bounded_loss", lambda r: r["empirical"]["bounded_loss"]),
                            ("data_complexity", lambda r: r["data_complexity"]),
                            ("empirical_mc_radius", lambda r: r["empirical_mc_radius"])):
            values = [getter(report) for report in reports]
            row[key], row[key + "_sd"] = statistics.fmean(values), statistics.stdev(values) if len(values)>1 else 0.
        rows.append(row)
    return rows


def _write_outputs(output_root: Path, data: dict):
    assets, tables = output_root / "assets", output_root / "tables"
    assets.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    combined_rows = [row for family in ("erasure", "lognormal") for row in data["summary"][family]]
    with (tables / "c1_evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    combined_tex = [r"\begin{table*}[t]", r"\centering\footnotesize",
        r"\caption{C1 evaluation with identical perfect-channel training: the same three frozen posteriors face "
        r"either hidden-feature erasures or centered lognormal output-logit noise. Test accuracy and risk "
        r"are mean $\pm$ sample SD across all three seeds; risks use normalized smooth loss. Empirical risk, "
        r"the data term, and the fresh empirical-MC correction are shared by both deployment cases. "
        r"Erasure uses C1's first inequality with the discrete metric and $K=1$. Centered lognormal output noise uses "
        r"the second inequality with the Euclidean metric and $K=\sqrt2/\log100$. For lognormal rows, "
        r"the first term is an independent MC diagnostic (not a certified bound); the risk upper uses the analytic second term. The upper columns are "
        r"uncapped means of the per-seed bounds, which include empirical MC error; values above one are vacuous.}",
        r"\label{tab:c1_evaluation}", r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Severity & Test acc. (\%) & Test risk & Emp. risk & First shift$^*$ & Jensen & Second shift & Risk upper\\", r"\midrule"]
    for family in ("erasure", "lognormal"):
        rows = data["summary"][family]
        combined_tex.append(r"\multicolumn{8}{l}{\emph{" +
                            (r"Hidden-feature erasure: severity $p$" if family == "erasure" else
                             r"Centered lognormal output noise: severity $\sigma$, shape $\tau=0.1$") + r"}}\\")
        for row in rows:
            values = [f"{row['severity']:.2f}",
                      f"${100*row['accuracy']:.2f}\\pm{100*row['accuracy_sd']:.2f}$",
                      f"${row['bounded_loss']:.4f}\\pm{row['bounded_loss_sd']:.4f}$",
                      f"{row['empirical_bounded_loss']:.4f}", f"{row['direct_first_shift_term']:.4f}",
                      "--" if family == "erasure" else f"{row['jensen_shift_upper']:.4f}",
                      "--" if family == "erasure" else f"{row['c1_second_shift_upper']:.4f}",
                      f"{row['first_risk_upper_uncapped'] if family == 'erasure' else row['second_risk_upper_uncapped']:.4f}"]
            combined_tex.append(" & ".join(values) + r"\\")
        if family == "erasure":
            combined_tex.append(r"\midrule")
        with (tables / f"c1_{family}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [r"\begin{table*}[t]", r"\centering\footnotesize"]
        if family == "erasure":
            lines += [r"\caption{C1 first inequality for hidden-feature erasures, evaluated without retraining on the same three perfect-trained posteriors. "
                      r"The discrete channel metric admits the uniform constant $K=1$. Risk denotes normalized bounded loss; "
                      r"test risk is a held-out estimate. Risk upper bounds include empirical MC correction and the data penalty. "
                      r"Means are over all three seeds; uncapped bounds above one are vacuous.}",
                      r"\label{tab:c1_erasure}", r"\begin{tabular}{rrrrrr}", r"\toprule",
                      r"$p$ & Test acc. (\%) & Test risk & Emp. risk & First shift & Risk upper\\", r"\midrule"]
        else:
            lines += [r"\caption{C1 second inequality for known centered lognormal noise on the output logits of the same perfect-trained posteriors. "
                      r"$N_i=\sigma(e^{\tau Z_i-\tau^2/2}-1)/\sqrt{e^{\tau^2}-1}$, $\tau=.1$; the fixed reference is $Q=\mathcal N(0,\sigma^2I)$. "
                      r"The first shift is an independent two-million-draw MC diagnostic, not a certified bound. The Euclidean loss-Lipschitz constant is "
                      r"$\sqrt2/\log100$. The exact reference log moment plus channel KL gives the second shift upper. "
                      r"Risk upper includes empirical MC correction and the data penalty; uncapped values above one are vacuous.}",
                      r"\label{tab:c1_lognormal}", r"\begin{tabular}{rrrrrrrr}", r"\toprule",
                      r"$\sigma$ & Test acc. (\%) & Test risk & First shift MC & $\log\mathbb E_Qe^{Kd}$ & $D(P\Vert Q)$ & Second shift & Risk upper\\", r"\midrule"]
        for row in rows:
            values = [f"{row['severity']:.2f}", f"{100*row['accuracy']:.2f}", f"{row['bounded_loss']:.4f}"]
            if family == "erasure":
                values += [f"{row['empirical_bounded_loss']:.4f}", f"{row['direct_first_shift_term']:.4f}", f"{row['first_risk_upper_uncapped']:.4f}"]
            else:
                values += [f"{row[key]:.4f}" for key in ("direct_first_shift_term", "log_exponential_moment", "channel_kl", "c1_second_shift_upper", "second_risk_upper_uncapped")]
            lines.append(" & ".join(values) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        (tables / f"c1_{family}.tex").write_text("\n".join(lines))
    combined_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (tables / "c1_evaluation.tex").write_text("\n".join(combined_tex))
    (assets / "c1_evaluation.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/c1-reference-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.25))
    for family, ax in zip(("erasure", "lognormal"), axes):
        rows = data["summary"][family]
        x = [r["severity"] for r in rows]
        ax.errorbar(x, [r["bounded_loss"] for r in rows], yerr=[r["bounded_loss_sd"] for r in rows],
                    color="#009E73", marker="o", label="Held-out bounded risk", capsize=2)
        if family == "erasure":
            ax.plot(x, [r["first_risk_upper_uncapped"] for r in rows], color="#D55E00", linestyle="--",
                    marker="s", markerfacecolor="none", label="First-inequality risk upper")
        else:
            ax.plot(x, [r["empirical_bounded_loss"]+r["data_complexity"]+r["empirical_mc_radius"]+r["direct_first_shift_term"] for r in rows], color="#D55E00", linestyle="--",
                    marker="s", markerfacecolor="none", label="Direct expectation MC diagnostic")
        if family == "lognormal":
            ax.plot(x[1:], [r["second_risk_upper_uncapped"] for r in rows[1:]], color="#0072B2", marker="D",
                    markersize=4, label="Second-inequality risk upper")
            ax.scatter([x[0]], [rows[0]["second_risk_upper_uncapped"]], color="#0072B2", marker="D", s=16)
            ax.plot(x, [r["empirical_bounded_loss"]+r["data_complexity"]+r["empirical_mc_radius"]+r["jensen_shift_upper"] for r in rows],
                    color="#777777", linestyle=":", linewidth=1.2, label="Jensen risk upper")
        ax.axhline(1., color="#777777", linestyle=":", linewidth=.8)
        ax.set(xlabel="Erasure probability" if family == "erasure" else r"Output-noise standard deviation $\sigma$",
               ylabel="Normalized bounded-loss units", title="(a) Feature erasure" if family == "erasure" else "(b) Centered lognormal output noise")
        ax.grid(alpha=.2, linewidth=.5)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout(w_pad=2.)
    for extension in ("png", "pdf"):
        fig.savefig(assets / f"fig_c1_evaluation.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=Path("results/journal_campaign/followup"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "paper")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=995271)
    parser.add_argument("--data-setups", type=int, default=473,
                        help="Conservative union allowance for earlier journal checkpoint/model/data setup selection")
    args = parser.parse_args()
    if args.draws < 1 or args.data_setups < 1:
        parser.error("draws and data-setups must be positive")
    torch.set_num_threads(2)
    erasures, sigmas, shape = [0., .01, .05, .1, .25, .5], [0., .1, .25, .5, 1., 2.], .1
    benchmark = lognormal_distance_benchmark(shape)
    reports = []
    for i, seed in enumerate((9001, 9002, 9003)):
        checkpoint = args.campaign_root / f"mechanism_moons_outage_perfect_n512_PAC_Perfect_886ce4f0737a_s{seed}"
        reports.append(evaluate_frozen(checkpoint, args.draws, args.seed + 1000*i, shape, erasures, sigmas, args.data_setups, benchmark))
        print(json.dumps({"seed": seed, "empirical_loss": reports[-1]["empirical"]["bounded_loss"],
                          "data_term": reports[-1]["data_complexity"], "empirical_mc_radius": reports[-1]["empirical_mc_radius"]}), flush=True)
    data = dict(description="Inference-only C1 evaluation of the same three frozen perfect-trained posteriors under two deployment channel families.",
        loss="-log(epsilon+(1-epsilon)*p_y)/log(1/epsilon), epsilon=.01",
        confidence=dict(total_failure=.05, data_failure_total=.04, empirical_mc_failure_total=.01,
                        simultaneous_data_setups=args.data_setups, fresh_empirical_mc_evaluations=3,
                        data_selection_allowance_source="journal/CAMPAIGN_RESULTS.md: 473 unique completed earlier journal trials; this intentionally overcounts repeated data/prior/model setups.",
                        note="Union allocation over all declared earlier journal trials accounts conservatively for choosing these checkpoints after prior campaign exploration. Uniformity over the posterior covers algorithm selection within each data/prior/model setup. Three checkpoints are fixed before the fresh empirical MC streams; MC budget is divided only among those three evaluations. Analytic channel shift bounds apply simultaneously to every displayed severity.",
                        limitation="Conditional on the supplied theorem assumptions and a declared data-setup count covering all considered data/prior/model setups. Held-out test metrics have no claimed population confidence interval."),
        split_change="Erasure remains at hidden width-eight features. Centered lognormal additive noise is on the two output logits, with the full trained NN as transmitter and softmax as receiver. Perfect-channel training predictor is identical.",
        smooth_loss_lipschitz_derivation="grad ell=a(p_y)(softmax-e_y)/log(1/epsilon), 0<=a<=1; hence norm<=sqrt(2)/log(1/epsilon) uniformly over weights, data, labels, logits.",
        lognormal_reference=dict(deployment="Independent known centered standardized lognormal coordinates; N_i=sigma*(exp(tau*Z_i-tau**2/2)-1)/sqrt(expm1(tau**2)), tau=.1", reference="N(0,s**2 I_2)",
            shape=shape, direct_distance_benchmark=benchmark,
            exact_kl="D*(log(s/sigma)+sigma**2/(2*s**2)-1/2+1/2*log(expm1(tau**2)/tau**2)+tau**2/2)",
            reference_choice="Fixed moment-matched s=sigma, known from the channel law. The full second expression requires no numerical integration or numerical optimization. No deployment-distance MC or test metrics enter the bound.",
            chi_mgf="M_2(t)=1+t*sqrt(2*pi)*exp(t**2/2)*Phi(t)",
            direct_benchmark="Independent2m-sample MC diagnostic with standard error; no simple known closed form for the two-dimensional centered-lognormal mean Euclidean norm. Not a certified first bound.",
            singularity="For sigma>0, KL(P||delta_0)=infinite; P<<Gaussian Q, reference exponential moment and exact KL are finite."),
        scope=["No model training or hyperparameter selection was performed.",
               "The second inequality is a closed-form computable relaxation; direct deployment mean distance is shown only as an independent MC diagnostic.",
               "Held-out risks and accuracies are finite-test Gibbs MC estimates. SD bars describe three initialization seeds, not rigorous test-population confidence intervals.",
               "Each per-seed risk upper concerns smooth bounded Gibbs loss, includes a fresh empirical MC correction, and is conditional on the supplied C1/data theorem assumptions.",
               "Means of per-seed bounds bound the mean risk of the uniform mixture over these posteriors on the simultaneous event; they do not bound each seed by that mean.",
               "Uncapped bounds above one are vacuous; no claim of a numerical classification-error or raw-CE certificate is made.",
               "The additive-output Lipschitz argument does not establish a uniform Euclidean K for the existing hidden feature Gaussian channel."],
        reports=reports, summary={family: _aggregate(reports, family) for family in ("erasure", "lognormal")})
    _write_outputs(args.output_root, data)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "draws": args.draws, "checkpoints": len(reports)}))


if __name__ == "__main__":
    main()
