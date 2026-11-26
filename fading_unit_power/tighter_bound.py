"""Tighter two-moons (width 8, m=2000) certificates by direct bound minimization.

Same data S, bank U, seeds, channels, architecture caps and confidence allocation as the
archived bound decomposition. Only the posterior (its training configuration) changes:
means *and* standard deviations are trained to minimize the exact certificate they are
scored with, instead of a cross-entropy surrogate with small penalty weights selected by
validation accuracy.

* C1 (perfect training): empirical (perfect channel) + Corollary 1 sample term + shift.
* T1 (Theorem 1): empirical on S x U + sample term with Q_W + channel term with the
  fresh data-dependent prior Q_{W|S} + shift. Q_{W|S} = N(mu_S, (c sigma_S)^2) where
  (mu_S, sigma_S) is a stage-1 posterior trained on S with fresh P_tr draws, never U.
  Stage-1 hyperparameters are chosen by an S-only criterion, so Q_{W|S} stays independent
  of U; the union bound counts the scale grid c (K = len(PRIOR_SCALES)).
* T2 (Theorem 2): empirical on S x U + joint term with Q_W + shift.

Q_W = N(0, prior_std^2 I) is the archived data-free prior. Every trained posterior is a
candidate: the bounds hold simultaneously for all posteriors under fixed priors, so the
smallest bound is picked with one Monte Carlo seed and certified with an independent one.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "results" / "final_chosen_results" / "data" / "bounds"
FRESH_ARCHIVE = REPO / "results" / "final_chosen_results" / "data_dependent" / "only_fresh" / "bounds"
PUBLISH = REPO / "results" / "final_chosen_results" / "tighter_bound"
FAMILIES = ("fading", "erasure")
CASES = ("perfect", "t1_matched", "t2_matched", "t1_mismatch", "t2_mismatch")

THEOREM_FAILURE = .00125          # archived: 20 statements at delta = 0.05
MC_FAILURE = .00125
SIGMA_SQ = .25                    # Hoeffding proxy for a loss in [0, 1]
SELECTION_MC_SEED = 271_828
CERTIFICATE_MC_SEED = 314_159
STAGE1_MC_SEED = 161_803
TEST_SEED = 577_215
SELECTION_OBSERVATIONS = 200_000
CERTIFICATE_OBSERVATIONS = 2_000_000
TEST_REPLICATES = 100             # passes over the 10,000 test points
PRIOR_SCALES = (1., 1.5, 2.)      # Q_{W|S} = N(mu_S, (c sigma_S)^2)

# Posterior training grid (free: the bounds are uniform over posteriors).
INIT_STDS = (.24, .1, .03)
LEARNING_RATES = (.003, .01)
# Training objective: empirical + weight * complexity. The certificate always uses weight 1;
# smaller weights only shape the posterior, so learned predictors become candidates.
PENALTY_WEIGHTS = (1., .3, .1, .03, 0.)
# Surrogate receiver cap during training, as a multiple of the certified cap L. The receiver
# projection scales w2 and b2 together, so argmax (accuracy) is unchanged and the certificate is
# always evaluated at the archived L: a larger training cap only makes the loss value accuracy.
TRAIN_CAP_SCALES = (1., 8., 32.)
CHECKPOINTS = (250, 1000, 2000, 4000, 8000)
STAGE2_PENALTY_WEIGHTS = (1., .3)
STAGE2_LEARNING_RATES = (.001, .003)
STAGE2_CHECKPOINTS = (0, 100, 300, 1000, 3000)
VALIDATION_REPLICATES = 100
# Selection rules: smallest bound among candidates whose validation accuracy is within the
# tolerance of the best candidate (None: no accuracy constraint).
TOLERANCES = dict(tightest=None, balanced=.03, accurate=.01)
RULES = tuple(TOLERANCES)
REPLICATES = 4                    # weight/channel draws per training example per step


# ----------------------------------------------------------------------------- configs

def archived(family, case):
    return json.loads((ARCHIVE / family / case / "bound.json").read_text())


def case_config(family, case, receiver_norm=None):
    from .trainer.run_experiment import normalize_config

    config = dict(archived(family, case)["config"])
    config.update(posterior_mean_clip=0., rho_lr_scale=1., threads=1, device="cpu")
    if receiver_norm is not None:
        config["bounded_receiver_norm"] = receiver_norm
    return normalize_config(config)


def certificate_kind(case):
    return "C1" if case == "perfect" else case[:2].upper()


# ----------------------------------------------------------------------------- model helpers

def build(config, init_std):
    import torch
    from .trainer.run_experiment import load_data, make_models

    torch.set_num_threads(1)
    train, val, test, input_dim, classes, split = load_data(config, include_test=True)
    model, bank = make_models({**config, "posterior_std": init_std}, input_dim, classes)
    model.rho.requires_grad_(True)

    def tensors(data):
        return (torch.stack([data[i][0] for i in range(len(data))]),
                torch.tensor([int(data[i][1]) for i in range(len(data))]))

    model.validation = tensors(val)
    return model, bank, tensors(train), test.tensors, split


def gibbs_logits(model, x, bank, source, channel, weight_generator, channel_generator):
    """Independent weight draw per row; channel from the bank U, fresh draws, or none."""
    import torch
    from torch.nn import functional as F

    raw = model.mu + F.softplus(model.rho) * torch.randn(
        (len(x), model.d), dtype=model.mu.dtype, generator=weight_generator)
    a, b, c = model.input_dim * model.width, model.d1, model.d1 + model.width * model.classes
    w1 = raw[:, :a].reshape(-1, model.input_dim, model.width)
    w2 = raw[:, b:c].reshape(-1, model.width, model.classes)
    hidden = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + raw[:, a:b]
    hidden = model.project_features(F.relu(hidden))
    if source != "perfect":
        hidden = bank.apply(hidden, source, channel, channel_generator)
    w2, b2 = model.project_receiver(w2, raw[:, c:])
    return torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2


def bounded(logits, y, config):
    from torch.nn import functional as F
    from .trainer.run_experiment import bounded_losses_from_ce

    return bounded_losses_from_ce(F.cross_entropy(logits, y, reduction="none"), config)


def kl_diag(mu, rho, prior_mu, prior_std):
    """KL(N(mu, softplus(rho)^2) || N(prior_mu, prior_std^2)); prior_std scalar or vector."""
    import torch
    from torch.nn import functional as F

    prior_std = torch.as_tensor(prior_std, dtype=mu.dtype)
    ratio = (F.softplus(rho) / prior_std).square()
    return .5 * (ratio + ((mu - prior_mu) / prior_std).square() - 1 - ratio.log()).sum()


# ----------------------------------------------------------------------------- bound terms

def sample_term(kl, n, eps):
    return (2 * SIGMA_SQ / (n - 1) * (kl + math.log(math.sqrt(n) / eps))) ** .5


def channel_term(kl, m, eps, count):
    return (2 * SIGMA_SQ / (m - 1) * (kl + math.log(math.sqrt(m) * count / eps))) ** .5


def joint_term(kl, n, m, delta):
    return ((2 * SIGMA_SQ / (n - 1) + 2 * SIGMA_SQ / (m - 1))
            * (kl + math.log(math.sqrt(n * m) / delta))) ** .5


def failures(kind, architectures):
    """Archived allocation; a grid of G architectures divides each theorem event by G."""
    theorem = THEOREM_FAILURE / architectures
    return dict(theorem=theorem, eps_1=theorem / 2, eps_2=theorem / 2) if kind in ("T1", "stage1") else dict(theorem=theorem)


def complexity(kind, model, config, prior, architectures):
    """Exact complexity terms (torch tensors when the posterior is a torch model)."""
    n, m = config["train_size"], config["m"]
    fail = failures(kind, architectures)
    kl = kl_diag(model.mu, model.rho, 0., config["prior_std"])
    if kind == "C1":
        return dict(sample=sample_term(kl, n, fail["theorem"])), dict(kl=kl)
    if kind == "T2":
        return dict(joint=joint_term(kl, n, m, fail["theorem"])), dict(kl=kl)
    if kind == "stage1":
        return dict(sample=sample_term(kl, n, fail["eps_1"])), dict(kl=kl)
    channel_kls = [kl_diag(model.mu, model.rho, prior["mu"], scale * prior["std"]) for scale in PRIOR_SCALES]
    channels = [channel_term(value, m, fail["eps_1"], len(PRIOR_SCALES)) for value in channel_kls]
    best = min(range(len(channels)), key=lambda i: float(channels[i]))
    return (dict(sample=sample_term(kl, n, fail["eps_2"]), channel=channels[best]),
            dict(kl=kl, channel_kl=channel_kls[best], prior_scale=PRIOR_SCALES[best],
                 channel_family=[dict(scale=s, kl=float(k), channel=float(c))
                                 for s, k, c in zip(PRIOR_SCALES, channel_kls, channels)]))


def shift_term(config, kind, model):
    """Posterior-independent shift; for C1 the smallest valid Corollary 1 inequality."""
    from .trainer.diagonal_fading import diagonal_gain_kl

    deployment, training, width = config["deployment_channel"], config["train_channel"], config["width"]
    eps = config["loss_epsilon"]
    if kind in ("T1", "T2"):
        if deployment == training:
            return dict(method=f"{kind} channel KL", analytic=True, upper=0., channel_kl=0.)
        if deployment["kind"] == "diagonal_fading":
            divergence = diagonal_gain_kl(deployment, training, width)
        else:
            p, q = deployment["outage_prob"], training["outage_prob"]
            divergence = width * (p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))
        return dict(method=f"{kind} channel KL", analytic=True, upper=math.sqrt(2 * SIGMA_SQ * divergence),
                    channel_kl=divergence)
    if deployment["kind"] == "outage":
        p = deployment["outage_prob"]
        distance = math.fsum(math.sqrt(k) * math.comb(width, k) * p ** k * (1 - p) ** (width - k)
                             for k in range(1, width + 1))
        lipschitz = model.feature_norm * model.channel_lipschitz_constant(eps)
        return dict(method="C1 first", metric="Frobenius mask distance", analytic=True, expectation=distance,
                    uniform_lipschitz=lipschitz, upper=lipschitz * distance)
    return fading_perfect_shift(deployment, width, model, eps)


def fading_perfect_shift(deployment, width, model, eps):
    """Corollary 1 for diagonal fading with D(C, c0) = sqrt(||h - 1||^2 + ||N||^2).

    first:  K E_{P_ch} D <= K sqrt(E D^2) = K sqrt(d (1 + 1/4) + d)   (Jensen; gains N(1/2, 1))
    second (Q = P_ch): log E_{P_ch} e^{K D}, KL = 0. D is rotation invariant, so rotate the
            gain to its correlation eigenbasis and use the Young/Gaussian quadratic-moment
            bound at the best a on a grid.
    second (Q = P_iid): the archived closed form, reported for comparison.
    All three are deterministic upper bounds on the same quantity; the smallest is used.
    """
    import numpy as np
    from .trainer.diagonal_fading import diagonal_gain_kl
    from .trainer.full_fading import correlation_factor

    k = model.channel_lipschitz_constant(eps) * math.hypot(model.feature_norm, 1.)
    first = k * math.sqrt(width * 1.25 + width)

    factor = correlation_factor(deployment["rho_c"], width).double().numpy()
    values, vectors = np.linalg.eigh(factor @ factor.T)
    variances = np.concatenate([values, np.ones(width)])
    means = np.concatenate([vectors.T @ np.full(width, -.5), np.zeros(width)])

    def young(a):
        return k * k / (2 * a) + .5 * float(np.sum(-np.log1p(-a * variances) + a * means ** 2 / (1 - a * variances)))

    second_pch = min(young(a) for a in np.geomspace(1e-6, .999 / variances.max(), 4000))

    noncentrality = width / 4.
    a = min(.5, k / math.sqrt(width * 2 + noncentrality))
    iid_moment = k * k / (2 * a) - width * math.log1p(-a) + a * noncentrality / (2 * (1 - a))
    second_iid = diagonal_gain_kl(deployment, dict(deployment, rho_c=0.), width) + iid_moment

    options = dict(first_jensen=first, second_reference_pch=second_pch, second_reference_iid=second_iid)
    method = min(options, key=options.get)
    return dict(method=f"C1 {method}", metric="sqrt(||H-H'||_F^2 + ||N-N'||_2^2)", analytic=True,
                uniform_lipschitz=k, options=options, upper=options[method])


# ----------------------------------------------------------------------------- training

def fit(config, kind, init_std, lr, weight, checkpoints, seed, prior=None, start=None, architectures=1,
        cap_scale=1.):
    """Adam on (mu, rho) for empirical + weight * exact complexity; states at checkpoints.

    ``cap_scale`` multiplies the receiver cap during training only (a surrogate loss).
    """
    import torch

    model, bank, (x, y), _, split = build(config, init_std)
    model.receiver_norm = model.receiver_norm * cap_scale
    if start is not None:
        with torch.no_grad():
            model.mu.copy_(start["mu"])
            model.rho.copy_(start["rho"])
    if kind == "stage1":
        source, channel = "deployment", config["train_channel"]   # fresh P_tr draws, never U
    elif config["train_channel"]["kind"] == "perfect":
        source, channel = "perfect", None
    else:
        source, channel = "bank", None
    optimizer = torch.optim.Adam([model.mu, model.rho], lr=lr)
    weight_generator = torch.Generator().manual_seed(seed)
    channel_generator = torch.Generator().manual_seed(seed + 1)
    xr, yr = x.repeat(REPLICATES, 1), y.repeat(REPLICATES)
    states = []
    for step in range(max(checkpoints) + 1):
        if step in checkpoints:
            states.append(dict(step=step, mu=model.mu.detach().clone(), rho=model.rho.detach().clone()))
        if step == max(checkpoints):
            break
        optimizer.zero_grad(set_to_none=True)
        empirical = bounded(gibbs_logits(model, xr, bank, source, channel, weight_generator, channel_generator),
                            yr, config).mean()
        loss = empirical
        if weight:
            terms, _ = complexity(kind, model, config, prior, architectures)
            loss = loss + weight * sum(terms.values())
        loss.backward()
        optimizer.step()
    return states, split


def load_state(config, state):
    import torch

    model, bank, train, test, split = build(config, .1)
    with torch.no_grad():
        model.mu.copy_(state["mu"])
        model.rho.copy_(state["rho"])
    model.requires_grad_(False)
    return model, bank, train, test, split


def mc_empirical(config, state, seed, observations, source):
    """Monte Carlo (bounded loss, accuracy) on S: uniform example, posterior draw, channel."""
    import torch

    model, bank, (x, y), _, _ = load_state(config, state)
    channel = config["train_channel"] if source == "deployment" else None
    generator, weight_generator, channel_generator = (torch.Generator().manual_seed(seed + i) for i in range(3))
    total, correct, done = 0., 0, 0
    with torch.no_grad():
        while done < observations:
            count = min(50_000, observations - done)
            index = torch.randint(len(y), (count,), generator=generator)
            logits = gibbs_logits(model, x[index], bank, source, channel, weight_generator, channel_generator)
            total += bounded(logits, y[index], config).sum().item()
            correct += (logits.argmax(-1) == y[index]).sum().item()
            done += count
    return total / observations, correct / observations


def heldout(config, state, which, replicates, seed):
    """Gibbs loss and accuracy under the deployment law on the validation or test split."""
    import torch

    model, bank, _, test, _ = load_state(config, state)
    x, y = model.validation if which == "validation" else test
    weight_generator, channel_generator = (torch.Generator().manual_seed(seed + i) for i in range(2))
    loss, correct = 0., 0
    with torch.no_grad():
        for _ in range(replicates):
            logits = gibbs_logits(model, x, bank, "deployment", config["deployment_channel"],
                                  weight_generator, channel_generator)
            loss += bounded(logits, y, config).sum().item()
            correct += (logits.argmax(-1) == y).sum().item()
    total = replicates * len(y)
    return dict(bounded_loss=loss / total, accuracy=correct / total, examples=len(y), replicates=replicates,
                channel_source="deployment", posterior_rule="independent weight and channel draw per example")


def population(config, state):
    return heldout(config, state, "test", TEST_REPLICATES, TEST_SEED)


def score(config, kind, state, prior, seed, observations, architectures):
    model, *_ = load_state(config, state)
    source = "perfect" if config["train_channel"]["kind"] == "perfect" else "bank"
    empirical, accuracy = mc_empirical(config, state, seed, observations, source)
    terms, extra = complexity(kind, model, config, prior, architectures)
    radius = math.sqrt(math.log(1 / MC_FAILURE) / (2 * observations))
    shift = shift_term(config, kind, model)
    components = dict(empirical=empirical, mc=radius, **{k: float(v) for k, v in terms.items()}, shift=shift["upper"])
    extra = {k: (float(v) if hasattr(v, "item") else v) for k, v in extra.items()}
    return dict(components=components, upper=sum(components.values()), shift=shift, train_accuracy=accuracy, **extra)


def select(candidates, rule, value="upper", accuracy="accuracy"):
    """Smallest value among candidates within the rule's accuracy tolerance of the best."""
    if TOLERANCES[rule] is not None:
        best = max(c[accuracy] for c in candidates)
        candidates = [c for c in candidates if c[accuracy] >= best - TOLERANCES[rule]]
    return min(candidates, key=lambda c: c[value])


def grid(config, kind, seed, architectures):
    candidates = []
    for init_std, lr, weight, cap in itertools.product(INIT_STDS, LEARNING_RATES, PENALTY_WEIGHTS, TRAIN_CAP_SCALES):
        states, _ = fit(config, kind, init_std, lr, weight, CHECKPOINTS, seed, architectures=architectures,
                        cap_scale=cap)
        candidates += [dict(label=f"init{init_std:g}-lr{lr:g}-w{weight:g}-cap{cap:g}-step{s['step']}", state=s)
                       for s in states]
    return candidates


# ----------------------------------------------------------------------------- stage 1 (Q_{W|S})

def stage1(config, architectures):
    """Fresh-channel priors trained on S with new P_tr draws; both rules use S-only criteria."""
    from torch.nn import functional as F

    candidates = grid(config, "stage1", config["seed"], architectures)
    for c in candidates:
        model, *_ = load_state(config, c["state"])
        c["fresh_empirical"], c["accuracy"] = mc_empirical(config, c["state"], STAGE1_MC_SEED,
                                                           SELECTION_OBSERVATIONS, "deployment")
        terms, extra = complexity("stage1", model, config, None, architectures)
        c["criterion"], c["kl"] = c["fresh_empirical"] + float(terms["sample"]), float(extra["kl"])
    priors = {}
    for rule in RULES:
        chosen = select(candidates, rule, value="criterion")
        prior = dict(mu=chosen["state"]["mu"], std=F.softplus(chosen["state"]["rho"]))
        record = {k: chosen[k] for k in ("label", "criterion", "fresh_empirical", "accuracy", "kl")}
        record.update(sigma_mean=float(prior["std"].mean()),
                      selection="S only: fresh-channel empirical + sample term"
                                + (f", within {TOLERANCES[rule]} of the best S accuracy" if TOLERANCES[rule] else ""))
        priors[rule] = (prior, chosen["state"], record)
    return priors


# ----------------------------------------------------------------------------- one case

def evaluate_candidates(config, kind, candidates, prior, architectures):
    for c in candidates:
        result = score(config, kind, c["state"], prior, SELECTION_MC_SEED, SELECTION_OBSERVATIONS, architectures)
        c["upper"], c["empirical"] = result["upper"], result["components"]["empirical"]
        c["accuracy"] = heldout(config, c["state"], "validation", VALIDATION_REPLICATES, config["eval_seed"])["accuracy"]
    return candidates


def run_case(args):
    family, case, output, receiver_norm, architectures = args
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    kind = certificate_kind(case)
    config = case_config(family, case, receiver_norm)
    old = archived(family, case)
    fresh_old = json.loads((FRESH_ARCHIVE / family / case / "bound.json").read_text())
    if kind == "T1":
        priors = stage1(config, architectures)
    else:
        shared = evaluate_candidates(config, kind, grid(config, kind, config["seed"], architectures), None,
                                     architectures)
    summary = []
    for rule in RULES:
        prior = prior_record = None
        if kind == "T1":
            prior, prior_state, prior_record = priors[rule]
            candidates = []
            for lr, weight, cap in itertools.product(STAGE2_LEARNING_RATES, STAGE2_PENALTY_WEIGHTS,
                                                     TRAIN_CAP_SCALES):
                states, _ = fit(config, "T1", .1, lr, weight, STAGE2_CHECKPOINTS, config["seed"] + 1,
                                prior=prior, start=prior_state, architectures=architectures, cap_scale=cap)
                candidates += [dict(label=f"stage2-lr{lr:g}-w{weight:g}-cap{cap:g}-step{s['step']}", state=s)
                               for s in states]
            candidates = evaluate_candidates(config, kind, candidates, prior, architectures)
        else:
            candidates = shared
        chosen = select(candidates, rule)
        result = score(config, kind, chosen["state"], prior, CERTIFICATE_MC_SEED, CERTIFICATE_OBSERVATIONS,
                       architectures)
        test = population(config, chosen["state"])
        ranked = sorted(candidates, key=lambda c: c["upper"])
        record = dict(
            dataset="moons", family=family, regime="perfect" if case == "perfect" else case.split("_")[-1],
            certificate=kind, theory={"C1": "Corollary 1", "T1": "Theorem 1 with fresh data-dependent Q_{W|S}",
                                      "T2": "Theorem 2"}[kind],
            config=config, posterior_kl=result.get("kl"),
            posterior=dict(objective="empirical bounded loss + weight x exact certificate complexity",
                           learned_std=True, label=chosen["label"],
                           std_mean=float(F.softplus(chosen["state"]["rho"]).mean()),
                           std_min=float(F.softplus(chosen["state"]["rho"]).min()),
                           std_max=float(F.softplus(chosen["state"]["rho"]).max()),
                           mean_norm=float(chosen["state"]["mu"].norm())),
            confidence=dict(family_failure=.05, family_size=20, architectures=architectures,
                            **failures(kind, architectures), mc=MC_FAILURE),
            empirical=dict(mean=result["components"]["empirical"], iid_observations=CERTIFICATE_OBSERVATIONS,
                           seed=CERTIFICATE_MC_SEED, train_accuracy=result["train_accuracy"]),
            empirical_mc_radius=result["components"]["mc"], components=result["components"],
            channel_shift=result["shift"], uncapped_risk_upper=result["upper"], nonvacuous=result["upper"] < 1,
            gap_to_population=result["upper"] - test["bounded_loss"], test=test,
            selection=dict(rule=rule, description=(
                "smallest bound" if TOLERANCES[rule] is None else
                f"smallest bound among candidates within {TOLERANCES[rule]} of the best validation accuracy")
                + " under an independent Monte Carlo seed", selection_mc_seed=SELECTION_MC_SEED,
                validation_accuracy=chosen["accuracy"], best_validation_accuracy=max(c["accuracy"] for c in candidates),
                candidates=len(candidates),
                leaderboard=[dict(label=c["label"], upper=c["upper"], empirical=c["empirical"],
                                  validation_accuracy=c["accuracy"]) for c in ranked[:10]],
                frontier=[dict(label=c["label"], upper=c["upper"], validation_accuracy=c["accuracy"])
                          for c in ranked if c["accuracy"] >= max(d["accuracy"] for d in ranked
                                                                  if d["upper"] <= c["upper"]) - 1e-12]),
            previous=dict(archived_upper=old["uncapped_risk_upper"], archived_test=old["test"]["bounded_loss"],
                          archived_accuracy=old["test"]["accuracy"],
                          only_fresh_upper=fresh_old["uncapped_risk_upper"],
                          only_fresh_test=fresh_old["test"]["bounded_loss"],
                          only_fresh_accuracy=fresh_old["test"]["accuracy"]))
        if kind == "T1":
            record["channel_prior"] = dict(construction="fresh", family_size=len(PRIOR_SCALES),
                                           scales=list(PRIOR_SCALES), selected_scale=result["prior_scale"],
                                           family=result["channel_family"], stage1=prior_record,
                                           union_bound="eps_1 / family_size")
        target = output / rule / family / case
        target.mkdir(parents=True, exist_ok=True)
        (target / "bound.json").write_text(json.dumps(record, indent=1) + "\n")
        torch.save(chosen["state"], target / "posterior.pt")
        if prior is not None:
            torch.save(dict(mu=prior["mu"], std=prior["std"]), target / "stage1_prior.pt")
        summary.append(dict(rule=rule, case=f"{family}/{case}", upper=result["upper"], test=test["bounded_loss"],
                            accuracy=test["accuracy"], gap=record["gap_to_population"],
                            components=result["components"], label=chosen["label"],
                            old=old["uncapped_risk_upper"], old_accuracy=old["test"]["accuracy"],
                            only_fresh=fresh_old["uncapped_risk_upper"],
                            only_fresh_accuracy=fresh_old["test"]["accuracy"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=PUBLISH / "bounds")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--receiver-norm", type=float, default=None,
                        help="override the archived receiver cap L (an exploration)")
    parser.add_argument("--architectures", type=int, default=1,
                        help="size of the declared architecture grid the result is selected from")
    parser.add_argument("--cases", nargs="*", default=None, help="family/case pairs, e.g. fading/t1_matched")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quick", action="store_true", help="tiny grid and Monte Carlo budget (smoke test)")
    args = parser.parse_args()
    if args.quick:
        global INIT_STDS, LEARNING_RATES, CHECKPOINTS, PENALTY_WEIGHTS, STAGE2_LEARNING_RATES
        global STAGE2_CHECKPOINTS, STAGE2_PENALTY_WEIGHTS, SELECTION_OBSERVATIONS, CERTIFICATE_OBSERVATIONS
        global TEST_REPLICATES, VALIDATION_REPLICATES, TRAIN_CAP_SCALES
        TRAIN_CAP_SCALES = (1., 8.)
        INIT_STDS, LEARNING_RATES, CHECKPOINTS, PENALTY_WEIGHTS = (.03,), (.01,), (50, 300), (1., 0.)
        STAGE2_LEARNING_RATES, STAGE2_CHECKPOINTS, STAGE2_PENALTY_WEIGHTS = (.003,), (0, 50), (1.,)
        SELECTION_OBSERVATIONS, CERTIFICATE_OBSERVATIONS, TEST_REPLICATES, VALIDATION_REPLICATES = \
            20_000, 50_000, 3, 5
    pairs = [tuple(c.split("/")) for c in args.cases] if args.cases else [(f, c) for f in FAMILIES for c in CASES]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(f, c, args.output, args.receiver_norm, args.architectures) for f, c in pairs]
    with ProcessPoolExecutor(args.workers) as pool:
        summary = [row for rows in pool.map(run_case, jobs) for row in rows]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for row in sorted(summary, key=lambda r: r["rule"]):
        print(f"{row['rule']:8s} {row['case']:20s} bound {row['upper']:.4f} (archived {row['old']:.4f}, "
              f"only_fresh {row['only_fresh']:.4f})  population {row['test']:.4f}  gap {row['gap']:.4f}  "
              f"acc {row['accuracy']:.3f} (only_fresh {row['only_fresh_accuracy']:.3f})  {row['label']}", flush=True)
    if args.plot:
        for rule in RULES:
            plot(args.output / rule)


def plot(bounds: Path):
    from results.final_chosen_results import plot_bound_decomposition_d8 as plotting

    plotting.BOUNDS = bounds
    plotting.YMAX = dict(fading=.65, erasure=.50)
    plotting.STEM = bounds.parent / f"bound_decomposition_d8_t1_t2_both_channels_{bounds.name}"
    plotting.main()


if __name__ == "__main__":
    main()
