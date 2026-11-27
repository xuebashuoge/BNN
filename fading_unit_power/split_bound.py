"""Two-moons (width 8, m = 2000) bound decomposition with the split-U prior for T1.

Same S, U, seeds, channels, Q_W = N(0, 0.24^2 I), loss and confidence allocation as
the archived figure, with three changes.

1. T1 uses Q_{W|S,U_1}. The bank is split by index into U_1 = U[:m_1] and
   U_2 = U[m_1:]. Stage 1 is a PAC-Bayes fit (means and standard deviations) on
   S x U_1; its hyperparameters are chosen by an (S, U_1)-only criterion. Then
   Q_{W|S,U_1} = N(mu, (c sigma)^2) with c in PRIOR_SCALES. Conditional on
   (S, U_1), U_2 is an i.i.d. sample of m_2 = m - m_1 channels, so Theorem 1
   holds with the empirical risk on S x U_2 and the channel term over m_2. The
   split fractions and scales are a finite family fixed in advance:
   K = len(SPLITS) * len(PRIOR_SCALES).
2. The receiver cap L is chosen from RECEIVER_NORMS instead of fixed at 0.075.
   At L = 0.075 every logit is below about 0.08, so every predictor's bounded
   loss is within a few hundredths of chance (0.148) whatever its accuracy.
   Every certificate (C1, T1, T2) divides its failure probability by
   G (below), so any L may be selected after the fact.
3. The feature cap F is chosen from FEATURE_NORMS. The archived F = 1 over
   width 8 gives each coordinate power at most 1/8; with unit-variance fading
   noise that is about -8 dB, and no two-moons fading predictor exceeds about
   58% accuracy, so every bounded loss stays near chance. F = sqrt(width) keeps
   each coordinate's power at most 1. The union bound covers every (F, L) pair:
   G = len(FEATURE_NORMS) * len(RECEIVER_NORMS).
4. Selection requires accuracy. A rule picks the smallest certified bound among
   posteriors whose validation accuracy is within a tolerance of the best one,
   using one Monte Carlo seed, then certifies with an independent seed. The
   ``tightest`` rule (no accuracy constraint) is reported for reference only.

C1 (perfect training) and T2 use the archived Corollary 1 and Theorem 2 with the
same posterior training as tighter_bound.py, at each L.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from . import tighter_bound as tb

PUBLISH = tb.REPO / "results" / "final_chosen_results" / "split_u" / "bounds"
RECEIVER_NORMS = (.075, .25, .5, 1., 2., 4.)
WIDTH = 8
FEATURE_NORMS = (1., math.sqrt(WIDTH))  # per-coordinate power <= 1/8 (archived) or <= 1
SPLITS = (.1, .25, .5)                 # m_1 / m
PRIOR_SCALES = tb.PRIOR_SCALES         # c in Q = N(mu, (c sigma)^2)
PRIOR_COUNT = len(SPLITS) * len(PRIOR_SCALES)
ARCHITECTURES = len(RECEIVER_NORMS) * len(FEATURE_NORMS)   # G
INIT_STDS, LEARNING_RATES, PENALTY_WEIGHTS = tb.INIT_STDS, tb.LEARNING_RATES, tb.PENALTY_WEIGHTS
CHECKPOINTS = tb.CHECKPOINTS
STAGE2_LEARNING_RATES, STAGE2_PENALTY_WEIGHTS = tb.STAGE2_LEARNING_RATES, tb.STAGE2_PENALTY_WEIGHTS
STAGE2_CHECKPOINTS = tb.STAGE2_CHECKPOINTS
STAGE2_PARTS = ("all", "second")       # posterior trains on U or on U_2
RULES = ("balanced", "accurate", "tightest")
TOLERANCES = tb.TOLERANCES            # tightest: None, balanced: 0.03, accurate: 0.01


def chance_loss(config):
    eps = config["loss_epsilon"]
    return math.log(eps + (1 - eps) / 2) / math.log(eps)


def case_config(family, case, receiver_norm, feature_norm):
    config = tb.case_config(family, case, receiver_norm)
    if config["width"] != WIDTH:
        raise ValueError("FEATURE_NORMS assume width 8")
    return {**config, "bounded_feature_norm": feature_norm}


def m_cut(config, fraction):
    return max(2, round(fraction * config["m"]))


def part_config(config, cut, part):
    return {**config, "bank_split": cut, "bank_part": part}


# ----------------------------------------------------------------------------- training

def fit(config, kind, init_std, lr, weight, checkpoints, seed, *, source, prior=None, start=None, m2=None):
    """tighter_bound.fit with an explicit channel source and the split-U T1 penalty."""
    import torch

    model, bank, (x, y), _, _ = tb.build(config, init_std)
    if start is not None:
        with torch.no_grad():
            model.mu.copy_(start["mu"])
            model.rho.copy_(start["rho"])
    optimizer = torch.optim.Adam([model.mu, model.rho], lr=lr)
    weight_generator = torch.Generator().manual_seed(seed)
    channel_generator = torch.Generator().manual_seed(seed + 1)
    xr, yr = x.repeat(tb.REPLICATES, 1), y.repeat(tb.REPLICATES)
    states = []
    for step in range(max(checkpoints) + 1):
        if step in checkpoints:
            states.append(dict(step=step, mu=model.mu.detach().clone(), rho=model.rho.detach().clone()))
        if step == max(checkpoints):
            break
        optimizer.zero_grad(set_to_none=True)
        loss = tb.bounded(tb.gibbs_logits(model, xr, bank, source, None, weight_generator, channel_generator),
                          yr, config).mean()
        if weight:
            terms = (t1_terms(model, config, prior, m2)[0] if kind == "T1" else
                     tb.complexity(kind, model, config, None, ARCHITECTURES)[0])
            loss = loss + weight * sum(terms.values())
        loss.backward()
        optimizer.step()
    return states


def t1_terms(model, config, prior, m2):
    """Theorem 1 complexities with Q_W (sample) and the split-U family over U_2 (channel)."""
    fail = tb.failures("T1", ARCHITECTURES)
    kl = tb.kl_diag(model.mu, model.rho, 0., config["prior_std"])
    channel_kls = [tb.kl_diag(model.mu, model.rho, prior["mu"], scale * prior["std"]) for scale in PRIOR_SCALES]
    channels = [tb.channel_term(value, m2, fail["eps_1"], PRIOR_COUNT) for value in channel_kls]
    best = min(range(len(channels)), key=lambda i: float(channels[i].detach()))
    return (dict(sample=tb.sample_term(kl, config["train_size"], fail["eps_2"]), channel=channels[best]),
            dict(kl=kl, channel_kl=channel_kls[best], prior_scale=PRIOR_SCALES[best],
                 channel_family=[dict(scale=s, kl=float(k), channel=float(c))
                                 for s, k, c in zip(PRIOR_SCALES, channel_kls, channels)]))


def score(config, kind, candidate, seed, observations, *, empirical=None, model=None):
    """Certified bound of one candidate; T1 uses S x U_2 and m_2.

    With ``empirical`` (a stored selection-seed estimate) no Monte Carlo is run; the
    complexity terms are recomputed at the current G. The shift is posterior-free,
    so ``model`` may be any model built from ``config``.
    """
    if model is None:
        model, *_ = tb.load_state(config, candidate["state"])
    state = candidate["state"]
    posterior = SimpleNamespace(mu=state["mu"], rho=state["rho"])
    radius = math.sqrt(math.log(1 / tb.MC_FAILURE) / (2 * observations))
    accuracy = None
    if kind == "T1":
        cut = candidate["cut"]
        if empirical is None:
            empirical, accuracy = tb.mc_empirical(part_config(config, cut, "second"), state, seed,
                                                  observations, "bank")
        terms, extra = t1_terms(posterior, config, candidate["prior"], config["m"] - cut)
    else:
        source = "perfect" if config["train_channel"]["kind"] == "perfect" else "bank"
        if empirical is None:
            empirical, accuracy = tb.mc_empirical(config, state, seed, observations, source)
        terms, extra = tb.complexity(kind, posterior, config, None, ARCHITECTURES)
    shift = tb.shift_term(config, kind, model)
    components = dict(empirical=empirical, mc=radius, **{k: float(v) for k, v in terms.items()}, shift=shift["upper"])
    extra = {k: (float(v) if hasattr(v, "item") else v) for k, v in extra.items()}
    return dict(components=components, upper=sum(components.values()), shift=shift, train_accuracy=accuracy, **extra)


def evaluate(config, kind, candidates):
    for c in candidates:
        result = score(config, kind, c, tb.SELECTION_MC_SEED, tb.SELECTION_OBSERVATIONS)
        c["upper"], c["empirical"] = result["upper"], result["components"]["empirical"]
        c["accuracy"] = tb.heldout(config, c["state"], "validation", tb.VALIDATION_REPLICATES,
                                   config["eval_seed"])["accuracy"]
    return candidates


def grid_fits(config, kind, source, seed, **kwargs):
    candidates = []
    for init_std, lr, weight in itertools.product(INIT_STDS, LEARNING_RATES, PENALTY_WEIGHTS):
        for state in fit(config, kind, init_std, lr, weight, CHECKPOINTS, seed, source=source, **kwargs):
            candidates.append(dict(label=f"init{init_std:g}-lr{lr:g}-w{weight:g}-step{state['step']}", state=state))
    return candidates


# ----------------------------------------------------------------------------- work units

def unit_c1_t2(config, kind):
    source = "perfect" if config["train_channel"]["kind"] == "perfect" else "bank"
    return evaluate(config, kind, grid_fits(config, kind, source, config["seed"]))


def stage1_priors(config, cut):
    """Q_{W|S,U_1} per rule, chosen by an (S, U_1)-only criterion."""
    from torch.nn import functional as F

    first = part_config(config, cut, "first")
    candidates = grid_fits(first, "stage1", "bank", config["seed"])
    for c in candidates:
        model, *_ = tb.load_state(first, c["state"])
        c["u1_empirical"], c["accuracy"] = tb.mc_empirical(first, c["state"], tb.STAGE1_MC_SEED,
                                                           tb.SELECTION_OBSERVATIONS, "bank")
        terms, extra = tb.complexity("stage1", model, first, None, ARCHITECTURES)
        c["criterion"], c["kl"] = c["u1_empirical"] + float(terms["sample"]), float(extra["kl"])
    priors = {}
    for rule in RULES:
        chosen = tb.select(candidates, rule, value="criterion")
        priors[rule] = dict(mu=chosen["state"]["mu"], std=F.softplus(chosen["state"]["rho"]), state=chosen["state"],
                            record=dict(label=chosen["label"], criterion=chosen["criterion"],
                                        u1_empirical=chosen["u1_empirical"], u1_accuracy=chosen["accuracy"],
                                        kl=chosen["kl"], sigma_mean=float(F.softplus(chosen["state"]["rho"]).mean()),
                                        selection="(S, U_1) only: S x U_1 empirical + sample term"
                                        + (f", within {TOLERANCES[rule]} of the best S x U_1 accuracy"
                                           if TOLERANCES[rule] else "")))
    return priors


def unit_t1(config, fraction):
    cut = m_cut(config, fraction)
    priors = stage1_priors(config, cut)
    candidates, fitted = [], {}
    for rule in RULES:
        prior = priors[rule]
        key = prior["record"]["label"]
        if key not in fitted:
            fitted[key] = []
            for lr, weight, part in itertools.product(STAGE2_LEARNING_RATES, STAGE2_PENALTY_WEIGHTS, STAGE2_PARTS):
                states = fit(part_config(config, cut, part), "T1", .1, lr, weight, STAGE2_CHECKPOINTS,
                             config["seed"] + 1, source="bank", prior=prior, start=prior["state"],
                             m2=config["m"] - cut)
                fitted[key] += [dict(label=f"split{fraction:g}-{part}-lr{lr:g}-w{weight:g}-step{s['step']}",
                                     state=s, cut=cut, fraction=fraction, prior=dict(mu=prior["mu"], std=prior["std"]),
                                     prior_record=prior["record"]) for s in states]
            evaluate(config, "T1", fitted[key])
        candidates += [dict(c, prior_rule=rule) for c in fitted[key]]
    return candidates


def run_unit(args):
    family, case, receiver_norm, feature_norm, fraction, path = args
    import torch

    torch.set_num_threads(1)
    if path.exists():
        return str(path)
    config = case_config(family, case, receiver_norm, feature_norm)
    kind = tb.certificate_kind(case)
    candidates = unit_t1(config, fraction) if kind == "T1" else unit_c1_t2(config, kind)
    for c in candidates:
        c["receiver_norm"], c["feature_norm"] = receiver_norm, feature_norm
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(candidates, temporary)
    temporary.replace(path)
    return str(path)


def units(root: Path):
    jobs = []
    for family, case in itertools.product(tb.FAMILIES, tb.CASES):
        for feature_norm, receiver_norm in itertools.product(FEATURE_NORMS, RECEIVER_NORMS):
            fractions = SPLITS if tb.certificate_kind(case) == "T1" else (None,)
            for fraction in fractions:
                # F = 1 keeps the first run's file names, so those units are reused.
                name = (f"L{receiver_norm:g}" + ("" if feature_norm == 1. else f"-F{feature_norm:.4g}")
                        + ("" if fraction is None else f"-split{fraction:g}"))
                jobs.append((family, case, receiver_norm, feature_norm, fraction,
                             root / "units" / family / case / f"{name}.pt"))
    # T1 units are the longest; start them first.
    return sorted(jobs, key=lambda job: job[3] is None)


# ----------------------------------------------------------------------------- selection and certificate

def certify_case(args):
    family, case, root, output = args
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    kind = tb.certificate_kind(case)
    pool = []
    for path in sorted((root / "units" / family / case).glob("*.pt")):
        pool += torch.load(path, weights_only=False)
    # Units store selection scores at the G of their run; recompute every score at
    # the current G from the stored selection-seed empirical risk (no new draws).
    models = {}
    for c in pool:
        c.setdefault("feature_norm", 1.)
        key = (c["receiver_norm"], c["feature_norm"])
        config = case_config(family, case, *key)
        if key not in models:
            models[key] = tb.load_state(config, c["state"])[0]
        c["upper"] = score(config, kind, c, None, tb.SELECTION_OBSERVATIONS, empirical=c["empirical"],
                           model=models[key])["upper"]
    archived = tb.archived(family, case)
    tightest = json.loads((tb.PUBLISH / "bounds" / "tightest" / family / case / "bound.json").read_text())
    summary = []
    for rule in RULES:
        candidates = [c for c in pool if kind != "T1" or c["prior_rule"] == rule]
        chosen = tb.select(candidates, rule) if TOLERANCES[rule] is not None else min(candidates, key=lambda c: c["upper"])
        config = case_config(family, case, chosen["receiver_norm"], chosen["feature_norm"])
        result = score(config, kind, chosen, tb.CERTIFICATE_MC_SEED, tb.CERTIFICATE_OBSERVATIONS)
        test = tb.population(config, chosen["state"])
        chance = chance_loss(config)
        ranked = sorted(candidates, key=lambda c: c["upper"])
        record = dict(
            dataset="moons", family=family, regime="perfect" if case == "perfect" else case.split("_")[-1],
            certificate=kind, theory={"C1": "Corollary 1", "T1": "Theorem 1 with split-U prior Q_{W|S,U_1}",
                                      "T2": "Theorem 2"}[kind],
            config=config, receiver_norm=chosen["receiver_norm"], feature_norm=chosen["feature_norm"],
            feature_power_per_coordinate_max=chosen["feature_norm"] ** 2 / WIDTH, posterior_kl=result.get("kl"),
            posterior=dict(objective="empirical bounded loss + weight x exact certificate complexity",
                           learned_std=True, label=chosen["label"],
                           std_mean=float(F.softplus(chosen["state"]["rho"]).mean()),
                           mean_norm=float(chosen["state"]["mu"].norm())),
            confidence=dict(family_failure=.05, family_size=20, architectures=ARCHITECTURES,
                            receiver_norms=list(RECEIVER_NORMS), feature_norms=list(FEATURE_NORMS),
                            **tb.failures(kind, ARCHITECTURES), mc=tb.MC_FAILURE),
            empirical=dict(mean=result["components"]["empirical"], iid_observations=tb.CERTIFICATE_OBSERVATIONS,
                           seed=tb.CERTIFICATE_MC_SEED, train_accuracy=result["train_accuracy"],
                           channels="S x U_2" if kind == "T1" else "S x U" if case != "perfect" else "S, perfect"),
            empirical_mc_radius=result["components"]["mc"], components=result["components"],
            channel_shift=result["shift"], uncapped_risk_upper=result["upper"], nonvacuous=result["upper"] < 1,
            chance_loss=chance, better_than_chance_population=test["bounded_loss"] < chance,
            gap_to_population=result["upper"] - test["bounded_loss"], test=test,
            selection=dict(rule=rule, tolerance=TOLERANCES[rule], selection_mc_seed=tb.SELECTION_MC_SEED,
                           validation_accuracy=chosen["accuracy"],
                           best_validation_accuracy=max(c["accuracy"] for c in candidates),
                           candidates=len(candidates),
                           leaderboard=[dict(label=c["label"], receiver_norm=c["receiver_norm"],
                                             feature_norm=c["feature_norm"], upper=c["upper"],
                                             validation_accuracy=c["accuracy"]) for c in ranked[:10]]),
            previous=dict(archived_upper=archived["uncapped_risk_upper"], archived_accuracy=archived["test"]["accuracy"],
                          tighter_bound_tightest_upper=tightest["uncapped_risk_upper"],
                          tighter_bound_tightest_accuracy=tightest["test"]["accuracy"]))
        if kind == "T1":
            record["channel_prior"] = dict(construction="split_U", split_fraction=chosen["fraction"],
                                           m_1=chosen["cut"], m_2=config["m"] - chosen["cut"],
                                           splits=list(SPLITS), scales=list(PRIOR_SCALES), family_size=PRIOR_COUNT,
                                           selected_scale=result["prior_scale"], family=result["channel_family"],
                                           stage1=chosen["prior_record"], union_bound="eps_1 / family_size")
        target = output / rule / family / case
        target.mkdir(parents=True, exist_ok=True)
        (target / "bound.json").write_text(json.dumps(record, indent=1) + "\n")
        torch.save(chosen["state"], target / "posterior.pt")
        summary.append(dict(rule=rule, case=f"{family}/{case}", upper=result["upper"], test=test["bounded_loss"],
                            chance=chance, accuracy=test["accuracy"], receiver_norm=chosen["receiver_norm"],
                            feature_norm=chosen["feature_norm"],
                            label=chosen["label"], components=result["components"],
                            archived=archived["uncapped_risk_upper"], tightest_fresh=tightest["uncapped_risk_upper"]))
    return summary


def plot(bounds: Path):
    from results.final_chosen_results import plot_bound_decomposition_d8 as plotting

    tops = {}
    for family in tb.FAMILIES:
        uppers = [json.loads((bounds / family / case / "bound.json").read_text())["uncapped_risk_upper"]
                  for case in tb.CASES]
        tops[family] = max(.5, math.ceil(max(uppers) * 10 + .5) / 10)
    plotting.BOUNDS, plotting.YMAX = bounds, tops
    plotting.STEM = bounds.parent / f"bound_decomposition_d8_t1_t2_both_channels_{bounds.name}"
    plotting.main()


def pipeline(root: Path, workers: int, output: Path = PUBLISH):
    root.mkdir(parents=True, exist_ok=True)
    jobs = units(root)
    with ProcessPoolExecutor(workers) as pool:
        for done in pool.map(run_unit, jobs):
            print(json.dumps(dict(event="unit", path=done)), flush=True)
        cases = [(f, c, root, output) for f in tb.FAMILIES for c in tb.CASES]
        summary = [row for rows in pool.map(certify_case, cases) for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for row in sorted(summary, key=lambda r: (r["rule"], r["case"])):
        print(f"{row['rule']:8s} {row['case']:20s} F={row['feature_norm']:.3g} L={row['receiver_norm']:<5g} bound {row['upper']:.4f} "
              f"population {row['test']:.4f} (chance {row['chance']:.4f}) acc {row['accuracy']:.3f}  "
              f"[archived {row['archived']:.4f}, tightest-fresh {row['tightest_fresh']:.4f}]  {row['label']}",
              flush=True)
    for rule in RULES:
        plot(output / rule)


def quick():
    """Tiny grids and Monte Carlo budgets for a smoke test."""
    global INIT_STDS, LEARNING_RATES, PENALTY_WEIGHTS, CHECKPOINTS, STAGE2_LEARNING_RATES
    global STAGE2_PENALTY_WEIGHTS, STAGE2_CHECKPOINTS, RECEIVER_NORMS, SPLITS, PRIOR_COUNT, ARCHITECTURES
    INIT_STDS, LEARNING_RATES, PENALTY_WEIGHTS, CHECKPOINTS = (.03,), (.01,), (1., 0.), (50, 300)
    STAGE2_LEARNING_RATES, STAGE2_PENALTY_WEIGHTS, STAGE2_CHECKPOINTS = (.003,), (1.,), (0, 50)
    RECEIVER_NORMS, SPLITS = (.075, 1.), (.25,)
    PRIOR_COUNT, ARCHITECTURES = len(SPLITS) * len(PRIOR_SCALES), len(RECEIVER_NORMS) * len(FEATURE_NORMS)
    tb.SELECTION_OBSERVATIONS, tb.CERTIFICATE_OBSERVATIONS = 20_000, 50_000
    tb.TEST_REPLICATES, tb.VALIDATION_REPLICATES = 3, 5


# Worker processes may be spawned, so the smoke-test switch travels in the environment.
if os.environ.get("SPLIT_BOUND_QUICK") == "1":
    quick()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PUBLISH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick and os.environ.get("SPLIT_BOUND_QUICK") != "1":
        os.environ["SPLIT_BOUND_QUICK"] = "1"
        quick()
    pipeline(args.root.resolve(), args.workers, args.output)


if __name__ == "__main__":
    main()
