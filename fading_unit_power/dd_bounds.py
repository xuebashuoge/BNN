"""Two-moons (width 8, m=2000) T1 certificates with a data-dependent channel prior.

The revised Theorem 1 holds, with probability at least (1-eps_1)(1-eps_2) over
(U, S), simultaneously for every posterior. Its channel complexity uses a prior
Q_{W|S} independent of U and the sample complexity a prior Q_W independent of
(U, S). The certificate here keeps the archived confidence allocation: the
family of 20 statements at delta=0.05 gives 0.00125 to each theorem and to each
Monte Carlo estimate of the empirical risk, and eps_1 = eps_2 = 0.000625.

Q_{W|S} ranges over a finite family fixed before U is used: the three stage-1
means mu_S (perfect, independent bank, fresh draws), each with the isotropic
scales in CERTIFICATE_STD_MULTIPLIERS times the posterior scale. A union bound
over the K members replaces eps_1 by eps_1/K, so the certificate may take the
smallest channel complexity over the family. Q_W stays N(0, prior_std^2 I).

Posterior selection is free because the bound holds for all posteriors. The
candidate with the smallest bound is chosen with one Monte Carlo estimate and
certified with an independent one. Fading, erasure, matched and mismatched
training reuse the archived configurations, data, banks and seeds of the
fixed-prior T1 fits. Perfect training and T2 cases are copied unchanged.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import shutil

from . import data_dependent as campaign

OLD_BOUNDS = campaign.SELECTED / "data" / "bounds"
CASES = [(family, regime) for family in ("fading", "erasure") for regime in ("matched", "mismatch")]
TRAIN_STD_MULTIPLIERS = (1., 1.5, 2., 3.)
CERTIFICATE_STD_MULTIPLIERS = (1., 1.25, 1.5, 2., 3., 5.)
ETA_CHANNEL_MULTIPLIERS = (1., 10., 100.)
ETA_DATA_MULTIPLIERS = (1., 5.)
THEOREM_FAILURE = .00125
MC_FAILURE = .00125
MC_OBSERVATIONS = 20_000
SELECTION_MC_SEED = 271_828
CERTIFICATE_MC_SEED = 314_159
SIGMA_SQ = .25


def base_config(family, regime):
    record = campaign.read_json(OLD_BOUNDS / family / f"t1_{regime}" / "bound.json")
    return record["config"], record


def plan(root: Path):
    tasks = []
    for family, regime in CASES:
        config, _ = base_config(family, regime)
        t1 = campaign.new_run_config(config)
        test = dict(name="test", split="test", scope="deployment",
                    overrides=dict(eval_draws=20, eval_seed=config["eval_seed"], device="cpu", threads=1))
        grid = itertools.product(campaign.CONSTRUCTIONS, TRAIN_STD_MULTIPLIERS, ETA_CHANNEL_MULTIPLIERS,
                                 ETA_DATA_MULTIPLIERS)
        for variant in grid:
            tasks.extend(campaign.t1_pair(f"bounds-{family}-{regime}", t1, variant, root, [test],
                                          dict(family=family, regime=regime),
                                          len(campaign.CONSTRUCTIONS) * len(CERTIFICATE_STD_MULTIPLIERS)))
        # Stage 1 for every construction, so the certificate family is complete.
        for construction in campaign.CONSTRUCTIONS:
            stage1, _ = campaign.t1_pair(f"bounds-{family}-{regime}", t1, (construction, 1., 1.), root, [], {}, 1)
            tasks.append(stage1)
    return campaign.dedupe(tasks)


def channel_shift(family, regime, old_record):
    from .trainer.diagonal_fading import diagonal_gain_kl

    config = old_record["config"]
    if regime == "matched":
        return dict(method="T1 channel KL", analytic=True, upper=0., channel_kl=0.)
    deployment, training, width = config["deployment_channel"], config["train_channel"], config["width"]
    if family == "fading":
        divergence = diagonal_gain_kl(deployment, training, width)
    else:
        p, q = deployment["outage_prob"], training["outage_prob"]
        divergence = width * (p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))
    if not math.isclose(divergence, old_record["channel_shift"]["channel_kl"], rel_tol=1e-9):
        raise ValueError(f"Channel KL differs from the archived {family}/{regime} value")
    return dict(method="T1 channel KL", analytic=True, upper=math.sqrt(2 * SIGMA_SQ * divergence),
                channel_kl=divergence)


def certificate(root: Path, task: dict, stage1_dirs: dict, seed: int):
    import torch
    from torch.nn import functional as F
    from .trainer import run_experiment as trainer
    from .trainer.theorem1 import gaussian_kl, gaussian_kl_to_prior

    checkpoint = campaign.run_dir(root, task["task_id"])
    metadata = campaign.read_json(checkpoint / "metadata.json")
    config = trainer.normalize_config(metadata["config"])
    torch.set_num_threads(1)
    train_data, _, _, input_dim, classes, split = trainer.load_data(config)
    model, bank = trainer.make_models(config, input_dim, classes)
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    if trainer.channel_bank_fingerprint(bank, metadata["training_channel_bank_hash_algorithm"]) != \
            metadata["training_channel_bank_hash"]:
        raise ValueError("Bank differs from its recorded hash")
    model.eval()
    x = torch.stack([train_data[i][0] for i in range(len(train_data))])
    y = torch.tensor([int(train_data[i][1]) for i in range(len(train_data))])
    generator = torch.Generator().manual_seed(seed)
    channel_generator = torch.Generator().manual_seed(seed + 1)
    weight_generator = torch.Generator().manual_seed(seed + 2)
    total, done = 0., 0
    with torch.no_grad():
        while done < MC_OBSERVATIONS:
            count = min(4096, MC_OBSERVATIONS - done)
            index = torch.randint(len(y), (count,), generator=generator)
            logits = model.forward_iid_weights(x[index], bank, weight_generator, channel_generator)
            losses = F.cross_entropy(logits, y[index], reduction="none")
            total += trainer.bounded_losses_from_ce(losses, config).sum().item()
            done += count
    empirical = total / MC_OBSERVATIONS
    radius = math.sqrt(math.log(1 / MC_FAILURE) / (2 * MC_OBSERVATIONS))
    n, m = config["train_size"], config["m"]
    eps = THEOREM_FAILURE / 2
    with torch.no_grad():
        kl = gaussian_kl(model.mu, model.rho, config["prior_std"]).item()
        family = []
        for construction, directory in stage1_dirs.items():
            prior_state = torch.load(Path(directory) / "weights.pt", map_location="cpu", weights_only=True)
            prior_metadata = campaign.read_json(Path(directory) / "metadata.json")
            for key in ("train_indices_hash", "validation_indices_hash"):
                if prior_metadata["split"][key] != split[key]:
                    raise ValueError("Stage-1 prior uses a different S")
            for multiplier in CERTIFICATE_STD_MULTIPLIERS:
                scale = config["posterior_std"] * multiplier
                family.append(dict(construction=construction, std=scale, stage1=str(directory),
                                   kl=gaussian_kl_to_prior(model.mu, model.rho, prior_state["model"]["mu"],
                                                           scale).item()))
    count = len(family)
    sample = math.sqrt(2 * SIGMA_SQ / (n - 1) * (kl + math.log(math.sqrt(n) / eps)))
    for member in family:
        member["channel"] = math.sqrt(2 * SIGMA_SQ / (m - 1) * (member["kl"] + math.log(math.sqrt(m) * count / eps)))
    best = min(family, key=lambda member: member["channel"])
    return dict(config=config, metadata=metadata, empirical=empirical, radius=radius, kl=kl, sample=sample,
                family=family, best=best, count=count, bank_hash=metadata["training_channel_bank_hash"])


def pipeline(root: Path, workers: int, constructions=None):
    """Certify T1; with ``constructions`` both the candidates and the prior family use only those.

    A restricted certificate has K = len(constructions) * len(CERTIFICATE_STD_MULTIPLIERS), so it
    is valid on its own only when the construction is fixed before U is used. Taking the minimum
    over separately restricted certificates would need the full-family union bound instead.
    """
    constructions = tuple(constructions or campaign.CONSTRUCTIONS)
    tasks_path = root / "manifests" / "tasks.jsonl"
    if not tasks_path.exists():
        campaign.write_manifest(tasks_path, plan(root))
    tasks = campaign.read_manifest(tasks_path)
    campaign.run_pool(tasks, root, workers)
    output = campaign.PUBLISH / "bounds"
    summary = []
    for family, regime in CASES:
        _, old = base_config(family, regime)
        shift = channel_shift(family, regime, old)
        prefix = f"bounds-{family}-{regime}"
        stage1_dirs = {t["meta"]["construction"]: campaign.run_dir(root, t["task_id"]) for t in tasks
                       if t["task_id"].startswith(prefix) and t["meta"]["role"] == "stage1"
                       and t["config"]["seed"] == old["config"]["seed"]
                       and t["meta"]["construction"] in constructions}
        candidates = [t for t in tasks if t["task_id"].startswith(prefix + "-T1")
                      and t["meta"]["variant"][0] in constructions]
        scored = []
        for task in candidates:
            result = certificate(root, task, stage1_dirs, SELECTION_MC_SEED)
            upper = result["empirical"] + result["radius"] + result["sample"] + result["best"]["channel"] + shift["upper"]
            scored.append((upper, task))
        scored.sort(key=lambda item: item[0])
        chosen = scored[0][1]
        result = certificate(root, chosen, stage1_dirs, CERTIFICATE_MC_SEED)
        test = campaign.read_json(campaign.run_dir(root, chosen["task_id"]) / "test" / "evaluation.json")
        components = dict(empirical=result["empirical"], mc=result["radius"], sample=result["sample"],
                          channel=result["best"]["channel"], shift=shift["upper"])
        upper = sum(components.values())
        record = dict(dataset="moons", family=family, regime=regime, certificate="T1", fallback=False,
                      config=result["config"],
                      checkpoint_weights=campaign.read_json(campaign.run_dir(root, chosen["task_id"]) / "done.json")["weights_sha256"],
                      saved_bank_hash=result["bank_hash"], selected_epoch=result["metadata"]["selected_epoch"],
                      empirical=dict(mean=result["empirical"], iid_observations=MC_OBSERVATIONS,
                                     examples_per_observation=1, total_evaluations=MC_OBSERVATIONS,
                                     seed=CERTIFICATE_MC_SEED),
                      test=test["heldout_gibbs_mc"], theory="Theorem 1 with data-dependent channel prior Q_{W|S}",
                      posterior_kl=result["kl"],
                      channel_prior=dict(selected=result["best"], family_size=result["count"],
                                         family=result["family"], union_bound="eps_1 / family_size"),
                      confidence=dict(family_failure=.05, family_size=20,
                                      failure_each_theorem_and_mc=THEOREM_FAILURE, eps_1=THEOREM_FAILURE / 2,
                                      eps_2=THEOREM_FAILURE / 2),
                      empirical_mc_radius=result["radius"], components=components, channel_shift=shift,
                      uncapped_risk_upper=upper, nonvacuous=upper < 1,
                      selection=dict(rule="smallest bound under an independent Monte Carlo seed",
                                     selection_mc_seed=SELECTION_MC_SEED, candidates=len(scored),
                                     task_id=chosen["task_id"], variant=chosen["meta"]["label"],
                                     leaderboard=[dict(task=t["task_id"], upper=u) for u, t in scored[:10]]),
                      previous_fixed_prior=dict(uncapped_risk_upper=old["uncapped_risk_upper"],
                                                components=old["components"]))
        campaign.write_json(output / family / f"t1_{regime}" / "bound.json", record)
        summary.append(dict(case=f"{family}/{regime}", constructions=list(constructions), new=upper, old=old["uncapped_risk_upper"],
                            channel_new=components["channel"], channel_old=old["components"]["channel"],
                            variant=chosen["meta"]["label"], test_bounded_loss=test["heldout_gibbs_mc"]["bounded_loss"]))
    for family in ("fading", "erasure"):
        for case in ("perfect", "t2_matched", "t2_mismatch"):
            target = output / family / case
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OLD_BOUNDS / family / case / "bound.json", target / "bound.json")
    campaign.write_json(output / "summary.json", summary)
    from results.final_chosen_results import plot_bound_decomposition_d8 as plotting
    plotting.BOUNDS = output
    plotting.STEM = campaign.PUBLISH / "bound_decomposition_d8_t1_t2_both_channels"
    plotting.main()
    print(json.dumps(summary, indent=1))
