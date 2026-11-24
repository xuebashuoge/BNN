"""Tune prior scale and posterior optimization at the validated channel-bank size."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics

from . import performance
from . import performance_sweep as primary


VALIDATION_SEEDS = (24112431, 24112432, 24112433)
CONFIRMATION_SEEDS = (24112441, 24112442, 24112443, 24112444, 24112445)
FIXED_BANK_VALIDATION_SEEDS = tuple(range(26112501, 26112507))
FIXED_BANK_CONFIRMATION_SEEDS = tuple(range(26112511, 26112516))
SCHEDULE_VALIDATION_SEEDS = tuple(range(26112521, 26112533))
SCHEDULE_CONFIRMATION_SEEDS = tuple(range(26112541, 26112546))
SCHEDULE_PROFILES = ("incumbent", "schedule-240-k2", "schedule-300-k1")
PROFILE_COUNT = 6
EVAL_DRAWS = {"search": 40, "confirmation": 64}
SEED_FIELDS = {"seed", "channel_seed", "eval_seed"}
MATCHED_FIELDS = (
    "dataset", "architecture", "width", "predictor", "train_size", "val_size", "data_seed", "batch_size",
    "train_channel", "deployment_channel", "transmit_unit_power", "training_loss", "loss_epsilon",
    "selection_metric", "moon_noise", "moons_sampler",
)


def plan_path(output, dataset):
    return output / "manifests" / f"{dataset}-regularization-plan.json"


def selection_path(output, dataset):
    return output / "manifests" / f"{dataset}-sweep-selection.json"


def seed_sets(fixed_bank, schedule_panel=False):
    if schedule_panel:
        return SCHEDULE_VALIDATION_SEEDS, SCHEDULE_CONFIRMATION_SEEDS
    if fixed_bank:
        return FIXED_BANK_VALIDATION_SEEDS, FIXED_BANK_CONFIRMATION_SEEDS
    return VALIDATION_SEEDS, CONFIRMATION_SEEDS


def count(phase, *, fixed_bank=False, schedule_panel=False):
    validation_seeds, confirmation_seeds = seed_sets(fixed_bank, schedule_panel)
    profiles_per_method = len(SCHEDULE_PROFILES) if schedule_panel else PROFILE_COUNT
    per_method = profiles_per_method * len(validation_seeds) if phase == "search" else len(confirmation_seeds)
    return len(primary.METHODS) * per_method


def search_design(planned):
    """Validate the bank, profile matrix and paired seed design recorded in a plan."""
    dataset, m = planned["dataset"], planned["bank_size"]
    fixed_bank = "fixed_bank_size" in planned
    schedule_panel = "schedule_panel" in planned
    if dataset not in primary.DATASETS or m not in primary.BANK_SIZES or planned.get("fixed_bank_size", m) != m:
        raise ValueError("Invalid regularization dataset or bank")
    if schedule_panel and (planned["schedule_panel"] is not True or not fixed_bank or (dataset, m) != ("moons", 2000)):
        raise ValueError("The schedule panel requires fixed m=2000 on Two moons")

    validation_seeds, confirmation_seeds = seed_sets(fixed_bank, schedule_panel)
    profiles_per_method = len(SCHEDULE_PROFILES) if schedule_panel else PROFILE_COUNT
    if (tuple(planned["validation_seeds"]), tuple(planned["confirmation_seeds"])) != (
            validation_seeds, confirmation_seeds):
        raise ValueError(f"Unexpected regularization seeds for {dataset}")
    profiles_by_method = planned["profiles"]
    if set(profiles_by_method) != set(primary.METHODS):
        raise ValueError(f"Incomplete regularization methods for {dataset}")
    for method, names in profiles_by_method.items():
        if (not isinstance(names, (list, tuple)) or any(not isinstance(name, str) or not name for name in names)
                or len(names) != profiles_per_method or len(set(names)) != len(names)):
            raise ValueError(f"Unexpected regularization profiles for {dataset}/{method}")
        if schedule_panel and tuple(names) != tuple(f"regularization-m{m}-{name}" for name in SCHEDULE_PROFILES):
            raise ValueError(f"Unexpected schedule profiles for {dataset}/{method}")
    return validation_seeds, confirmation_seeds, profiles_per_method


def primary_anchors(source, dataset, bank_size=None):
    """Replay the refined validation candidates and verify the frozen bank choice."""
    manifest = performance.manifest_path(source, dataset, "refine")
    scores_path = source / "manifests" / f"{dataset}-refine-scores.json"
    selected_path = selection_path(source, dataset)
    recorded_scores = performance.read_json(scores_path)
    selection = performance.read_json(selected_path)
    manifest_hash = primary.sha256(manifest)
    if (recorded_scores["dataset"], recorded_scores["manifest_sha256"], selection["dataset"]) != (
            dataset, manifest_hash, dataset):
        raise ValueError(f"Primary validation manifests differ for {dataset}")

    groups, scores = primary.candidates(source, dataset, "refine", primary.REFINE_SEEDS)
    if recorded_scores["bank_sizes"] != {
            str(m): {method: scores[m, method] for method in primary.METHODS} for m in primary.BANK_SIZES}:
        raise ValueError(f"Primary validation score records differ for {dataset}")

    leaderboard = {}
    for m in primary.BANK_SIZES:
        chosen = {method: primary.best(scores[m, method]) for method in primary.METHODS}
        winners = {method: scores[m, method][name] for method, name in chosen.items()}
        proposed = primary.best({method: winners[method] for method in primary.PROPOSED})
        baseline = primary.best({method: winners[method] for method in primary.BASELINES})
        proposed_accuracy = winners[proposed]["mean_accuracy"]
        gap = 100 * (proposed_accuracy - winners[baseline]["mean_accuracy"])
        expected = dict(chosen=chosen, proposed=proposed, baseline=baseline, gap_pp=gap,
                        proposed_accuracy=proposed_accuracy, scores=winners)
        saved = selection["validation_leaderboard"][str(m)]
        if (recorded_scores["winners"][str(m)] != chosen or
                any(saved[key] != value for key, value in expected.items())):
            raise ValueError(f"Primary validation winners differ for {dataset}/m={m}")
        leaderboard[m] = expected

    primary_m = min(leaderboard, key=lambda size: (-leaderboard[size]["gap_pp"], -leaderboard[size]["proposed_accuracy"], size))
    selected = leaderboard[primary_m]
    if (selection["chosen_bank_size"], selection["chosen_proposed"], selection["chosen_baseline"],
            selection["validation_gap_pp"]) != (primary_m, selected["proposed"], selected["baseline"], selected["gap_pp"]):
        raise ValueError(f"Primary bank selection differs for {dataset}")
    m = primary_m if bank_size is None else bank_size
    if m not in leaderboard:
        raise ValueError(f"No primary validation at m={m} for {dataset}")
    anchors = {method: groups[m, method][name][0] for method, name in leaderboard[m]["chosen"].items()}
    common = anchors[primary.METHODS[0]]["config"]
    for method, row in anchors.items():
        config = row["config"]
        if (config["method"], config["m"], config["predictor"], config["train_channel"],
                config["deployment_channel"], config["transmit_unit_power"]) != (
                primary.METHOD_OBJECTIVE[method], m, "gaussian", performance.FADING, performance.FADING, True):
            raise ValueError(f"Invalid matched anchor for {dataset}/{method}")
        if any(config.get(field) != common.get(field) for field in MATCHED_FIELDS):
            raise ValueError(f"Primary settings are unpaired for {dataset}/{method}")
    return m, anchors, dict(refine_manifest_sha256=manifest_hash, refine_scores_sha256=primary.sha256(scores_path),
                             validation_selection_sha256=primary.sha256(selected_path))


def initial_mean_geometry(config):
    """Use the documented fan-in uniform weight initialization and zero biases."""
    width = config["width"]
    dataset = config["dataset"]
    architecture = config["architecture"]
    if architecture == "mlp" and dataset in {"moons", "mnist", "fashionmnist"}:
        inputs, classes = (2, 2) if dataset == "moons" else (784, 10)
        layers = ((inputs, width), (width, classes))
    elif architecture == "cnn_spatial3" and dataset == "cifar10":
        latent = width // 16
        layers = ((3 * 9, 16), (16 * 9, 32), (32 * 9, 64), (64, latent),
                  (latent * 9, 64), (64 * 16, 128), (128, 10))
    else:
        raise ValueError(f"No initial KL calibration for {dataset}/{architecture}")
    dimension = sum((fan_in + 1) * fan_out for fan_in, fan_out in layers)
    mean_squared_norm = sum(fan_out for _, fan_out in layers) / 3
    return dimension, mean_squared_norm


def calibrated_penalty(config, method, prior_std, posterior_std, l2_strength, channel_share):
    """Match the initial KL penalty gradient on mu to 2*l2_strength*mu.

    For C=sqrt(A*(KL+c)), dC/dmu = A*mu/(2*C*prior_std**2). Substitute the
    expected initial KL under the model's random mean initialization. Channel
    and data shares distribute that gradient in T1. T2 has only a joint term.
    Calibration targets the initial mean gradient. Posterior-scale gradients
    and the subsequent trajectories differ from L2.
    """
    dimension, mean_squared_norm = initial_mean_geometry(config)
    prior_variance = prior_std**2
    ratio = posterior_std**2 / prior_variance
    kl = .5 * (dimension * (ratio - 1 - math.log(ratio)) + mean_squared_norm / prior_variance)
    n, m, delta = config["train_size"], config["m"], config["delta"]
    data_factor, channel_factor = .5 / (n - 1), .5 / (m - 1)

    def coefficient(factor, log_term, share):
        term = math.sqrt(factor * (kl + log_term))
        return 4 * l2_strength * prior_variance * term * share / factor

    if method == "T1-Matched":
        return dict(
            eta_data=coefficient(data_factor, math.log(math.sqrt(n) / (delta / 2)), 1 - channel_share),
            eta_channel=coefficient(channel_factor, math.log(math.sqrt(m) / (delta / 2)), channel_share),
        )
    return dict(eta_data=coefficient(data_factor + channel_factor, math.log(math.sqrt(n * m) / delta), 1.),
                eta_channel=0.)


def profiles(method, config, l2_anchor, hot_posterior, tight_prior):
    initial_posterior = config["posterior_std"]
    initial_rho_scale = config["rho_lr_scale"]
    hot = dict(posterior_std=hot_posterior, rho_lr_scale=1.)
    yield "incumbent", config, {}
    if method == "ERM-Matched":
        cold = initial_posterior / 3
        for name, update in (
                ("hot-learned", hot),
                ("hot-fixed", {**hot, "rho_lr_scale": 0.}),
                ("cool-learned", dict(posterior_std=cold, rho_lr_scale=1.)),
                ("cool-fixed", dict(posterior_std=cold, rho_lr_scale=0.)),
                ("current-fast-scale", dict(rho_lr_scale=1.))):
            yield name, {**config, **update}, {}
    elif method == "L2-Matched":
        for name, multiplier, scale_settings in (
                ("l2-weak", .1, {}),
                ("l2-strong", 10., {}),
                ("hot-current-l2", 1., hot),
                ("hot-weak-l2", .1, hot),
                ("hot-strong-l2", 10., hot)):
            yield name, {**config, **scale_settings, "l2": l2_anchor * multiplier}, {}
    else:
        for name, prior_std, posterior_std, rho_scale, multiplier, channel_share in (
                ("matched-shrinkage", config["prior_std"], initial_posterior, initial_rho_scale, 1., .5),
                ("tight-prior", tight_prior, initial_posterior, initial_rho_scale, 1., .5),
                ("tight-hot", tight_prior, hot_posterior, 1., 1., .5),
                ("tight-hot-weak", tight_prior, hot_posterior, 1., .1, .2),
                ("tight-hot-strong", tight_prior, hot_posterior, 1., 10., .8)):
            l2_target = l2_anchor * multiplier
            coefficients = calibrated_penalty(config, method, prior_std, posterior_std, l2_target, channel_share)
            settings = dict(prior_std=prior_std, posterior_std=posterior_std, rho_lr_scale=rho_scale)
            calibration = dict(initial_l2_target=l2_target,
                               initial_channel_gradient_share=channel_share if method == "T1-Matched" else None)
            yield name, {**config, **settings, **coefficients}, calibration


def task(phase, method, name, config, source, seed):
    draws = EVAL_DRAWS[phase]
    values = {**config, "seed": seed, "channel_seed": seed + 10_000,
              "eval_seed": seed + 20_000, "eval_draws": draws}
    evaluation = dict(eval_draws=draws, eval_seed=seed + 1_000_003)
    candidate = f"regularization-m{config['m']}-{name}"
    return performance.task_entry(phase, method, candidate, values, evaluation, source)


def schedule_anchors(source, dataset, bank_size):
    source_plan = plan_path(source, dataset)
    planned = performance.read_json(source_plan)
    if planned.get("fixed_bank_size") != bank_size or "schedule_panel" in planned:
        raise ValueError(f"Expected the fixed-bank regularization search for {dataset}")
    anchors, confirmation, replayed = replay_selection(source, dataset)
    selection = selection_path(source, dataset)
    confirmation_manifest = performance.manifest_path(source, dataset, "confirmation")
    replayed["confirmation_manifest_sha256"] = primary.sha256(confirmation_manifest)
    if (performance.read_json(selection) != replayed or
            performance.read_manifest(confirmation_manifest) != confirmation):
        raise ValueError(f"Prior regularization selection differs from validation replay for {dataset}")
    fingerprints = dict(
        regularization_search_manifest_sha256=replayed["search_manifest_sha256"],
        regularization_plan_sha256=replayed["search_plan_sha256"],
        regularization_selection_sha256=primary.sha256(selection),
    )
    return anchors, fingerprints


def plan_schedule_panel(output, source, dataset, bank_size):
    if (dataset, bank_size) != ("moons", 2000):
        raise ValueError("The schedule panel requires --dataset moons --bank-size 2000")
    anchors, fingerprints = schedule_anchors(source, dataset, bank_size)
    variants = (
        {},
        dict(epochs=240, lr=.003125, lr_decay=.20 ** (1 / 239), train_mc=2),
        dict(epochs=300, lr=.00625, lr_decay=.18 ** (1 / 299), train_mc=1),
    )
    rows, planned = [], {}
    common = anchors[primary.METHODS[0]]["config"]
    for method, anchor in anchors.items():
        base = anchor["config"]
        if ((base["method"], base["m"], base["predictor"], base["train_channel"],
                base["deployment_channel"], base["transmit_unit_power"]) != (
                primary.METHOD_OBJECTIVE[method], bank_size, "gaussian", performance.FADING, performance.FADING, True)
                or any(base.get(field) != common.get(field) for field in MATCHED_FIELDS)):
            raise ValueError(f"Unmatched prior regularization anchor for {dataset}/{method}")
        configs = [{**base, **update} for update in variants]
        if any(config in configs[:i] for i, config in enumerate(configs)):
            raise ValueError(f"Expected distinct schedules for {dataset}/{method}")
        planned[method] = []
        for name, config in zip(SCHEDULE_PROFILES, configs):
            provenance = dict(regularization_search_task=anchor["task_id"], profile=name, **fingerprints)
            entries = [task("search", method, name, config, provenance, seed) for seed in SCHEDULE_VALIDATION_SEEDS]
            planned[method].append(entries[0]["candidate"])
            rows.extend(entries)

    manifest = performance.manifest_path(output, dataset, "search")
    performance.write_manifest(manifest, rows)
    performance.write_json(plan_path(output, dataset), dict(
        dataset=dataset, bank_size=bank_size, source=str(source), source_fingerprints=fingerprints,
        anchor_tasks={method: row["task_id"] for method, row in anchors.items()},
        validation_seeds=SCHEDULE_VALIDATION_SEEDS, confirmation_seeds=SCHEDULE_CONFIRMATION_SEEDS,
        profiles=planned, search_manifest_sha256=primary.sha256(manifest),
        fixed_bank_size=bank_size, schedule_panel=True,
    ), frozen=True)
    print(f"{manifest} ({len(rows)} tasks), m={bank_size}")


def plan(output, source, dataset, *, bank_size=None, schedule_panel=False):
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Use a follow-up directory separate from the primary performance directory")
    if schedule_panel:
        plan_schedule_panel(output, source, dataset, bank_size)
        return
    m, anchors, fingerprints = primary_anchors(source, dataset, bank_size)
    validation_seeds, confirmation_seeds = seed_sets(bank_size is not None)
    l2_anchor = anchors["L2-Matched"]["config"]["l2"]
    if not math.isfinite(l2_anchor) or l2_anchor <= 0:
        raise ValueError(f"Cannot calibrate a zero or invalid L2 anchor for {dataset}")
    hot_posterior = max(.025 if dataset == "moons" else .01,
                        2 * max(row["config"]["posterior_std"] for row in anchors.values()))
    tight_prior = .25 if dataset == "moons" else .125
    rows, planned = [], {}
    for method, anchor in anchors.items():
        planned[method] = []
        variants = list(profiles(method, anchor["config"], l2_anchor, hot_posterior, tight_prior))
        if len(variants) != PROFILE_COUNT or any(
                config == earlier for i, (_, config, _) in enumerate(variants) for _, earlier, _ in variants[:i]):
            raise ValueError(f"Expected {PROFILE_COUNT} distinct profiles for {dataset}/{method}")
        for name, config, calibration in variants:
            provenance = dict(primary_refine_task=anchor["task_id"], profile=name, calibration=calibration,
                              **fingerprints)
            generated = [task("search", method, name, config, provenance, seed) for seed in validation_seeds]
            planned[method].append(generated[0]["candidate"])
            rows.extend(generated)

    manifest = performance.manifest_path(output, dataset, "search")
    performance.write_manifest(manifest, rows)
    fixed_bank = {} if bank_size is None else {"fixed_bank_size": bank_size}
    performance.write_json(plan_path(output, dataset), dict(
        dataset=dataset, bank_size=m, source=str(source), source_fingerprints=fingerprints,
        anchor_tasks={method: row["task_id"] for method, row in anchors.items()},
        l2_anchor=l2_anchor, hot_posterior=hot_posterior, tight_prior=tight_prior,
        validation_seeds=validation_seeds, confirmation_seeds=confirmation_seeds, profiles=planned,
        search_manifest_sha256=primary.sha256(manifest), **fixed_bank,
    ), frozen=True)
    print(f"{manifest} ({len(rows)} tasks), m={m}")


def replay_selection(output, dataset):
    planned_path = plan_path(output, dataset)
    planned = performance.read_json(planned_path)
    manifest = performance.manifest_path(output, dataset, "search")
    manifest_hash = primary.sha256(manifest)
    rows = performance.read_manifest(manifest)
    m = planned["bank_size"]
    validation_seeds, confirmation_seeds, profiles_per_method = search_design(planned)
    full_count = len(primary.METHODS) * profiles_per_method * len(validation_seeds)
    if (planned["dataset"], planned["search_manifest_sha256"], len(rows)) != (dataset, manifest_hash, full_count):
        raise ValueError(f"Incomplete or inconsistent search plan for {dataset}")

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        config = row["config"]
        if ((row["phase"], config["dataset"], config["m"]) != ("search", dataset, m) or
                row["method"] not in primary.METHODS):
            raise ValueError(f"Invalid regularization validation row {row['task_id']}")
        groups[row["method"]][row["candidate"]].append(row)

    scores, winners = {}, {}
    for method in primary.METHODS:
        members = groups[method]
        if set(members) != set(planned["profiles"][method]):
            raise ValueError(f"Unexpected profiles for {dataset}/{method}")
        scores[method] = {}
        for candidate, entries in members.items():
            configs = [{key: value for key, value in entry["config"].items() if key not in SEED_FIELDS}
                       for entry in entries]
            if (sorted(entry["config"]["seed"] for entry in entries) != list(validation_seeds) or
                    any(config != configs[0] for config in configs)):
                raise ValueError(f"Unpaired regularization validation entries for {dataset}/{candidate}")
            metrics = [performance.evaluate_result(output, entry)["heldout_gibbs_mc"] for entry in entries]
            scores[method][candidate] = dict(
                mean_accuracy=statistics.fmean(metric["accuracy"] for metric in metrics),
                mean_cross_entropy=statistics.fmean(metric["cross_entropy"] for metric in metrics),
                accuracy_by_seed={str(entry["config"]["seed"]): metric["accuracy"]
                                  for entry, metric in zip(entries, metrics)},
            )
        winners[method] = primary.best(scores[method])

    winning_scores = {method: scores[method][name] for method, name in winners.items()}
    proposed = primary.best({method: winning_scores[method] for method in primary.PROPOSED})
    baseline = primary.best({method: winning_scores[method] for method in primary.BASELINES})
    gap = 100 * (winning_scores[proposed]["mean_accuracy"] - winning_scores[baseline]["mean_accuracy"])
    anchors = {method: groups[method][name][0] for method, name in winners.items()}
    confirmation = []
    for method, entry in anchors.items():
        provenance = {**entry["source"], "validation_task": entry["task_id"]}
        confirmation.extend(task("confirmation", method, provenance["profile"], entry["config"], provenance, seed)
                            for seed in confirmation_seeds)

    selection = dict(
        dataset=dataset, chosen_bank_size=m, chosen_proposed=proposed, chosen_baseline=baseline,
        validation_gap_pp=gap, chosen_candidates=winners, validation_scores=scores,
        search_manifest_sha256=manifest_hash, search_plan_sha256=primary.sha256(planned_path),
        confirmation_seeds=list(confirmation_seeds),
    )
    return anchors, confirmation, selection


def select(output, dataset):
    _, confirmation, selection = replay_selection(output, dataset)
    selected_manifest = performance.manifest_path(output, dataset, "confirmation")
    performance.write_manifest(selected_manifest, confirmation)
    selection["confirmation_manifest_sha256"] = primary.sha256(selected_manifest)
    performance.write_json(selection_path(output, dataset), selection, frozen=True)
    print(f"{selected_manifest} ({len(confirmation)} tasks), m={selection['chosen_bank_size']}, "
          f"validation gap {selection['validation_gap_pp']:+.3f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    count_command = commands.add_parser("count")
    count_command.add_argument("--phase", choices=("search", "confirmation"), required=True)
    count_command.add_argument("--schedule-panel", action="store_true")
    for name in ("plan", "select"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dataset", choices=primary.DATASETS, required=True)
    commands.choices["plan"].add_argument("--source", type=Path, required=True, help="Primary performance directory")
    commands.choices["plan"].add_argument("--bank-size", type=int, choices=primary.BANK_SIZES,
                                         help="Use the same validated bank for all requested datasets")
    commands.choices["plan"].add_argument("--schedule-panel", action="store_true",
                                         help="Use fixed-bank regularization validation as the Two moons schedule anchor")
    args = parser.parse_args()
    if args.command == "count":
        print(count(args.phase, schedule_panel=args.schedule_panel))
    elif args.command == "plan":
        plan(args.output.resolve(), args.source.resolve(), args.dataset, bank_size=args.bank_size,
             schedule_panel=args.schedule_panel)
    else:
        select(args.output.resolve(), args.dataset)


if __name__ == "__main__":
    main()
