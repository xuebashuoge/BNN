"""Tune four matched training objectives on the same unit-power fading channel."""

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from . import performance


DATASETS = performance.DATASETS
METHODS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched")
BASELINES = METHODS[:2]
PROPOSED = METHODS[2:]
BANK_SIZES = (16, 128, 2000)
SEARCH_SEEDS = (24112401, 24112402)
REFINE_SEEDS = (24112411, 24112412, 24112413)
TEST_SEEDS = (24112421, 24112422, 24112423, 24112424, 24112425)
SEARCH_CANDIDATES = ("archived", "long-decay", "fast-decay", "mc3", "regularization")
ARCHIVE = Path(__file__).resolve().parents[1] / "results/final_chosen_results/fading_unit_power/performance/provenance"
METHOD_OBJECTIVE = dict(zip(METHODS, ("erm", "l2", "proposed", "joint")))
EVAL_DRAWS = {"search": 24, "refine": 40, "confirmation": 64}


def sha256(path):
    return performance.historical_table.file_sha256(path)


def archive_config(dataset, method):
    root = ARCHIVE / dataset / "refined"
    manifest = root / "search.jsonl"
    selection = performance.read_json(root / "selection.json")
    manifest_hash = sha256(manifest)
    if selection["dataset"] != dataset or selection["search_manifest_sha256"] != manifest_hash:
        raise ValueError(f"Archived selection and manifest differ for {dataset}")
    candidate = selection["chosen"][method]
    rows = [row for row in performance.read_manifest(manifest)
            if (row["method"], row["candidate"]) == (method, candidate)]
    configs = [{key: value for key, value in row["config"].items()
                if key not in {"seed", "channel_seed", "eval_seed"}} for row in rows]
    if len(configs) != 2 or configs[0] != configs[1]:
        raise ValueError(f"Expected a paired archived configuration for {dataset}/{method}")
    config = configs[0]
    if (config["dataset"], config["method"], config["predictor"], config["train_channel"],
            config["deployment_channel"], config["transmit_unit_power"]) != (
            dataset, METHOD_OBJECTIVE[method], "gaussian", performance.FADING, performance.FADING, True):
        raise ValueError(f"Unexpected archived comparison conditions for {dataset}/{method}")
    return config, dict(archived_validation_task=rows[0]["task_id"], archive_manifest_sha256=manifest_hash)


def regularization(config, method, data_scale, channel_scale):
    if method == "L2-Matched":
        return {"l2": config["l2"] * data_scale}
    if method == "T1-Matched":
        return {"eta_data": config["eta_data"] * data_scale, "eta_channel": config["eta_channel"] * channel_scale}
    if method == "T2-Matched":
        return {"eta_data": config["eta_data"] * data_scale}
    return {"posterior_std": config["posterior_std"] * data_scale}


def sweep_variant(base, method, m, candidate):
    config = {**base, "m": m, "lr_decay": 1., "train_mc": 1}
    if candidate == "archived":
        return config

    # Keep the final LR fraction comparable across different epoch budgets.
    lr_scale, epoch_scale, end_lr, draws, data_scale, channel_scale = {
        "long-decay": (1., 1.6, .2, 1, .3, .3),
        "fast-decay": (3., 1., .12, 2, 3., 3.),
        "mc3": (1.5, .8, .4, 3, .1, 10.),
        "regularization": (.7, 1.25, 1., 1, 10., .1),
    }[candidate]
    if method == "ERM-Matched":
        data_scale = {"long-decay": 1., "fast-decay": 1., "mc3": .33, "regularization": 3.}[candidate]
    epochs = round(base["epochs"] * epoch_scale)
    return {
        **config, **regularization(config, method, data_scale, channel_scale),
        "lr": base["lr"] * lr_scale, "epochs": epochs, "train_mc": draws,
        "lr_decay": end_lr ** (1 / max(1, epochs - 1)),
    }


def record(phase, method, candidate, config, source, seed):
    values = {**config, "seed": seed, "channel_seed": seed + 10_000, "eval_seed": seed + 20_000}
    values["eval_draws"] = EVAL_DRAWS[phase]
    evaluation = dict(eval_draws=values["eval_draws"], eval_seed=seed + 1_000_003)
    name = f"sweep-{phase}-m{config['m']}-{candidate}"
    return performance.task_entry(phase, method, name, values, evaluation, source)


def count(phase):
    per_method = {"search": len(BANK_SIZES) * len(SEARCH_CANDIDATES) * len(SEARCH_SEEDS),
                  "refine": len(BANK_SIZES) * 2 * len(REFINE_SEEDS),
                  "confirmation": len(TEST_SEEDS)}
    return len(METHODS) * per_method[phase]


def plan(output):
    for dataset in DATASETS:
        rows = []
        for method in METHODS:
            base, source = archive_config(dataset, method)
            for m in BANK_SIZES:
                for candidate in SEARCH_CANDIDATES:
                    config = sweep_variant(base, method, m, candidate)
                    rows.extend(record("search", method, candidate, config, source, seed) for seed in SEARCH_SEEDS)
        path = performance.manifest_path(output, dataset, "search")
        performance.write_manifest(path, rows)
        print(f"{path} ({len(rows)} tasks)")


def candidates(output, dataset, phase, seeds):
    path = performance.manifest_path(output, dataset, phase)
    groups = defaultdict(lambda: defaultdict(list))
    rows = performance.read_manifest(path)
    if len(rows) != count(phase):
        raise ValueError(f"Expected {count(phase)} {phase} tasks in {path}")
    for row in rows:
        config = row["config"]
        if (row["phase"], config["dataset"]) != (phase, dataset):
            raise ValueError(f"Unexpected {phase} task in {path}: {row['task_id']}")
        groups[(config["m"], row["method"])][row["candidate"]].append(row)
    if set(groups) != {(m, method) for m in BANK_SIZES for method in METHODS}:
        raise ValueError(f"Missing bank sizes or methods in {path}")

    scores = {}
    for key, members in groups.items():
        if len(members) != (len(SEARCH_CANDIDATES) if phase == "search" else 2):
            raise ValueError(f"Unbalanced candidate count for {dataset}/{key}")
        scores[key] = {}
        for name, entries in members.items():
            configs = [{k: v for k, v in entry["config"].items() if k not in {"seed", "channel_seed", "eval_seed"}}
                       for entry in entries]
            if sorted(entry["config"]["seed"] for entry in entries) != list(seeds) or any(
                    config != configs[0] for config in configs):
                raise ValueError(f"Unpaired validation entries for {dataset}/{name}")
            metrics = [performance.evaluate_result(output, row)["heldout_gibbs_mc"] for row in entries]
            scores[key][name] = dict(
                mean_accuracy=statistics.fmean(metric["accuracy"] for metric in metrics),
                mean_cross_entropy=statistics.fmean(metric["cross_entropy"] for metric in metrics),
                accuracy_by_seed={str(row["config"]["seed"]): metric["accuracy"]
                                  for row, metric in zip(entries, metrics)},
            )
    return groups, scores


def best(scores):
    return min(scores, key=lambda name: (-scores[name]["mean_accuracy"], scores[name]["mean_cross_entropy"], name))


def refined_variant(base, method, m, archive_epochs):
    # Alternate a stronger MC average, a longer schedule and a penalty/LR change.
    if m == BANK_SIZES[0]:
        return {**base, "train_mc": 4, "lr": base["lr"] * .85, "epochs": min(base["epochs"], archive_epochs)}
    if m == BANK_SIZES[1]:
        epochs = archive_epochs * 2
        return {**base, "epochs": epochs, "lr": base["lr"] * 1.25,
                "lr_decay": .18 ** (1 / (epochs - 1)), "train_mc": min(base["train_mc"], 2)}
    return {**base, **regularization(base, method, .4, 2.), "lr": base["lr"] * .65}


def write_leaderboard(output, dataset, phase, scores, winners):
    path = performance.manifest_path(output, dataset, phase)
    result = dict(dataset=dataset, manifest_sha256=sha256(path), winners=winners,
                  bank_sizes={m: {method: scores[m, method] for method in METHODS} for m in BANK_SIZES})
    performance.write_json(output / "manifests" / f"{dataset}-{phase}-scores.json", result, frozen=True)


def promote(output, dataset):
    groups, scores = candidates(output, dataset, "search", SEARCH_SEEDS)
    rows = []
    winners = {}
    for m in BANK_SIZES:
        winners[m] = {}
        for method in METHODS:
            key = m, method
            name = best(scores[key])
            anchor = groups[key][name][0]
            archive_epochs = groups[key][f"sweep-search-m{m}-archived"][0]["config"]["epochs"]
            winners[m][method] = name
            source = {**anchor["source"], "search_candidate": name, "search_task": anchor["task_id"]}
            for candidate, config in (("promoted", anchor["config"]),
                                      ("local", refined_variant(anchor["config"], method, m, archive_epochs))):
                rows.extend(record("refine", method, candidate, config, source, seed) for seed in REFINE_SEEDS)
    write_leaderboard(output, dataset, "search", scores, winners)
    path = performance.manifest_path(output, dataset, "refine")
    performance.write_manifest(path, rows)
    print(f"{path} ({len(rows)} tasks)")


def select(output, dataset):
    groups, scores = candidates(output, dataset, "refine", REFINE_SEEDS)
    chosen = {m: {method: best(scores[m, method]) for method in METHODS} for m in BANK_SIZES}
    leaderboard = {}
    for m, by_method in chosen.items():
        winning_scores = {method: scores[m, method][name] for method, name in by_method.items()}
        proposed = best({method: winning_scores[method] for method in PROPOSED})
        baseline = best({method: winning_scores[method] for method in BASELINES})
        gap = winning_scores[proposed]["mean_accuracy"] - winning_scores[baseline]["mean_accuracy"]
        seed_gaps = {}
        for seed in map(str, REFINE_SEEDS):
            baseline_accuracy = max(winning_scores[method]["accuracy_by_seed"][seed] for method in BASELINES)
            seed_gaps[seed] = 100 * (winning_scores[proposed]["accuracy_by_seed"][seed] - baseline_accuracy)
        leaderboard[m] = dict(
            chosen=by_method, proposed=proposed, baseline=baseline, gap_pp=100 * gap,
            proposed_accuracy=winning_scores[proposed]["mean_accuracy"], seed_gap_pp=seed_gaps,
            strongest_validation_seed=max(seed_gaps, key=seed_gaps.get), scores=winning_scores,
        )
    m = min(leaderboard, key=lambda size: (-leaderboard[size]["gap_pp"], -leaderboard[size]["proposed_accuracy"], size))
    selected = leaderboard[m]
    rows = []
    for method, name in selected["chosen"].items():
        anchor = groups[m, method][name][0]
        source = {**anchor["source"], "refinement_candidate": name, "refinement_task": anchor["task_id"]}
        rows.extend(record("confirmation", method, name.removeprefix(f"sweep-refine-m{m}-"),
                           anchor["config"], source, seed) for seed in TEST_SEEDS)
    path = performance.manifest_path(output, dataset, "confirmation")
    performance.write_manifest(path, rows)
    write_leaderboard(output, dataset, "refine", scores, chosen)
    performance.write_json(output / "manifests" / f"{dataset}-sweep-selection.json", dict(
        dataset=dataset, chosen_bank_size=m, chosen_proposed=selected["proposed"], chosen_baseline=selected["baseline"],
        validation_gap_pp=selected["gap_pp"], validation_leaderboard=leaderboard, confirmation_seeds=TEST_SEEDS,
        confirmation_manifest_sha256=sha256(path),
    ), frozen=True)
    print(f"{path} ({len(rows)} tasks), m={m}, validation gap {selected['gap_pp']:+.3f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("count").add_argument("--phase", choices=("search", "refine", "confirmation"), required=True)
    for name in ("plan", "promote", "select", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name != "plan":
            command.add_argument("--dataset", choices=DATASETS, required=name != "summarize")
    args = parser.parse_args()
    if args.command == "count":
        print(count(args.phase))
        return
    output = args.output.resolve()
    if args.command == "plan":
        plan(output)
    elif args.command == "promote":
        promote(output, args.dataset)
    elif args.command == "select":
        select(output, args.dataset)
    else:
        from .performance_sweep_report import summarize
        summarize(output, args.dataset)


if __name__ == "__main__":
    main()
