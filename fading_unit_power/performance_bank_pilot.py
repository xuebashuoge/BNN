"""Screen smaller fading channel banks on validation with a matched m=2000 control."""

import argparse
import csv
import io
import math
from pathlib import Path
import statistics

from . import performance, performance_regularization_sweep as regularization
from . import performance_sweep as sweep


DATASETS = ("mnist", "fashionmnist")
BANK_SIZES = (128, 144, 160, 176, 192, 200, 2000)
ANCHOR_SIZES = (128, 2000)
CONTROL_M = 2000
SEEDS = (26112581, 26112582)
ARCHIVE = sweep.ARCHIVE.parents[1] / "performance_sweep"


def archived_anchors(dataset):
    manifest = performance.manifest_path(ARCHIVE, dataset, "refine")
    ledger_path = ARCHIVE / "manifests" / f"{dataset}-refine-scores.json"
    prior_plan = performance.read_json(regularization.plan_path(ARCHIVE / "regularization" / dataset, dataset))
    manifest_hash, scores_hash = sweep.sha256(manifest), sweep.sha256(ledger_path)
    ledger = performance.read_json(ledger_path)
    fingerprints = prior_plan["source_fingerprints"]
    if ((ledger["dataset"], ledger["manifest_sha256"]) != (dataset, manifest_hash)
            or (fingerprints["refine_manifest_sha256"], fingerprints["refine_scores_sha256"])
            != (manifest_hash, scores_hash)):
        raise ValueError(f"Archived refinement manifest or scores differ for {dataset}")
    entries = performance.read_manifest(manifest)

    anchors = {}
    for m in ANCHOR_SIZES:
        scores = ledger["bank_sizes"][str(m)]
        for method in sweep.METHODS:
            candidate = sweep.best(scores[method])
            rows = [entry for entry in entries if (entry["config"]["m"], entry["method"], entry["candidate"]) ==
                    (m, method, candidate)]
            configs = [{key: value for key, value in row["config"].items() if key not in regularization.SEED_FIELDS}
                       for row in rows]
            if (candidate != ledger["winners"][str(m)][method]
                    or sorted(row["config"]["seed"] for row in rows) != list(sweep.REFINE_SEEDS)
                    or any(row["phase"] != "refine" or row["config"]["dataset"] != dataset for row in rows)
                    or any(config != configs[0] for config in configs)):
                raise ValueError(f"Invalid archived anchor for {dataset}/{method}/m={m}")
            anchors[m, method] = configs[0], dict(
                anchor_m=m, anchor_candidate=candidate, anchor_task=rows[0]["task_id"],
                refine_manifest_sha256=manifest_hash, refine_scores_sha256=scores_hash,
            )
    common, _ = anchors[CONTROL_M, sweep.METHODS[0]]
    for (m, method), (config, _) in anchors.items():
        regularization.validate_matched_anchor(config, common, dataset, method, m)
    return anchors


def transfer_bank(config, method, m):
    target = {**config, "m": m}
    if method in sweep.PROPOSED and m != config["m"]:
        calibration = (method, config["prior_std"], config["posterior_std"], 1., .5)
        ref_coefficients = regularization.calibrated_penalty(config, *calibration)
        new_coefficients = regularization.calibrated_penalty(target, *calibration)
        for name, coefficient in ref_coefficients.items():
            if config[name] != 0:
                target[name] = config[name] * (new_coefficients[name] / coefficient)
    return target


def planned_rows(dataset):
    anchors = archived_anchors(dataset)
    rows = []
    for m in BANK_SIZES:
        for method in sweep.METHODS:
            for anchor_m in ANCHOR_SIZES:
                config, provenance = anchors[anchor_m, method]
                transferred = transfer_bank(config, method, m)
                rows.extend(sweep.record("search", method, f"anchor-{anchor_m}", transferred, provenance, seed)
                            for seed in SEEDS)
    return rows


def plan(output, dataset):
    path = performance.manifest_path(output, dataset, "search")
    rows = planned_rows(dataset)
    performance.write_manifest(path, rows)
    print(f"{path} ({len(rows)} validation-only tasks)")


def bank_result(m, scores):
    means = {}
    for method, candidates in scores.items():
        means[method] = {}
        for name, trials in candidates.items():
            means[method][name] = dict(
                mean_accuracy=statistics.fmean(trial["accuracy"] for trial in trials.values()),
                mean_cross_entropy=statistics.fmean(trial["cross_entropy"] for trial in trials.values()),
                accuracy_by_seed={str(seed): trial["accuracy"] for seed, trial in trials.items()},
            )
    methods = {}
    for method, candidates in means.items():
        candidate = sweep.best(candidates)
        metrics = candidates[candidate]
        methods[method] = dict(candidate=candidate, mean_accuracy_pct=metrics["mean_accuracy"] * 100,
                               accuracy_pct_by_seed={seed: accuracy * 100 for seed, accuracy in
                                                     metrics["accuracy_by_seed"].items()})

    proposed, baseline = (max(group, key=lambda method: methods[method]["mean_accuracy_pct"])
                          for group in (sweep.PROPOSED, sweep.BASELINES))
    mean_gap_pp = methods[proposed]["mean_accuracy_pct"] - methods[baseline]["mean_accuracy_pct"]
    paired = {seed: methods[proposed]["accuracy_pct_by_seed"][seed] - methods[baseline]["accuracy_pct_by_seed"][seed]
              for seed in map(str, SEEDS)}
    paired_both = {seed: methods[proposed]["accuracy_pct_by_seed"][seed] -
                   max(methods[b]["accuracy_pct_by_seed"][seed] for b in sweep.BASELINES) for seed in paired}

    by_seed = []
    for seed in SEEDS:
        winners = {}
        for method, candidates in scores.items():
            name = min(candidates, key=lambda item: (-candidates[item][seed]["accuracy"],
                                                       candidates[item][seed]["cross_entropy"], item))
            winners[method] = dict(candidate=name, accuracy_pct=candidates[name][seed]["accuracy"] * 100)
        seed_proposed, seed_baseline = (max(group, key=lambda method: winners[method]["accuracy_pct"])
                                        for group in (sweep.PROPOSED, sweep.BASELINES))
        by_seed.append(dict(seed=seed, proposed=seed_proposed, baseline=seed_baseline, methods=winners,
                            gap_pp=winners[seed_proposed]["accuracy_pct"] - winners[seed_baseline]["accuracy_pct"]))
    best_seed = max(by_seed, key=lambda row: (row["gap_pp"], row["methods"][row["proposed"]]["accuracy_pct"]))
    return dict(m=m, proposed=proposed, baseline=baseline, methods=methods, mean_gap_pp=mean_gap_pp,
                paired_gap_pp=paired, paired_gap_to_both_baselines_pp=paired_both, best_seed=best_seed,
                candidate_validation=means)


def summarize(output, dataset):
    manifest = performance.manifest_path(output, dataset, "search")
    entries = performance.read_manifest(manifest)
    if entries != planned_rows(dataset):
        raise ValueError(f"Validation manifest differs from the frozen pilot design for {dataset}")
    scores = {m: {method: {} for method in sweep.METHODS} for m in BANK_SIZES}
    for entry in entries:
        config = entry["config"]
        metrics = performance.evaluate_result(output, entry)["heldout_gibbs_mc"]
        if (metrics["examples"] != config["val_size"] or not 0 <= metrics["accuracy"] <= 1
                or not math.isfinite(metrics["cross_entropy"])):
            raise ValueError(f"Invalid validation scores for {entry['task_id']}")
        scores[config["m"]][entry["method"]].setdefault(entry["candidate"], {})[config["seed"]] = metrics

    rows = [bank_result(m, scores[m]) for m in BANK_SIZES]
    control_gap = next(row["mean_gap_pp"] for row in rows if row["m"] == CONTROL_M)
    for row in rows:
        row["mean_gap_delta_vs_control_pp"] = row["mean_gap_pp"] - control_gap
    prefix = output / f"bank_pilot_{dataset}"
    performance.write_json(prefix.with_suffix(".json"), dict(
        dataset=dataset, scope="validation only", control_m=CONTROL_M, training_seeds=SEEDS,
        validation_draws=sweep.EVAL_DRAWS["search"], manifest_sha256=sweep.sha256(manifest), rows=rows,
    ))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["m", *(f"{method}_mean_accuracy_pct" for method in sweep.METHODS),
                     "proposed", "baseline", "mean_gap_pp", "mean_gap_delta_vs_control_pp",
                     *(f"paired_gap_to_both_baselines_pp_seed_{seed}" for seed in SEEDS),
                     "best_validation_seed", "best_seed_gap_pp"])
    for row in rows:
        writer.writerow([row["m"], *(row["methods"][method]["mean_accuracy_pct"] for method in sweep.METHODS),
                         row["proposed"], row["baseline"], row["mean_gap_pp"], row["mean_gap_delta_vs_control_pp"],
                         *row["paired_gap_to_both_baselines_pp"].values(), row["best_seed"]["seed"],
                         row["best_seed"]["gap_pp"]])
    performance.write_text(prefix.with_suffix(".csv"), stream.getvalue())
    print(prefix.with_suffix(".json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    (plan if args.command == "plan" else summarize)(args.output, args.dataset)


if __name__ == "__main__":
    main()
