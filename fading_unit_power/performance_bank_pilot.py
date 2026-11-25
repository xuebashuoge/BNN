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
MARGIN_BANK_SIZES = (200, CONTROL_M)
MARGIN_SEEDS = (26112591, 26112592, 26112593)
MNIST_PILOT = ARCHIVE.parent / "performance_common_bank" / "bank_pilot" / "mnist"
MNIST_PILOT_REPORT_SHA256 = "fe6eb58e1f8a7f48a2213d0e2c7315e0199a53f0e09d2c994468c0da5a217cce"
MARGIN_PROFILES = (
    ("slow-cold", dict(lr=.00055, epochs=48, train_mc=1, posterior_std=.001, rho_lr_scale=.1), .35, 3e-7, 1.),
    ("decay-low", dict(lr=.00075, epochs=60, train_mc=1, posterior_std=.003, rho_lr_scale=.1), .30, 3e-6, .25),
    ("mc2-medium", dict(lr=.00125, epochs=48, train_mc=2, posterior_std=.01, rho_lr_scale=.3), .25, 3e-6, .25),
    ("warm-decay", dict(lr=.00075, epochs=60, train_mc=1, posterior_std=.025, rho_lr_scale=1.), .25, 3e-5, .15),
    ("warm-strong", dict(lr=.00125, epochs=48, train_mc=1, posterior_std=.04, rho_lr_scale=.5), .30, 1e-4, .10),
)


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


def margin_anchors(dataset):
    if dataset != "mnist":
        raise ValueError("The margin search is defined for MNIST")
    report_path = MNIST_PILOT / "bank_pilot_mnist.json"
    manifest_path = performance.manifest_path(MNIST_PILOT, dataset, "search")
    report_hash, manifest_hash = sweep.sha256(report_path), sweep.sha256(manifest_path)
    if report_hash != MNIST_PILOT_REPORT_SHA256:
        raise ValueError("The archived MNIST pilot report differs from the frozen margin-search source")
    report = performance.read_json(report_path)
    entries = performance.read_manifest(manifest_path)
    if (report["dataset"] != dataset or report["scope"] != "validation only" or report["control_m"] != CONTROL_M
            or report["training_seeds"] != list(SEEDS) or report["manifest_sha256"] != manifest_hash
            or entries != planned_rows(dataset)):
        raise ValueError("The archived MNIST pilot manifest differs from the frozen validation design")

    anchors = {}
    for row in report["rows"]:
        m = row["m"]
        if m not in MARGIN_BANK_SIZES:
            continue
        for method in sweep.METHODS:
            scores = row["candidate_validation"][method]
            candidate = sweep.best(scores)
            matched = [entry for entry in entries if (entry["config"]["m"], entry["method"], entry["candidate"])
                       == (m, method, candidate)]
            if candidate != row["methods"][method]["candidate"] or sorted(
                    entry["config"]["seed"] for entry in matched) != list(SEEDS):
                raise ValueError(f"Archived validation winner differs for {dataset}/{method}/m={m}")
            anchor = matched[0]
            config = {key: value for key, value in anchor["config"].items() if key not in regularization.SEED_FIELDS}
            anchors[m, method] = config, dict(
                pilot_task=anchor["task_id"], pilot_candidate=candidate,
                pilot_manifest_sha256=manifest_hash, pilot_report_sha256=report_hash,
            )
    common, _ = anchors[CONTROL_M, sweep.METHODS[0]]
    for (m, method), (config, _) in anchors.items():
        regularization.validate_matched_anchor(config, common, dataset, method, m)
    return anchors


def margin_profiles(base, method):
    yield "margin-incumbent", base
    for name, settings, end_lr, l2_strength, prior_std in MARGIN_PROFILES:
        config = {**base, **settings, "lr_decay": end_lr ** (1 / (settings["epochs"] - 1))}
        if method == "L2-Matched":
            config["l2"] = l2_strength
        if method in sweep.PROPOSED:
            config["prior_std"] = prior_std
            config.update(regularization.calibrated_penalty(
                config, method, prior_std, config["posterior_std"], l2_strength, .5))
        yield f"margin-{name}", config


def margin_rows(dataset):
    anchors = margin_anchors(dataset)
    rows = []
    for (m, method), (base, provenance) in anchors.items():
        for name, config in margin_profiles(base, method):
            rows.extend(regularization.task("search", method, name, config, provenance, seed)
                        for seed in MARGIN_SEEDS)
    return rows


def plan(output, dataset, *, margin_search=False):
    path = performance.manifest_path(output, dataset, "search")
    rows = margin_rows(dataset) if margin_search else planned_rows(dataset)
    performance.write_manifest(path, rows)
    print(f"{path} ({len(rows)} validation-only tasks)")


def pairwise_margins(accuracies):
    gaps = {f"{proposed.split('-')[0]}_vs_{baseline.split('-')[0]}": accuracies[proposed] - accuracies[baseline]
            for proposed in sweep.PROPOSED for baseline in sweep.BASELINES}
    best_baseline = max(accuracies[method] for method in sweep.BASELINES)
    margins = {method: accuracies[method] - best_baseline for method in sweep.PROPOSED}
    return dict(pairwise_gaps_pp=gaps, gap_to_strongest_baseline_pp=margins,
                both_floor_pp=min(margins.values()), either_floor_pp=max(margins.values()))


def bank_result(m, scores, *, seeds=SEEDS, include_margin=False):
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
              for seed in map(str, seeds)}
    paired_both = {seed: methods[proposed]["accuracy_pct_by_seed"][seed] -
                   max(methods[b]["accuracy_pct_by_seed"][seed] for b in sweep.BASELINES) for seed in paired}

    by_seed = []
    for seed in seeds:
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
    result = dict(m=m, proposed=proposed, baseline=baseline, methods=methods, mean_gap_pp=mean_gap_pp,
                  paired_gap_pp=paired, paired_gap_to_both_baselines_pp=paired_both, best_seed=best_seed,
                  candidate_validation=means)
    if include_margin:
        result.update(pairwise_margins({method: values["mean_accuracy_pct"] for method, values in methods.items()}))
        result["paired_diagnostics"] = {}
        for seed in map(str, seeds):
            accuracies = {method: values["accuracy_pct_by_seed"][seed] for method, values in methods.items()}
            result["paired_diagnostics"][seed] = dict(accuracy_pct=accuracies, **pairwise_margins(accuracies))
        for row in by_seed:
            row.update(pairwise_margins({method: values["accuracy_pct"] for method, values in row["methods"].items()}))
        result["best_pairwise_seed"] = max(by_seed, key=lambda row: (row["both_floor_pp"], row["either_floor_pp"]))
    return result


def write_margin_report(output, dataset, manifest, rows):
    control = next(row for row in rows if row["m"] == CONTROL_M)
    for row in rows:
        row["pairwise_delta_vs_control_pp"] = {pair: gap - control["pairwise_gaps_pp"][pair]
                                               for pair, gap in row["pairwise_gaps_pp"].items()}
        for name in ("both_floor", "either_floor"):
            row[f"{name}_delta_vs_control_pp"] = row[f"{name}_pp"] - control[f"{name}_pp"]
    prefix = output / f"bank_margin_{dataset}"
    performance.write_json(prefix.with_suffix(".json"), dict(
        dataset=dataset, scope="validation only", mode="margin search", control_m=CONTROL_M,
        training_seeds=MARGIN_SEEDS, validation_draws=regularization.EVAL_DRAWS["search"],
        manifest_sha256=sweep.sha256(manifest), rows=rows,
    ))
    pairs = tuple(control["pairwise_gaps_pp"])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["m", *(f"{method}_mean_accuracy_pct" for method in sweep.METHODS),
                     *(f"{pair}_pp" for pair in pairs),
                     *(f"{method}_gap_to_strongest_baseline_pp" for method in sweep.PROPOSED),
                     "both_floor_pp", "either_floor_pp", "both_floor_delta_vs_control_pp",
                     "either_floor_delta_vs_control_pp",
                     *(f"both_floor_pp_seed_{seed}" for seed in MARGIN_SEEDS),
                     "best_pairwise_seed", "best_pairwise_seed_both_floor_pp"])
    for row in rows:
        writer.writerow([row["m"], *(row["methods"][method]["mean_accuracy_pct"] for method in sweep.METHODS),
                         *(row["pairwise_gaps_pp"][pair] for pair in pairs),
                         *(row["gap_to_strongest_baseline_pp"][method] for method in sweep.PROPOSED),
                         row["both_floor_pp"], row["either_floor_pp"], row["both_floor_delta_vs_control_pp"],
                         row["either_floor_delta_vs_control_pp"],
                         *(row["paired_diagnostics"][str(seed)]["both_floor_pp"] for seed in MARGIN_SEEDS),
                         row["best_pairwise_seed"]["seed"], row["best_pairwise_seed"]["both_floor_pp"]])
    performance.write_text(prefix.with_suffix(".csv"), stream.getvalue())
    print(prefix.with_suffix(".json"))


def summarize(output, dataset, *, margin_search=False):
    manifest = performance.manifest_path(output, dataset, "search")
    entries = performance.read_manifest(manifest)
    expected = margin_rows(dataset) if margin_search else planned_rows(dataset)
    if entries != expected:
        raise ValueError(f"Validation manifest differs from the frozen pilot design for {dataset}")
    sizes, seeds = (MARGIN_BANK_SIZES, MARGIN_SEEDS) if margin_search else (BANK_SIZES, SEEDS)
    scores = {m: {method: {} for method in sweep.METHODS} for m in sizes}
    for entry in entries:
        config = entry["config"]
        metrics = performance.evaluate_result(output, entry)["heldout_gibbs_mc"]
        if (metrics["examples"] != config["val_size"] or not 0 <= metrics["accuracy"] <= 1
                or not math.isfinite(metrics["cross_entropy"])):
            raise ValueError(f"Invalid validation scores for {entry['task_id']}")
        scores[config["m"]][entry["method"]].setdefault(entry["candidate"], {})[config["seed"]] = metrics

    rows = [bank_result(m, scores[m], seeds=seeds, include_margin=margin_search) for m in sizes]
    if margin_search:
        write_margin_report(output, dataset, manifest, rows)
        return
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
        command.add_argument("--margin-search", action="store_true", help="Retune the MNIST m=200 and m=2000 validation banks")
    args = parser.parse_args()
    (plan if args.command == "plan" else summarize)(args.output, args.dataset, margin_search=args.margin_search)


if __name__ == "__main__":
    main()
