"""Audit the frozen four-method CIFAR confirmation using saved JSON only.

No weights are opened, and no training or checkpoint inference is performed.
The validation-prespecified primary and the observed test-mean winner remain
distinct. Every method/seed, including unfavorable outcomes, is retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO / "journal/cifar_partition_selection.json"
DEFAULT_PREFIX = REPO / "results/journal_cifar_search/confirmation"
METHODS = {"ERM-Matched": "erm", "L2-Matched": "l2",
           "T1-Matched": "proposed", "T2-Matched": "joint"}
SEEDS = [8101, 8102, 8103]
METRICS = ("accuracy", "bounded_loss", "cross_entropy", "feature_power_per_coordinate")
COHORT = ("dataset", "architecture", "train_size", "val_size", "epochs", "batch_size",
          "width", "m", "data_seed", "deployment_channel", "training_loss", "loss_epsilon",
          "clip_ce", "prior_std", "delta", "selection_metric", "validation_seed_policy",
          "val_draws", "device", "threads", "cpu_flush_denormal")
SEED_FIELDS = {"seed", "channel_seed", "eval_seed"}
TOLERANCE = 1e-12


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def numeric(value, name, upper=None):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0
            or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.fmean(values), sd=statistics.stdev(values))


def active_hyperparameters(config):
    result = dict(learning_rate=config["lr"])
    if config["method"] == "l2":
        result["l2_coefficient"] = config["l2"]
    if config["method"] in {"proposed", "joint"}:
        result.update(initial_posterior_std=config["posterior_std"],
                      rho_lr_scale=config["rho_lr_scale"],
                      rho_learning_rate=config["lr"] * config["rho_lr_scale"],
                      posterior_scale_policy=("fixed positive Gaussian" if config["rho_lr_scale"] == 0
                                              else "learned diagonal Gaussian"))
        if config["method"] == "proposed":
            result.update(channel_coefficient=config["eta_channel"], data_coefficient=config["eta_data"])
        else:
            result["joint_coefficient"] = config["eta_data"]
    return result


def comparison(method, best_baseline, by_method):
    candidate = by_method[method]
    baselines = [by_method[label] for label in ("ERM-Matched", "L2-Matched")]
    margins = []
    for observation in candidate["observations"]:
        seed = observation["seed"]
        fixed = next(row for row in best_baseline["observations"] if row["seed"] == seed)
        strongest = max(((base["method"], next(row for row in base["observations"] if row["seed"] == seed))
                         for base in baselines), key=lambda pair: pair[1]["accuracy"])
        margins.append(dict(seed=seed,
                            margin_vs_best_baseline_by_mean_pp=100 * (observation["accuracy"]-fixed["accuracy"]),
                            strongest_baseline_at_this_seed=strongest[0],
                            margin_vs_strongest_baseline_at_seed_pp=100 * (observation["accuracy"]-strongest[1]["accuracy"]),
                            strictly_beats_both_baselines=observation["accuracy"] > strongest[1]["accuracy"]+TOLERANCE))
    margin = candidate["accuracy"]["mean"] - best_baseline["accuracy"]["mean"]
    return dict(method=method, baseline_fixed_by_three_seed_mean=best_baseline["method"],
                mean_accuracy_margin_pp=100 * margin, goal_met=margin > TOLERANCE,
                per_seed_margins=margins,
                all_seeds_strictly_beat_both_baselines=all(row["strictly_beats_both_baselines"] for row in margins))


def load_report(root, selection_path=DEFAULT_SELECTION):
    root, selection_path = Path(root).resolve(), Path(selection_path).resolve()
    sources = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        sources[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    summary, rows = read(root / "summary.json"), read(root / "test_results.json")
    if (summary["training_trials"] != 12 or summary["completed_trials"] != 12
            or summary["failed_trials"] or summary["failed_evaluations"]
            or summary.get("selection_uses_test") is not False
            or not isinstance(rows, list) or len(rows) != 12):
        raise ValueError("Expected a complete twelve-case validation-selected campaign")
    if len({row["task_id"] for row in rows}) != 12:
        raise ValueError("Duplicate test rows cannot replace a missing seed")
    if ({row["method_label"] for row in rows} != set(METHODS)
            or len({row["group"] for row in rows}) != 1):
        raise ValueError("Expected all four matched methods in one group")
    group = rows[0]["group"]
    choices = [choice for choice in summary["selected_variants"] if choice.get("rank", 1) == 1]
    if (len(choices) != 4 or {choice["method_label"] for choice in choices} != set(METHODS)
            or any(choice.get("complete_seed_set") is not True or sorted(choice["seeds"]) != SEEDS
                   for choice in choices)):
        raise ValueError("Expected four complete three-seed selected configurations")
    chosen_tasks = [task for choice in choices for task in choice["task_ids"]]
    if len(chosen_tasks) != 12 or set(chosen_tasks) != {row["task_id"] for row in rows}:
        raise ValueError("Selected tasks disagree with the twelve raw evaluations")

    freeze = read(selection_path)
    if (freeze["status"] != "selected" or freeze["selection_uses_test"] is not False
            or freeze["selected_group"] != group or freeze["selected_architecture"] != "cnn_spatial3"
            or sorted(freeze["confirmation_seeds"]) != SEEDS):
        raise ValueError("Missing or inconsistent validation-only frozen selection")
    declared = {entry["task_id"]: entry for entry in freeze["confirmation_entries"]}
    if len(declared) != 12 or set(declared) != {row["task_id"] for row in rows}:
        raise ValueError("Frozen confirmation entries do not match all evaluated tasks")
    primary_groups = [item for item in freeze["groups"] if item["group"] == group]
    if len(primary_groups) != 1 or primary_groups[0]["best_proposed_method"] != "T2-Matched":
        raise ValueError("This frozen confirmation prespecified T2 from validation")
    primary = "T2-Matched"

    common = {key: rows[0]["config"][key] for key in COHORT}
    required = dict(dataset="cifar10", architecture="cnn_spatial3", train_size=45000,
                    val_size=5000, epochs=25, width=128, m=100, device="cpu", threads=1,
                    cpu_flush_denormal=True, training_loss="bounded_ce_smooth", loss_epsilon=.01,
                    selection_metric="accuracy", validation_seed_policy="fixed")
    if any(common[key] != value for key, value in required.items()):
        raise ValueError("Unexpected frozen spatial-CNN confirmation protocol")
    methods, split_signatures, split_records, bank_hashes, network_signatures, code_revisions = [], set(), [], {}, set(), set()
    network = None
    for label, algorithm in METHODS.items():
        runs = sorted((row for row in rows if row["method_label"] == label), key=lambda row: row["seed"])
        if [row["seed"] for row in runs] != SEEDS or len({row["variant_id"] for row in runs}) != 1:
            raise ValueError("Each method must retain one frozen configuration and all three seeds")
        fixed = {key: value for key, value in runs[0]["config"].items() if key not in SEED_FIELDS}
        choice = next(item for item in choices if item["method_label"] == label)
        if (set(choice["task_ids"]) != {row["task_id"] for row in runs}
                or choice["group"] != group or choice["variant_id"] != runs[0]["variant_id"]
                or {key: value for key, value in choice["config"].items() if key not in SEED_FIELDS} != fixed):
            raise ValueError("Campaign choices disagree with frozen method settings")
        observations = []
        for row in runs:
            config = row["config"]
            entry = declared[row["task_id"]]
            family = "baseline" if algorithm in {"erm", "l2"} else "proposed"
            if (row.get("status") != "complete" or row.get("test_status") != "complete"
                    or config["method"] != algorithm or row["family"] != family
                    or config != entry["config"] or entry.get("selection_uses_test") is not False
                    or any(row[key] != entry[key] for key in ("group", "method_label", "variant_id", "family"))
                    or {key: config[key] for key in COHORT} != common
                    or {key: value for key, value in config.items() if key not in SEED_FIELDS} != fixed
                    or config["train_channel"] != config["deployment_channel"]):
                raise ValueError("Changed, failed, mismatched or unfrozen method configuration")
            directory = root / row["task_id"]
            report = read(directory / "test_deployment/evaluation.json")
            metadata = read(directory / "metadata.json")
            history = read(directory / "history.json")
            if (metadata["config"] != config or metadata.get("git_dirty") is not False
                    or metadata["git_sha"] != row["git_sha"]
                    or metadata["parameter_count"] != 161266
                    or metadata["cpu_numerics"] != dict(flush_denormal_requested=True, flush_denormal_supported=True)
                    or metadata["cpu_threading"]["torch_intraop_threads"] != 1):
                raise ValueError("Checkpoint metadata/configuration/numerical policy disagrees")
            code_revisions.add(metadata["git_sha"])
            network = metadata["network_split"]
            if (network["transmitted_shape"] != [8, 4, 4] or network["real_channel_uses"] != 128
                    or network["transmitter_parameter_count"] != 24104
                    or network["receiver_parameter_count"] != 137162
                    or not isinstance(network["description"], str) or not network["description"]):
                raise ValueError("Unexpected transmitter/receiver split or communication budget")
            network_signatures.add(canonical(network))
            split = metadata["split"]
            if (split["train_size"] != 45000 or split["validation_size"] != 5000
                    or split["data_seed"] != config["data_seed"]):
                raise ValueError("Metadata records a different data split")
            split_signatures.add((split["train_indices_hash"], split["validation_indices_hash"], split["transform"]))
            split_records.append(split)
            bank_hashes.setdefault(row["seed"], set()).add(metadata["training_channel_bank_hash"])
            epoch = row["selected_epoch"]
            if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 25
                    or len(history) != 25 or [record["epoch"] for record in history] != list(range(1, 26))
                    or epoch != metadata["selected_epoch"] or epoch != report["selected_epoch"]):
                raise ValueError("Incomplete or inconsistent 25-epoch checkpoint history")
            for record in history:
                numeric(record["validation"]["accuracy"], "validation accuracy", 1)
                numeric(record["validation"]["cross_entropy"], "validation CE")
            best = max(history, key=lambda record: (record["validation"]["accuracy"], -record["validation"]["cross_entropy"]))
            if best["epoch"] != epoch:
                raise ValueError("Checkpoint is not the declared validation-selected epoch")
            evaluation_config = {**config, "eval_draws": 100, "eval_seed": config["eval_seed"]+1_000_003}
            if (report.get("split") != "test" or report.get("risk_scope") != "deployment"
                    or Path(report["checkpoint"]).name != row["task_id"]
                    or Path(row["checkpoint"]).name != row["task_id"]
                    or report["evaluation_config"] != evaluation_config):
                raise ValueError("Expected fresh deployment-only test evaluation of the frozen configuration")
            heldout = report["heldout_gibbs_mc"]
            if (heldout["examples"] != 10000 or heldout["draws"] != 100
                    or len(heldout["replicate_statistics"]) != 100
                    or heldout["channel_source"] != "deployment"
                    or heldout["posterior_rule"] != "one weight draw fixed across entire loader per replicate"
                    or heldout["bounded_loss_definition"] != "-log(epsilon+(1-epsilon)*p_y)/(-log(epsilon)); epsilon=0.01"):
                raise ValueError("Expected 100 full-loader Gibbs draws on the official 10k test set")
            computed = {}
            for metric in METRICS:
                upper = 1 if metric in {"accuracy", "bounded_loss"} else None
                values = [numeric(rep[metric], metric, upper) for rep in heldout["replicate_statistics"]]
                computed[metric] = statistics.fmean(values)
                if abs(computed[metric]-numeric(heldout[metric], metric, upper)) > TOLERANCE:
                    raise ValueError(f"Raw replicate/aggregate mismatch: {metric}")
            if (abs(computed["accuracy"]-numeric(row["test_accuracy"], "campaign accuracy", 1)) > TOLERANCE
                    or abs(computed["cross_entropy"]-numeric(row["test_cross_entropy"], "campaign CE")) > TOLERANCE):
                raise ValueError("Campaign test row disagrees with recomputed raw replicates")
            observations.append(dict(seed=row["seed"], task_id=row["task_id"], selected_epoch=epoch,
                                     checkpoint_git_sha=metadata["git_sha"], evaluation_draws=100,
                                     examples=10000, **computed))
        methods.append(dict(method=label, family=family, variant_id=runs[0]["variant_id"],
                            active_hyperparameters=active_hyperparameters(runs[0]["config"]),
                            config=runs[0]["config"], observations=observations,
                            **{metric: stats(row[metric] for row in observations) for metric in METRICS}))
    if (len(split_signatures) != 1 or len(network_signatures) != 1 or len(code_revisions) != 1
            or any(len(values) != 1 for values in bank_hashes.values())):
        raise ValueError("Data/model/code/bank conditions differ across the comparison")
    by_method = {item["method"]: item for item in methods}
    best_baseline = max((by_method[label] for label in ("ERM-Matched", "L2-Matched")), key=lambda item: item["accuracy"]["mean"])
    best_proposed = max((by_method[label] for label in ("T1-Matched", "T2-Matched")), key=lambda item: item["accuracy"]["mean"])
    comparisons = {label: comparison(label, best_baseline, by_method) for label in ("T1-Matched", "T2-Matched")}
    return dict(schema_version=1, root=str(root), training_trials=12, completed_trials=12,
                failed_trials=[], failed_evaluations=[], methods=methods,
                protocol={**common, "seeds": SEEDS, "test_examples": 10000, "test_draws": 100,
                          "parameter_count": 161266, "network_split": network,
                          "split": split_records[0], "training_code_revision": next(iter(code_revisions))},
                frozen_primary_from_validation=primary,
                frozen_primary_comparison=comparisons[primary],
                observed_best_proposed_by_test_mean=best_proposed["method"],
                observed_best_baseline_by_test_mean=best_baseline["method"],
                fixed_algorithm_comparisons=comparisons,
                goal_met=comparisons[best_proposed["method"]]["goal_met"],
                goal_definition="One fixed proposed algorithm's three-seed mean accuracy strictly exceeds both ERM/L2 means; no per-seed objective switching",
                strict_tolerance=TOLERANCE, selection_uses_test=False, hyperconfigs_locked=True,
                dataset_reused_from_prior_study=True,
                uncertainty="Sample SD over three initialization/channel-bank seeds conditional on a reused fixed data split; MC draws are not training replications",
                interpretation="The test-mean winner is descriptive; it does not replace the validation-prespecified T2 primary. All four methods and every declared seed are retained.",
                validation_freeze=dict(path=str(selection_path), selection_rule=freeze["selection_rule"],
                                       interpretation=freeze["interpretation"]),
                certificate_note="These are finite-test Gibbs estimates of accuracy and bounded risk, not population-risk or accuracy certificates",
                source_hashes=sources)


def write_report(report, output_prefix):
    output_prefix = Path(output_prefix)
    stream = io.StringIO(newline="")
    records = []
    for item in report["methods"]:
        records.append(dict(method=item["method"], seeds=json.dumps(SEEDS),
                            accuracy_percent=100*item["accuracy"]["mean"], accuracy_sd_pp=100*item["accuracy"]["sd"],
                            bounded_risk=item["bounded_loss"]["mean"], bounded_risk_sd=item["bounded_loss"]["sd"],
                            cross_entropy=item["cross_entropy"]["mean"], cross_entropy_sd=item["cross_entropy"]["sd"],
                            feature_power=item["feature_power_per_coordinate"]["mean"],
                            feature_power_sd=item["feature_power_per_coordinate"]["sd"],
                            selected_epochs=json.dumps([row["selected_epoch"] for row in item["observations"]]),
                            active_hyperparameters=canonical(item["active_hyperparameters"]),
                            transmitter_parameters=24104, receiver_parameters=137162, real_channel_uses=128,
                            prespecified_primary=item["method"] == report["frozen_primary_from_validation"],
                            observed_best_proposed_mean=item["method"] == report["observed_best_proposed_by_test_mean"]))
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    payloads = {Path(str(output_prefix)+".csv"): stream.getvalue(),
                Path(str(output_prefix)+".json"): json.dumps(report, indent=2, allow_nan=False)+"\n"}
    for path, value in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="Validation-only frozen partition selection, including all twelve confirmation entries")
    args = parser.parse_args()
    try:
        report = load_report(args.root, args.selection)
        write_report(report, args.output_prefix)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"Four-method confirmation report not generated: {error}\n")
    print(json.dumps({key: report[key] for key in ("frozen_primary_from_validation", "observed_best_proposed_by_test_mean", "observed_best_baseline_by_test_mean", "goal_met")}))


if __name__ == "__main__":
    main()
