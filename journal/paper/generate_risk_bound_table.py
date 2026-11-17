"""Report selected image algorithms with honest risk/bound availability.

This script reads saved JSON only. It never trains, loads weights, evaluates a
checkpoint, or submits a job. The JSONL queue describes missing checkpoint-only
inference, performed separately by ``journal.evaluate_empirical_bank``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tables"
METHODS = ["ERM-Perfect", "ERM-Matched", "L2-Perfect", "L2-Matched",
           "PAC-Perfect", "T1-Matched", "T2-Matched"]
LABELS = {"ERM-Perfect": "ERM (perfect)", "ERM-Matched": "ERM (matched)",
          "L2-Perfect": r"$\ell_2$ (perfect)", "L2-Matched": r"$\ell_2$ (matched)",
          "PAC-Perfect": "Proposed (perfect)",
          "T1-Matched": "Proposed T1 (matched)",
          "T2-Matched": "Proposed T2 (matched)"}
COHORTS = [("MNIST", "results/journal_campaign/full_refine", "mnist_analog"),
           ("FashionMNIST", "results/journal_campaign/fashion_controls", "fashion_analog"),
           ("CIFAR-10", "results/journal_cifar/confirm_cpu", "cifar10_cnn_analog_severe")]
EVALUATION_SETTINGS = {"device", "threads", "batch_size", "download", "eval_draws", "eval_seed"}


def relative(path):
    return Path(path).relative_to(REPO).as_posix()


def read_json(path, sources):
    raw = path.read_bytes()
    sources[relative(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw)


def statistic(values):
    values = list(values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("A complete finite observation set is required")
    return dict(mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None,
                n=len(values))


def matching_config(actual, expected):
    return all(actual.get(key) == value for key, value in expected.items()
               if key not in EVALUATION_SETTINGS)


def validate_evaluation(report, trial):
    if (report.get("split") != "test"
            or Path(report["checkpoint"]).name != trial["task_id"]
            or not matching_config(report["evaluation_config"], trial["config"])):
        raise ValueError(f"Evaluation does not identify the selected checkpoint: {trial['task_id']}")
    heldout = report["heldout_gibbs_mc"]
    if heldout["draws"] != 10 or heldout["examples"] != 10000:
        raise ValueError("The image table requires ten draws on the official 10,000-example test set")
    if heldout["channel_source"] != "deployment":
        raise ValueError("Test risk must use the deployment channel law")
    if not all(0 <= heldout[key] <= 1 for key in ("accuracy", "bounded_loss")):
        raise ValueError("Accuracy or bounded loss outside [0,1]")


def pending_entry(dataset, root, trial, deployment):
    checkpoint = root / trial["task_id"]
    output = checkpoint / "empirical_bank"
    return dict(dataset=dataset, method=trial["method_label"], seed=trial["seed"],
                task_id=trial["task_id"], checkpoint=relative(checkpoint),
                operation="saved_checkpoint_inference_only_no_training",
                output=relative(output),
                draws=10, confidence_total_runs=51)


def observation(dataset, root, trial, sources, queue):
    directory = root / trial["task_id"]
    path = directory / "test_deployment" / "evaluation.json"
    deployment = read_json(path, sources)
    validate_evaluation(deployment, trial)
    if abs(deployment["heldout_gibbs_mc"]["accuracy"] - trial["test_accuracy"]) > 1e-12:
        raise ValueError("Selected summary and saved deployment inference disagree")
    config = deployment["evaluation_config"]
    if config["training_loss"] != "bounded_ce_smooth" or config["loss_epsilon"] != .01:
        raise ValueError("Do not combine different risk definitions in this table")
    empirical, empirical_source, corrected = None, None, None
    for candidate in (directory / "empirical_bank" / "evaluation.json",
                      directory / "test_risk_table" / "evaluation.json",
                      directory / "test_full" / "evaluation.json", path):
        if not candidate.is_file():
            continue
        full = deployment if candidate == path else read_json(candidate, sources)
        if candidate.parent.name == "empirical_bank":
            if (Path(full["checkpoint"]).name != trial["task_id"]
                    or not matching_config(full["training_config"], trial["config"])):
                raise ValueError("Bank inference does not identify the selected frozen checkpoint")
            allocation = full["confidence_allocation"]
            if (allocation.get("total_runs", allocation.get("simultaneous_runs")) != 51
                    or abs(allocation["posterior_failure_per_theorem_run"] - .04 / (2 * 51)) > 1e-12
                    or abs(allocation.get("mc_failure_per_run", allocation.get("monte_carlo_failure_per_run")) - .01 / 51) > 1e-12):
                raise ValueError("Bank report must cover the shared 51-run confidence family")
            corrected = {"allocation": allocation,
                         "posterior_complexities": full["final_posterior_complexities"],
                         "t1": full["theorem_1"], "t2": full["theorem_2"]}
        else:
            validate_evaluation(full, trial)
        if "empirical_gibbs_mc" not in full:
            continue
        bank = full["empirical_gibbs_mc"]
        if (bank["channel_source"] != "bank" or bank["examples"] != config["train_size"]
                or bank["draws"] != 10 or not 0 <= bank["bounded_loss"] <= 1
                or bank["bounded_loss_definition"] != deployment["heldout_gibbs_mc"]["bounded_loss_definition"]):
            raise ValueError("Empirical risk must evaluate the same loss on the complete reused training bank")
        empirical, empirical_source = bank["bounded_loss"], relative(candidate)
        break
    queue.append(pending_entry(dataset, root, trial, deployment))
    deterministic = config["method"] in {"erm", "l2"}
    terms = deployment["final_posterior_complexities"]
    if deterministic and terms:
        raise ValueError("Deterministic methods must not receive a finite Gaussian-posterior KL")
    shift = deployment["channel_shift_diagnostic"]["channel_shift_complexity"]
    floors = {} if deterministic else {
        "t1": terms["channel_complexity"] + terms["sample_complexity"] + shift,
        "t2": terms["joint_complexity"] + shift}
    return dict(task_id=trial["task_id"], seed=trial["seed"],
                source=relative(path), empirical_source=empirical_source,
                checkpoint_weights_available_locally=(directory / "weights.pt").is_file(),
                accuracy=deployment["heldout_gibbs_mc"]["accuracy"],
                test_bounded_risk=deployment["heldout_gibbs_mc"]["bounded_loss"],
                empirical_bounded_risk=empirical, deterministic=deterministic,
                nominal_delta=config["delta"], posterior_complexities=terms,
                corrected_bound_evaluation=corrected,
                bounded_loss_shift=shift, nominal_rhs_complexity_shift_floor=floors,
                nominal_risk_rhs={} if deterministic or empirical is None else
                    {key: empirical + value for key, value in floors.items()})


def aggregate(dataset, method, config, observations):
    deterministic = observations[0]["deterministic"]
    complete = all(row["empirical_bounded_risk"] is not None for row in observations)
    row = dict(dataset=dataset, method=method, display_method=LABELS[method],
               seeds=[item["seed"] for item in observations], n_train=config["train_size"],
               n_validation=config["val_size"], m=config["m"], width=config["width"],
               architecture=config.get("architecture", "mlp"), epochs=config["epochs"],
               accuracy=statistic(item["accuracy"] for item in observations),
               test_bounded_risk=statistic(item["test_bounded_risk"] for item in observations),
               empirical_available_count=sum(item["empirical_bounded_risk"] is not None for item in observations),
               empirical_bounded_risk=statistic(item["empirical_bounded_risk"] for item in observations)
                    if complete else None,
               observations=observations)
    for theorem in ("t1", "t2"):
        floor = None if deterministic else statistic(
            item["nominal_rhs_complexity_shift_floor"][theorem] for item in observations)
        known_vacuous = not deterministic and all(
            item["nominal_rhs_complexity_shift_floor"][theorem] > 1 for item in observations)
        nominal_value = None if deterministic or not complete else statistic(
            item["nominal_risk_rhs"][theorem] for item in observations)
        corrected_complete = not deterministic and all(
            item["corrected_bound_evaluation"] is not None for item in observations)
        value = statistic(item["corrected_bound_evaluation"][theorem]["population_risk_upper_uncapped"]
                          for item in observations) if corrected_complete else None
        status = ("not_applicable_deterministic_dirac" if deterministic else
                  "corrected_vacuous_risk_rhs" if corrected_complete and all(
                      item["corrected_bound_evaluation"][theorem]["population_risk_upper_uncapped"] > 1
                      for item in observations) else
                  "corrected_risk_rhs_available" if corrected_complete else
                  "vacuous_from_complexity_alone_empirical_pending" if known_vacuous and not complete else
                  "empirical_pending" if not complete else
                  "nominal_rhs_available_requires_confidence_correction")
        row[theorem] = dict(status=status, risk_rhs=value,
                            nominal_risk_rhs=nominal_value,
                            complexity_shift_floor=floor, known_vacuous_in_every_seed=known_vacuous)
    return row


def tex_stat(value, percent=False):
    if value is None:
        return "Pending"
    scale, digits = (100, 2) if percent else (1, 4)
    result = f"{value['mean'] * scale:.{digits}f}"
    if value["sd"] is not None:
        result += rf"\pm{value['sd'] * scale:.{digits}f}"
    return "$" + result + "$"


def tex_bound(row, theorem):
    value = row[theorem]
    if value["status"] == "not_applicable_deterministic_dirac":
        return "N/A"
    if value["risk_rhs"] is not None:
        return tex_stat(value["risk_rhs"])
    return "Vacuous" if value["known_vacuous_in_every_seed"] else "Pending"


def write_outputs(output, rows, queue, sources):
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "algorithm_risk_bounds"
    definitions = dict(
        predictor="Gibbs risk for Gaussian-posterior methods; deterministic predictor for ERM/L2. Each sampled weight is fixed for a full test pass.",
        loss="-log(0.01+0.99*p_y)/log(100), in [0,1]; all reported risks and RHSs use this same loss.",
        test_risk="Ten-draw held-out estimate on the official 10,000-example test set under deployment channels; not the known population risk.",
        empirical_risk="Same frozen checkpoint on all training examples with the saved reused channel bank; not the average training trajectory loss.",
        uncertainty="Mean +/- sample SD over all validation-selected initialization/channel-bank seeds, conditional on the fixed data.",
        nominal_rhs="Unweighted empirical risk + T1 channel/data or T2 joint complexity + bounded-loss channel shift at the saved nominal delta. Training eta coefficients never scale these terms.",
        corrected_risk_rhs="min(1, empirical MC estimate + its upper correction) + unweighted T1 channel/data or T2 joint complexity + bounded-loss shift, using overall delta .05 across 51 selected runs: theorem failure .04/(2*51), MC failure .01/51. Displayed before final clipping so vacuity is visible; final clipping to the loss range gives the trivial bound one whenever the RHS exceeds one.",
        complexity_shift_floor="Nonnegative part of the nominal risk-bound RHS before empirical risk. It is not a lower bound on true population risk.",
        bound_status="Vacuous means complexity+shift alone exceeds one for every seed. Missing empirical risk is not replaced by zero. N/A means a Dirac posterior has infinite KL to the continuous Gaussian prior.",
        certificate_limit="Nominal complexity terms remain diagnostics. Corrected rows include fresh bank-risk Monte Carlo control and the declared 51-run confidence allocation, conditional on theorem assumptions and the declared candidate family; they do not certify classification error or unbounded CE.",
        pending_queue="Run each entry's argument-vector command from repository root only after saved checkpoint files and DATASET are available. The bank evaluator has no training fallback.")
    pending_ids = {item["task_id"] for row in rows for item in row["observations"]
                   if item["empirical_bounded_risk"] is None}
    queue_path = prefix.with_name(prefix.name + "_queue.jsonl")
    for index, item in enumerate(queue):
        item["index"] = index
        item["command"] = ["conda", "run", "-n", "torch28", "python", "-m", "journal.evaluate_empirical_bank",
                           "--queue", relative(queue_path), "--index", str(index),
                           "--draws", "10", "--threads", "2"]
    payload = dict(schema_version=1, generated_by=relative(Path(__file__)),
                   definitions=definitions, rows=rows, pending_evaluations=len(pending_ids),
                   inference_family_runs=len(queue), source_hashes=sources)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        item = {key: row[key] for key in ("dataset", "method", "n_train", "n_validation", "m", "width", "architecture", "epochs", "empirical_available_count")}
        item["seeds"] = json.dumps(row["seeds"])
        for key in ("accuracy", "test_bounded_risk", "empirical_bounded_risk"):
            item[key + "_mean"] = None if row[key] is None else row[key]["mean"]
            item[key + "_sample_sd"] = None if row[key] is None else row[key]["sd"]
        for theorem in ("t1", "t2"):
            item[theorem + "_status"] = row[theorem]["status"]
            for key in ("risk_rhs", "nominal_risk_rhs", "complexity_shift_floor"):
                item[theorem + "_" + key + "_mean"] = (
                    None if row[theorem][key] is None else row[theorem][key]["mean"])
        flat.append(item)
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    queue_path.write_text("".join(json.dumps(item, allow_nan=False) + "\n" for item in queue))
    prefix.with_name(prefix.name + "_pending.jsonl").write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in queue if item["task_id"] in pending_ids))
    caption = (r"Test performance, reused-bank empirical risk, and bounds. Training counts are "
               r"$n=55{,}000$ for MNIST, $n=10{,}000$ for FashionMNIST, and $n=45{,}000$ for CIFAR-10; "
               r"all use $m=100$. Perfect denotes training only. Accuracy is in percent; risks use "
               r"$-\log(0.01+0.99p_y)/\log100$. Test risk is a held-out estimate. Entries are mean "
               r"$\pm$ sample SD over all selected seeds (three for MNIST/CIFAR-10, two for FashionMNIST), "
               r"with ten Gibbs/channel draws. Unweighted T1/T2 bounds include fresh empirical Monte Carlo "
               r"correction and total failure budget $0.05$: $0.04/(2\times51)$ per theorem/run and "
               r"$0.01/51$ per MC event. Values above one are vacuous; final clipping gives one. "
               r"N/A denotes deterministic ERM/L2, whose Dirac posterior has infinite KL to the continuous "
               r"Gaussian prior. Bounds concern smooth Gibbs loss under the theorem assumptions and declared confidence family.")
    if pending_ids:
        caption += (r" Pending entries require saved-checkpoint inference. ``Vacuous'' without a number "
                    r"means complexity plus shift alone exceeds one in every seed; the corrected bound is pending.")
    bound_headers = (r"\shortstack{T1 RHS/\\status} & \shortstack{T2 RHS/\\status}"
                     if pending_ids else r"T1 bound & T2 bound")
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             r"\caption{" + caption + "}",
             r"\label{tab:sim_algorithm_risk_bounds}",
             r"\label{tab:sim_accuracy_risk}", r"\label{tab:sim_cifar}",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"Dataset & Training method & \shortstack{Test accuracy\\(\%)} & \shortstack{Test risk\\estimate} & \shortstack{Bank empirical\\risk} & " + bound_headers + r"\\", r"\midrule"]
    previous = None
    for row in rows:
        if previous and previous != row["dataset"]:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([row["dataset"] if row["dataset"] != previous else "", row["display_method"],
                                 tex_stat(row["accuracy"], True), tex_stat(row["test_bounded_risk"]),
                                 tex_stat(row["empirical_bounded_risk"]), tex_bound(row, "t1"), tex_bound(row, "t2")]) + r"\\")
        previous = row["dataset"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, queue, sources = [], [], {}
    for dataset, directory, group in COHORTS:
        root = REPO / directory
        summary = read_json(root / "summary.json", sources)
        choices = [item for item in summary["selected_variants"]
                   if item.get("rank", 1) == 1 and item["group"] == group and item["method_label"] in METHODS]
        trials = {item["task_id"]: item for item in read_json(root / "test_results.json", sources)}
        for method in METHODS:
            selected = [item for item in choices if item["method_label"] == method]
            if not selected:
                if method == "L2-Perfect" and dataset != "CIFAR-10":
                    continue
                raise ValueError(f"Missing selected method: {dataset}/{method}")
            if len(selected) != 1:
                raise ValueError("Exactly one validation-selected configuration per method is required")
            choice = selected[0]
            selected_trials = sorted((trials[task] for task in choice["task_ids"]), key=lambda item: item["seed"])
            if (not choice.get("complete_seed_set", True)
                    or {item["seed"] for item in selected_trials} != set(choice["seeds"])
                    or any(item["test_status"] != "complete" for item in selected_trials)):
                raise ValueError("Do not omit selected seeds or incomplete evaluations")
            observations = [observation(dataset, root, trial, sources, queue) for trial in selected_trials]
            rows.append(aggregate(dataset, method, selected_trials[0]["config"], observations))
    write_outputs(args.output, rows, queue, sources)
    print(json.dumps(dict(output=str(args.output), algorithm_rows=len(rows),
                          inference_family_runs=len(queue),
                          pending_checkpoint_evaluations=sum(item["empirical_bounded_risk"] is None for row in rows for item in row["observations"]),
                          pending_weights_available_locally=sum(item["checkpoint_weights_available_locally"] and item["empirical_bounded_risk"] is None for row in rows for item in row["observations"]),
                          source_files=len(sources)), indent=2))


if __name__ == "__main__":
    main()
