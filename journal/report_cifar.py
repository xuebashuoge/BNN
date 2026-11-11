"""Summarize completed CIFAR campaigns without training or inference.

Only aggregate-validation-selected configurations are included in the table.
Every evaluation seed for each selected configuration is retained. Source
hashes allow the paper assets to be traced to the saved inference reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ["ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched",
           "ERM-Perfect", "L2-Perfect", "PAC-Perfect"]
LABELS = {"ERM-Matched": "ERM (matched)", "L2-Matched": "L2 (matched)",
          "T1-Matched": "Proposed T1 (matched)", "T2-Matched": "Proposed T2 (matched)",
          "ERM-Perfect": "ERM (perfect)", "L2-Perfect": "L2 (perfect)",
          "PAC-Perfect": "Proposed (perfect)"}
COLORS = dict(zip(METHODS, ["#777777", "#9b6a46", "#0072b2", "#009e73",
                           "#aaaaaa", "#c59c76", "#cc79a7"]))


def stats(values):
    values = list(values)
    if not values or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) for value in values):
        raise ValueError("Statistics require nonempty finite numeric observations")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None)


def save_text(path, value):
    """Publish report files atomically while another process may read them."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
                    "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


COHORT_FIELDS = ("dataset", "architecture", "train_size", "val_size", "width", "m", "epochs",
                 "batch_size", "deployment_channel", "data_seed", "training_loss", "loss_epsilon",
                 "clip_ce", "prior_std", "delta", "val_draws", "selection_metric", "validation_seed_policy",
                 "device", "cpu_flush_denormal")
SEED_FIELDS = {"seed", "channel_seed", "eval_seed", "device", "threads"}


def cohort(config):
    # Checkpoints predating the explicit CPU numerical policy used its False
    # default. Keep their reports readable while rejecting mixed policies.
    return {key: config.get(key, False) if key == "cpu_flush_denormal" else config[key]
            for key in COHORT_FIELDS}


def load_campaign(root, sources):
    root = Path(root)

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    summary = read(root / "summary.json")
    if (summary["completed_trials"] != summary["training_trials"] or
            summary.get("failed_trials") or summary["failed_evaluations"]):
        raise ValueError(f"Incomplete campaign: {root}")
    choices = [choice for choice in summary["selected_variants"] if choice.get("rank", 1) == 1]
    if len(choices) != len(METHODS) or {choice["method_label"] for choice in choices} != set(METHODS):
        raise ValueError("Expected one validation-selected configuration for each of seven methods")
    if any(not choice.get("complete_seed_set", True) for choice in choices):
        raise ValueError("Selected configurations must include every planned seed")
    selected_list = [task for choice in choices for task in choice["task_ids"]]
    if len(selected_list) != len(set(selected_list)):
        raise ValueError("A selected task cannot be counted more than once")
    selected = set(selected_list)
    rows = [row for row in read(root / "test_results.json") if row["task_id"] in selected]
    if len(rows) != len(selected) or any(row["test_status"] != "complete" for row in rows):
        raise ValueError("Every selected seed must have a completed evaluation")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task evaluations cannot replace a missing selected seed")
    if len({row["group"] for row in rows}) != 1:
        raise ValueError("A named campaign table column must contain one comparison group")
    if {row["method_label"] for row in rows} != set(METHODS):
        raise ValueError("Expected all seven matched/perfect methods")
    common = cohort(rows[0]["config"])
    if common["dataset"] != "cifar10":
        raise ValueError("This report is for CIFAR-10 campaigns")
    if any(cohort(row["config"]) != common for row in rows):
        raise ValueError("Do not pool distinct data, model, loss, channel or training-budget conditions")
    result = []
    for method in METHODS:
        runs = sorted((row for row in rows if row["method_label"] == method), key=lambda row: row["seed"])
        if len({run["variant_id"] for run in runs}) != 1:
            raise ValueError("Do not pool distinct hyperparameter configurations")
        if len({run["seed"] for run in runs}) != len(runs):
            raise ValueError("Each method must contribute exactly one observation per seed")
        fixed_config = {key: value for key, value in runs[0]["config"].items() if key not in SEED_FIELDS}
        if any({key: value for key, value in run["config"].items() if key not in SEED_FIELDS}
               != fixed_config for run in runs):
            raise ValueError("Configuration values disagree within a selected variant")
        choice = next(choice for choice in choices if choice["method_label"] == method)
        if set(choice.get("seeds", [])) != {run["seed"] for run in runs}:
            raise ValueError("Evaluation seed coverage disagrees with validation selection")
        observations = []
        for run in runs:
            directory = root / run["task_id"]
            report = read(directory / "test_deployment" / "evaluation.json")
            heldout = report["heldout_gibbs_mc"]
            if report.get("split") != "test" or heldout["examples"] != 10000:
                raise ValueError("Expected evaluation on the official 10,000-example test partition")
            for key in ("accuracy", "bounded_loss", "cross_entropy", "feature_power_per_coordinate"):
                value = heldout[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f"Nonfinite or nonnumeric evaluation metric: {key}")
                if value < 0 or (key in {"accuracy", "bounded_loss"} and value > 1):
                    raise ValueError(f"Evaluation metric outside its range: {key}")
            if abs(heldout["accuracy"] - run["test_accuracy"]) > 1e-12:
                raise ValueError("Summary and frozen evaluation disagree")
            if abs(heldout["cross_entropy"] - run["test_cross_entropy"]) > 1e-12:
                raise ValueError("Summary and frozen cross entropy disagree")
            if heldout["draws"] != 10 or not 0 <= heldout["bounded_loss"] <= 1:
                raise ValueError("Expected common ten-draw evaluation and bounded risk")
            evaluation_config = report["evaluation_config"]
            if any(evaluation_config[key] != run["config"][key]
                   for key in run["config"] if key not in
                   {"device", "threads", "batch_size", "download", "eval_draws", "eval_seed"}):
                raise ValueError("Frozen evaluation does not match the trained configuration")
            history_path = directory / "history.json"
            history = read(history_path)
            epochs = run["config"]["epochs"]
            if len(history) != epochs or [record["epoch"] for record in history] != list(range(1, epochs + 1)):
                raise ValueError("Training history must cover the complete configured epoch budget")
            if not 1 <= run["selected_epoch"] <= epochs or report["selected_epoch"] != run["selected_epoch"]:
                raise ValueError("Selected epoch disagrees with the trained checkpoint")
            if not all(math.isfinite(record["validation"]["accuracy"])
                       and 0 <= record["validation"]["accuracy"] <= 1 for record in history):
                raise ValueError("Validation trajectory contains an invalid accuracy")
            observations.append(dict(seed=run["seed"], task_id=run["task_id"],
                                     accuracy=heldout["accuracy"], bounded_loss=heldout["bounded_loss"],
                                     cross_entropy=heldout["cross_entropy"],
                                     feature_power_per_coordinate=heldout["feature_power_per_coordinate"],
                                     evaluation_draws=heldout["draws"], examples=heldout["examples"],
                                     checkpoint_git_sha=run.get("git_sha"),
                                     selected_epoch=run["selected_epoch"],
                                     history_source=str(history_path), history=history))
        result.append(dict(method=method, display=LABELS[method], config=runs[0]["config"],
                           accuracy=stats(x["accuracy"] for x in observations),
                           bounded_loss=stats(x["bounded_loss"] for x in observations),
                           cross_entropy=stats(x["cross_entropy"] for x in observations),
                           observations=observations))
    seed_sets = {tuple(x["seed"] for x in item["observations"]) for item in result}
    if len(seed_sets) != 1:
        raise ValueError("Methods must retain the same seed set")
    return result


def protocol(methods):
    config = methods[0]["config"]
    return {**cohort(config),
            "seeds": [row["seed"] for row in methods[0]["observations"]],
            "test_examples": methods[0]["observations"][0]["examples"],
            "evaluation_draws": methods[0]["observations"][0]["evaluation_draws"],
            "sd_scope": "Initialization/channel-bank variation conditional on the fixed data split"}


def compact_campaign(methods):
    """Keep original histories at hashed source paths instead of duplicating them."""
    return [{**item, "observations": [{key: value for key, value in row.items() if key != "history"}
                                      for row in item["observations"]]} for item in methods]


def tex_metric(item, key, percent=False):
    scale, digits = (100, 2) if percent else (1, 4)
    metric = item[key]
    value = f"{metric['mean'] * scale:.{digits}f}"
    if metric["sd"] is not None:
        value += rf"\pm{metric['sd'] * scale:.{digits}f}"
    return "$" + value + "$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, default=Path("results/journal_cifar/report"))
    args = parser.parse_args()
    sources, stages = {}, {}
    for specification in args.campaign:
        name, path = specification.split("=", 1)
        if name in stages:
            raise ValueError("Duplicate campaign name")
        stages[name] = load_campaign(path, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, methods in stages.items():
        for item in methods:
            config = item["config"]
            records.append(dict(stage=stage, method=item["method"],
                                n_train=config["train_size"], epochs=config["epochs"],
                                n_validation=config["val_size"], width=config["width"], m=config["m"],
                                architecture=config["architecture"],
                                seeds=[x["seed"] for x in item["observations"]],
                                accuracy_percent=100 * item["accuracy"]["mean"],
                                accuracy_sd_pp=100 * item["accuracy"]["sd"] if item["accuracy"]["sd"] is not None else None,
                                bounded_risk=item["bounded_loss"]["mean"],
                                bounded_risk_sd=item["bounded_loss"]["sd"],
                                selected_epochs=[x["selected_epoch"] for x in item["observations"]]))
    with (args.output / "cifar_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    protocols = {stage: protocol(methods) for stage, methods in stages.items()}
    descriptions = []
    for stage, value in protocols.items():
        descriptions.append(tex_escape(stage) + rf": $n={value['train_size']:,}$, "
                            rf"{value['epochs']} epochs, {len(value['seeds'])} paired seeds")
    caption = (r"CIFAR-10 deployment performance under Gaussian fading and additive noise. "
               r"Entries are mean $\pm$ sample standard deviation across initialization/channel-bank seeds "
               r"conditional on the fixed data split, using ten fresh Gibbs/channel draws per checkpoint. "
               + "; ".join(descriptions) + ". "
               + r"Configurations use aggregate validation accuracy; checkpoints use validation accuracy. "
               r"Perfect refers to the training forward channel only.")
    tex = [r"\begin{table*}[t]", r"\centering\small",
           r"\caption{" + caption + "}",
           r"\label{tab:sim_cifar}",
           r"\begin{tabular}{l" + "cc" * len(stages) + "}", r"\toprule"]
    tex.append("Method & " + " & ".join(r"\multicolumn{2}{c}{" + tex_escape(stage) + "}" for stage in stages) + r"\\")
    tex.append(" & " + " & ".join([r"Accuracy (\%) & Bounded risk"] * len(stages)) + r"\\\midrule")
    for index, method in enumerate(METHODS):
        cells = [LABELS[method]]
        for methods in stages.values():
            cells += [tex_metric(methods[index], "accuracy", True), tex_metric(methods[index], "bounded_loss")]
        tex.append(" & ".join(cells) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    save_text(args.output / "cifar_results.tex", "\n".join(tex) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(stages), 2, figsize=(10, 3.3 * len(stages)), squeeze=False)
    for index, (stage, methods) in enumerate(stages.items()):
        value = protocols[stage]
        for column, subset in enumerate((methods[:4], methods[4:])):
            ax = axes[index, column]
            for item in subset:
                trajectories = [[record["validation"]["accuracy"] for record in obs["history"]]
                                for obs in item["observations"]]
                means = [100 * statistics.fmean(vals) for vals in zip(*trajectories)]
                deviations = [100 * statistics.stdev(vals) if len(vals) > 1 else 0.
                              for vals in zip(*trajectories)]
                ax.plot(range(1, len(means) + 1), means, label=item["display"], color=COLORS[item["method"]])
                ax.fill_between(range(1, len(means) + 1),
                                [mean-deviation for mean, deviation in zip(means, deviations)],
                                [mean+deviation for mean, deviation in zip(means, deviations)],
                                color=COLORS[item["method"]], alpha=.10, linewidth=0)
            context = (f"n={value['train_size']:,}; d={value['width']}; m={value['m']}; "
                       f"{value['epochs']} epochs; {len(value['seeds'])} seeds")
            ax.set(title=f"{stage}: {'matched' if column == 0 else 'perfect'} training\n{context}",
                   xlabel="Epoch", ylabel="Validation accuracy (%)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.text(.5, .005, "Mean validation trajectories and one sample SD; fixed data split; selected configurations only",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(args.output / "cifar_validation.pdf", bbox_inches="tight")
    fig.savefig(args.output / "cifar_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_text(args.output / "cifar_results.json", json.dumps(dict(
        schema_version=2, campaigns={stage: compact_campaign(methods) for stage, methods in stages.items()},
        protocols=protocols, source_hashes=sources,
        selection="Aggregate-validation-selected configurations; every selected seed retained",
        uncertainty="Sample SD over initialization/channel-bank seeds, not MC draws or resampled datasets",
        history_policy="Full histories remain at their hashed source paths; they are not copied into this compact summary"),
        indent=2, allow_nan=False) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
