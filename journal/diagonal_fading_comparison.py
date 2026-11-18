"""Compare Gibbs deployment accuracy for ERM, L2, T1 and T2 on diagonal fading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path

from .build_bayesian_campaign import DATASETS, base_config
from .build_campaign import make_entry
from .campaign import evaluate_trial, save_json, select_variants, train_trial
from .full_fading_comparison import (
    IMAGE_DATASETS, METHODS, combine_reports, export_accuracy_means, group_results,
    read_results, report_file, scope, validate_test, variant_grid, write_csv,
    write_manifest, write_once,
)
from .run_experiment import file_fingerprint
from .run_task import read_manifest, verify_submission_snapshot


SCREEN_SEEDS = (122101, 122102)
CONFIRM_SEEDS = {"confirm": (123101, 123102, 123103),
                 "refined_confirm": (124101, 124102, 124103)}


def candidates(base, method, perfect, refine):
    """Four candidates per label and round, with common learning rates across labels."""
    dataset = base["dataset"]
    rates = {"mnist": (.003, .01), "fashionmnist": (.001, .003),
             "cifar10": (.001, .003), "moons": (.01, .03)}[dataset]
    if refine:
        rates = {"mnist": (.006, .03), "fashionmnist": (.006, .01),
                 "cifar10": (.0003, .0006), "moons": (.003, .02)}[dataset]
    for rate in rates:
        for index in range(2):
            config = dict(base, lr=rate, method=method)
            if method == "erm":
                config["posterior_std"] = (.001, .003)[index]
            elif method == "l2":
                config["l2"] = (1e-6, 1e-5)[index] if not refine else (1e-7, 1e-4)[index]
            elif perfect:
                config.update(eta_channel=0., eta_data=(.003, .03)[index])
            else:
                strength = ((.001, .01) if not refine else (.0003, .03))[index]
                ratio = math.sqrt((config["train_size"] - 1) / (config["m"] - 1))
                if method == "proposed":
                    config.update(eta_channel=strength, eta_data=strength * ratio)
                else:
                    config.update(eta_channel=0., eta_data=2 * strength / math.sqrt(1 + 1 / ratio**2))
            yield config


def entry(config, seed, group, label, stage):
    item = make_entry(config, seed, group, label, "diagonal_fading_accuracy_" + stage)
    item["family"] = "proposed" if config["method"] in {"proposed", "joint"} else "baseline"
    return item


def screening_entries(datasets, m, rho_c, stage="screen"):
    if m < 2 or not 0 <= rho_c < 1:
        raise ValueError("Require m >= 2 and 0 <= rho_c < 1")
    deployment = dict(kind="diagonal_fading", rho_c=rho_c)
    entries = []
    for dataset in datasets:
        base = base_config(dataset, "analog", stage)
        # Use the larger validation-selected initial posterior scale from the
        # earlier i.i.d. analog study for FashionMNIST.
        base.update(m=m, deployment_channel=deployment, data_seed=20261118,
                    posterior_std=.003 if dataset == "fashionmnist" else .001,
                    val_draws=3, eval_draws=20)
        group = f"{dataset}_diagonal_fading_r{str(rho_c).replace('.', 'p')}_m{m}"
        for label, (method, perfect) in METHODS.items():
            train_channel = dict(kind="perfect") if perfect else deployment
            variants = candidates(dict(base, train_channel=train_channel), method, perfect, stage == "refine")
            for config in variants:
                seeds = (122000,) if stage == "pilot" else SCREEN_SEEDS
                entries.extend(entry(config, seed, group, label, stage) for seed in seeds)
                if stage == "pilot":
                    break
    return entries


def conditions(entries):
    return sorted({(item["config"]["dataset"], item["config"]["m"],
                    item["config"]["deployment_channel"]["rho_c"]) for item in entries})


def grid(design, stage):
    return [item for dataset, m, rho in design for item in screening_entries((dataset,), m, rho, stage)]


def confirmation_entries(selection):
    stage = selection["stage"]
    return [{**entry(dict(choice["config"], eval_draws=selection["eval_draws"]),
                     seed, choice["group"], choice["method_label"], stage),
             "selected_from_validation_variant": choice["variant_id"]}
            for choice in selection["selected_variants"] for seed in CONFIRM_SEEDS[stage]]


def prepare_confirmation(args):
    if args.eval_draws < 1 or (args.refinement_manifest is None) != (args.refinement_root is None):
        raise ValueError("Require positive draws and a results root for every manifest")
    stages = [("screen", args.screen_manifest, args.screen_root)]
    if args.refinement_manifest:
        stages.append(("refine", args.refinement_manifest, args.refinement_root))
    entries, rows, design = [], [], None
    for stage, manifest, root in stages:
        additional = read_manifest(manifest)
        if design is None:
            design = conditions(additional)
        if not variant_grid(additional, grid(design, stage)):
            raise ValueError(f"Incomplete or altered {stage} tuning grid")
        entries.extend(additional)
        rows.extend(read_results(root, additional))
    for row in rows:
        if (any(key.startswith("test_") for key in row)
                or not 0 <= row["validation_accuracy"] <= 1
                or not math.isfinite(row["validation_cross_entropy"]) or row["validation_cross_entropy"] < 0):
            raise ValueError(f"Selection requires successful validation-only observations: {row['task_id']}")
    variants, selected = select_variants(rows, entries, top=1)
    accuracy = defaultdict(dict)
    for choice in selected:
        accuracy[choice["group"]][choice["method_label"]] = choice["validation_accuracy"]["mean"]
    margins = {group: max(scores["T1-Matched"], scores["T2-Matched"]) -
               max(score for label, score in scores.items() if label not in {"T1-Matched", "T2-Matched"})
               for group, scores in accuracy.items()}
    selection = dict(stage="refined_confirm" if args.refinement_manifest else "confirm",
                     eval_draws=args.eval_draws, screening_seeds=SCREEN_SEEDS,
                     sources={stage: file_fingerprint(manifest) for stage, manifest, _ in stages},
                     validation_trials=variants, selected_variants=selected, validation_proposed_margins=margins,
                     selection_rule="Highest mean validation accuracy, then lowest mean validation CE, then variant ID")
    write_once(args.output.with_suffix(".selection.json"), json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return {**write_manifest(args.output, confirmation_entries(selection)), "validation_proposed_margins": margins}


def run_task(args):
    verify_submission_snapshot(args.manifest)
    entries = read_manifest(args.manifest)
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(entries) or args.threads < 1:
        raise ValueError("Invalid task index or thread count")
    item = entries[index]
    result = report_file(args.output_root, item)
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = train_trial(item, args.output_root, args.threads, resume=True, device=args.device)
        save_json(result, row)
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
        if item["stage"] in {"diagonal_fading_accuracy_" + stage for stage in CONFIRM_SEEDS}:
            row = evaluate_trial(row, item["config"]["eval_draws"], full_risks=False, resume=True)
            save_json(result, row)
            if row["test_status"] != "complete":
                raise RuntimeError(row["test_error"])
            validate_test(item, row, args.output_root)
    return {key: row[key] for key in ("task_id", "validation_accuracy", "test_accuracy", "seconds") if key in row}


def report(manifest, root):
    selection_path = manifest.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    entries = read_manifest(manifest)
    if entries != confirmation_entries(selection):
        raise ValueError("Confirmation tasks differ from the validation selection")
    rows = [validate_test(item, row, root) for item, row in zip(entries, read_results(root, entries))]
    seeds = CONFIRM_SEEDS[selection["stage"]]
    groups = group_results(rows, seeds)
    summary = dict(trials=len(rows), groups=groups, confirmation_seeds=seeds, stage=selection["stage"],
                   selection=file_fingerprint(selection_path), manifest=file_fingerprint(manifest),
                   validation_proposed_margins=selection["validation_proposed_margins"],
                   all_groups_t1_or_t2_strictly_highest=all(g["t1_or_t2_strictly_highest"] for g in groups),
                   sampling="Fresh training, bank and deployment randomness; fixed official test examples")
    write_csv(root / "accuracy_by_seed.csv", rows,
              ["dataset", "channel", "m", "method_label", "seed", "variant_id", "validation_accuracy",
               "test_accuracy", "test_cross_entropy", "test_bounded_loss", "task_id"])
    export_accuracy_means(groups, root)
    save_json(root / "accuracy_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--datasets", default=",".join(IMAGE_DATASETS))
    screen.add_argument("--m", type=int, default=2000)
    screen.add_argument("--rho-c", type=float, default=.35)
    screen.add_argument("--pilot", action="store_true")
    refine = commands.add_parser("refine-screen")
    refine.add_argument("--screen-manifest", type=Path, required=True)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--screen-manifest", type=Path, required=True)
    confirm.add_argument("--screen-root", type=Path, required=True)
    confirm.add_argument("--refinement-manifest", type=Path)
    confirm.add_argument("--refinement-root", type=Path)
    confirm.add_argument("--eval-draws", type=int, default=20)
    combine = commands.add_parser("combine")
    combine.add_argument("--summary", type=Path, action="append", required=True)
    for command in (screen, refine, confirm, combine):
        command.add_argument("--output", type=Path, required=True)
    task = commands.add_parser("task")
    task.add_argument("--index", type=int)
    task.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    task.add_argument("--threads", type=int, default=2)
    for command in (task, commands.add_parser("report")):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "screen":
        entries = screening_entries(scope(args.datasets, DATASETS), args.m, args.rho_c,
                                    "pilot" if args.pilot else "screen")
        result = write_manifest(args.output, entries)
    elif args.command == "refine-screen":
        entries = read_manifest(args.screen_manifest)
        design = conditions(entries)
        if not variant_grid(entries, grid(design, "screen")):
            raise ValueError("Refinement requires a complete original screen design")
        result = write_manifest(args.output, grid(design, "refine"))
    elif args.command == "task":
        result = run_task(args)
    elif args.command == "confirm":
        result = prepare_confirmation(args)
    elif args.command == "combine":
        result = combine_reports(args.summary, args.output)
    else:
        result = report(args.manifest, args.output_root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
