"""Compare Gibbs deployment accuracy for ERM, L2, T1 and T2 on diagonal fading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from itertools import product
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
                 "refined_confirm": (124101, 124102, 124103),
                 "cifar_balanced_confirm": (151101, 151102, 151103),
                 "moons_balanced_confirm": (162101, 162102, 162103),
                 "fashion_posterior_confirm": (161101, 161102, 161103),
                 "moons_revalidated_confirm": (163101, 163102, 163103),
                 "small_sample_confirm": (141101, 141102, 141103),
                 "small_sample_refined_confirm": (142101, 142102, 142103)}
SMALL_SAMPLE_SCREEN_SEEDS = (140101, 140102)


def candidates(base, method, perfect, refine):
    """Four candidates per label and round, with common learning rates across labels."""
    dataset = base["dataset"]
    rates = {"mnist": (.003, .01), "fashionmnist": (.001, .003),
             "cifar10": (.001, .003), "moons": (.01, .03)}[dataset]
    if refine:
        rates = {"mnist": (.006, .03), "fashionmnist": (.0006, .0015),
                 "cifar10": (.0003, .0006), "moons": (.003, .02)}[dataset]
    small_sample = dataset == "moons" and base["train_size"] != 800
    if small_sample:
        rates = (.006, .02) if not refine else (.01, .03)
    for rate in rates:
        for index in range(2):
            config = dict(base, lr=rate, method=method)
            if method == "erm":
                config["posterior_std"] = ((.03, .10) if small_sample else (.001, .003))[index]
            elif method == "l2":
                strengths = (1e-6, 1e-5) if not refine else (1e-7, 1e-4)
                if small_sample:
                    strengths = (1e-5, 1e-3) if not refine else (1e-4, 3e-3)
                config["l2"] = strengths[index]
            elif perfect:
                config.update(eta_channel=0., eta_data=(.003, .03)[index])
            else:
                strengths = (.0003, .03) if refine else (.001, .01)
                if refine and dataset == "fashionmnist":
                    strengths = (.03, .1)
                strength = strengths[index]
                if small_sample:
                    strength = ((.03, .3) if not refine else (.1, 1.))[index]
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


def screening_entries(datasets, m, rho_c, stage="screen", moons_train_size=800):
    if m < 2 or not 0 <= rho_c < 1:
        raise ValueError("Require m >= 2 and 0 <= rho_c < 1")
    deployment = dict(kind="diagonal_fading", rho_c=rho_c)
    entries = []
    if moons_train_size != 800 and (set(datasets) != {"moons"} or moons_train_size < 2):
        raise ValueError("A custom training size requires a moons-only comparison with at least two examples")
    for dataset in datasets:
        base = base_config(dataset, "analog", stage)
        # Use the larger validation-selected initial posterior scale from the
        # earlier i.i.d. analog study for FashionMNIST.
        base.update(m=m, deployment_channel=deployment, data_seed=20261118,
                    posterior_std=.003 if dataset == "fashionmnist" else .001,
                    val_draws=3, eval_draws=20)
        group = f"{dataset}_diagonal_fading_r{str(rho_c).replace('.', 'p')}_m{m}"
        seeds = SCREEN_SEEDS
        if moons_train_size != 800:
            base.update(train_size=moons_train_size, val_size=1000, batch_size=32,
                        posterior_std=.03, data_seed=20262220, val_draws=8)
            group += f"_n{moons_train_size}"
            seeds = SMALL_SAMPLE_SCREEN_SEEDS
        if stage == "pilot":
            seeds = (122000,)
        for label, (method, perfect) in METHODS.items():
            train_channel = dict(kind="perfect") if perfect else deployment
            variants = candidates(dict(base, train_channel=train_channel), method, perfect, stage == "refine")
            for config in variants:
                entries.extend(entry(config, seed, group, label, stage) for seed in seeds)
                if stage == "pilot":
                    break
    return entries


def conditions(entries):
    return sorted({(item["config"]["dataset"], item["config"]["m"],
                    item["config"]["deployment_channel"]["rho_c"],
                    item["config"]["train_size"] if item["config"]["dataset"] == "moons" else 800)
                   for item in entries})


def grid(design, stage):
    return [item for dataset, m, rho, n_moons in design
            for item in screening_entries((dataset,), m, rho, stage, n_moons)]


def confirmation_entries(selection):
    stage = selection["stage"]
    return [{**entry(dict(choice["config"], eval_draws=selection["eval_draws"]),
                     seed, choice["group"], choice["method_label"], stage),
             "selected_from_validation_variant": choice["variant_id"]}
            for choice in selection["selected_variants"] for seed in CONFIRM_SEEDS[stage]]


def validation_rows(root, entries):
    rows = read_results(root, entries)
    for row in rows:
        if (any(key.startswith("test_") for key in row)
                or not 0 <= row["validation_accuracy"] <= 1
                or not math.isfinite(row["validation_cross_entropy"]) or row["validation_cross_entropy"] < 0):
            raise ValueError(f"Selection requires successful validation-only observations: {row['task_id']}")
    return rows


def cifar_screen_bank_size(original):
    """Validate every configuration and seed in one complete CIFAR screen."""
    if not original:
        raise ValueError("Empty CIFAR screen")
    m = original[0]["config"]["m"]
    if not variant_grid(original, screening_entries(("cifar10",), m, .35)):
        raise ValueError("Require one complete CIFAR screen at a common m and rho=.35")
    return m


def balanced_cifar_entries(original, rows):
    """Carry each label's original validation-selected penalty into a shared 2x2 grid."""
    cifar_screen_bank_size(original)
    _, selected = select_variants(rows, original, top=1)
    by_id = {item["task_id"]: item for item in original}
    entries = []
    for choice in selected:
        config = by_id[choice["task_ids"][0]]["config"]
        for rate in (.0015, .002):
            for scale in (.001, .003):
                candidate = {**config, "lr": rate, "posterior_std": scale}
                for seed in SCREEN_SEEDS:
                    item = entry(candidate, seed, choice["group"], choice["method_label"], "cifar_balanced")
                    item["penalty_from_validation_variant"] = choice["variant_id"]
                    entries.append(item)
    return entries


def balanced_entries(original, rows):
    """Use an equal-budget scale/penalty grid for n=800 moons at each bank size."""
    design = conditions(original)
    if len(design) != 1 or design[0][0] != "moons":
        return balanced_cifar_entries(original, rows)
    dataset, m, rho, n = design[0]
    if n != 800 or rho != .35 or not variant_grid(original, grid(design, "screen")):
        raise ValueError("Require the original n=800, d=16, rho=.35 moons screen")
    ratio = math.sqrt((n-1)/(m-1))
    entries = []
    for label, (method, perfect) in METHODS.items():
        source = next(item for item in original if item["method_label"] == label)
        for rate in (.015, .04):
            for scale in (.003, .03, .1):
                for index in (0, 1):
                    config = dict(source["config"], lr=rate, posterior_std=scale)
                    strength = (.10, .60)[index]
                    if method == "erm":
                        config["rho_lr_scale"] = (.1, 1.)[index]
                    elif method == "l2":
                        config["l2"] = (1e-4, 1e-3)[index]
                    elif perfect:
                        config.update(eta_channel=0., eta_data=strength)
                    elif method == "proposed":
                        config.update(eta_channel=strength, eta_data=strength*ratio)
                    else:
                        config.update(eta_channel=0., eta_data=2*strength/math.sqrt(1+1/ratio**2))
                    entries.extend(entry(config, seed, source["group"], label, "moons_balanced")
                                   for seed in SCREEN_SEEDS)
    return entries


def fashion_posterior_entries(earlier, rows):
    """Cross learning rate, initial scale and scale learning rate for all seven labels."""
    if len(conditions(earlier)) != 1:
        raise ValueError("Select a single Fashion-MNIST bank size")
    dataset, m, rho, _ = conditions(earlier)[0]
    if dataset != "fashionmnist" or m not in (200, 500, 1000) or rho != .35:
        raise ValueError("Unexpected Fashion-MNIST bank-size comparison")
    _, selected = select_variants(rows, earlier, top=1)
    inputs = {item["task_id"]: item["config"] for item in earlier}
    result = []
    for choice in selected:
        source = inputs[choice["task_ids"][0]]
        for rate, std, scale_lr in product((.0008, .0012), (.001, .01), (0., .3)):
            config = {**source, "lr": rate, "posterior_std": std, "rho_lr_scale": scale_lr}
            for seed in SCREEN_SEEDS:
                item = entry(config, seed, choice["group"], choice["method_label"], "fashion_posterior")
                item["penalty_from_validation_variant"] = choice["variant_id"]
                result.append(item)
    return result


def prepare_confirmation(args):
    posterior_manifest = getattr(args, "posterior_manifest", None)
    posterior_root = getattr(args, "posterior_root", None)
    if (args.eval_draws < 1
            or (posterior_manifest is None) != (posterior_root is None)
            or (args.refinement_manifest is None) != (args.refinement_root is None)
            or (args.balanced_manifest is None) != (args.balanced_root is None)):
        raise ValueError("Require positive draws and a results root for every manifest")
    stages = [("screen", args.screen_manifest, args.screen_root)]
    if args.refinement_manifest:
        stages.append(("refine", args.refinement_manifest, args.refinement_root))
    if args.balanced_manifest:
        if not args.refinement_manifest:
            raise ValueError("Balanced confirmation includes the original screen and both refinements")
        dataset = read_manifest(args.screen_manifest)[0]["config"]["dataset"]
        balanced_stage = "moons_balanced" if dataset == "moons" else "cifar_balanced"
        stages.append((balanced_stage, args.balanced_manifest, args.balanced_root))
    if posterior_manifest:
        if not args.refinement_manifest or args.balanced_manifest:
            raise ValueError("Fashion posterior selection requires the original screen and refinement")
        stages.append(("fashion_posterior", posterior_manifest, posterior_root))
    entries, rows, design = [], [], None
    for stage, manifest, root in stages:
        additional = read_manifest(manifest)
        if design is None:
            design = conditions(additional)
        if stage == "fashion_posterior":
            expected = fashion_posterior_entries(entries, rows)
        elif stage in {"cifar_balanced", "moons_balanced"}:
            expected = balanced_grid
        else:
            expected = grid(design, stage)
        if not variant_grid(additional, expected):
            raise ValueError(f"Incomplete or altered {stage} tuning grid")
        entries.extend(additional)
        stage_rows = validation_rows(root, additional)
        rows.extend(stage_rows)
        if stage == "screen" and args.balanced_manifest:
            balanced_grid = balanced_entries(additional, stage_rows)
    if getattr(args, "command", None) == "posterior-screen":
        return write_manifest(args.output, fashion_posterior_entries(entries, rows))
    variants, selected = select_variants(rows, entries, top=1)
    accuracy = defaultdict(dict)
    for choice in selected:
        accuracy[choice["group"]][choice["method_label"]] = choice["validation_accuracy"]["mean"]
    margins = {group: max(scores["T1-Matched"], scores["T2-Matched"]) -
               max(score for label, score in scores.items() if label not in {"T1-Matched", "T2-Matched"})
               for group, scores in accuracy.items()}
    stage = "refined_confirm" if args.refinement_manifest else "confirm"
    if args.balanced_manifest:
        stage = balanced_stage + "_confirm"
    if posterior_manifest:
        stage = "fashion_posterior_confirm"
    if any(n_moons != 800 for _, _, _, n_moons in design):
        if len(design) != 1:
            raise ValueError("Select a separate confirmation cohort for each small-sample design")
        stage = "small_sample_" + stage
    selection = dict(stage=stage, eval_draws=args.eval_draws,
                     screening_seeds=sorted({row["seed"] for row in rows}),
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


def ranking(scores):
    proposed = ("T1-Matched", "T2-Matched")
    best = max(scores[label] for label in proposed)
    competitor = max(score for label, score in scores.items() if label not in proposed)
    return dict(best_matched_proposed=[label for label in proposed if abs(scores[label] - best) <= 1e-12],
                best_other_methods=sorted(label for label, score in scores.items()
                                          if label not in proposed and abs(score - competitor) <= 1e-12),
                margin=best - competitor, above_all_other_methods=best - competitor > 1e-12)


def report(manifest, root):
    selection_path = manifest.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    entries = read_manifest(manifest)
    if entries != confirmation_entries(selection):
        raise ValueError("Confirmation tasks differ from the validation selection")
    rows = [validate_test(item, row, root) for item, row in zip(entries, read_results(root, entries))]
    seeds = CONFIRM_SEEDS[selection["stage"]]
    groups = group_results(rows, seeds)
    for group in groups:
        subset = [row for row in rows if row["group"] == group["group"]]
        group["rho_c"] = subset[0]["config"]["deployment_channel"]["rho_c"]
        group["mean_ranking"] = ranking({label: metrics["test_accuracy"]["mean"]
                                         for label, metrics in group["methods"].items()})
        group["descriptive_ranking_by_seed"] = {
            seed: ranking({row["method_label"]: row["test_accuracy"] for row in subset if row["seed"] == seed})
            for seed in seeds}
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
    screen.add_argument("--moons-train-size", type=int, default=800,
                        help="Use a separate small-sample moons comparison when set below the reference size 800")
    screen.add_argument("--pilot", action="store_true")
    refine = commands.add_parser("refine-screen")
    refine.add_argument("--screen-manifest", type=Path, required=True)
    balanced = commands.add_parser("balanced-screen")
    balanced.add_argument("--screen-manifest", type=Path, required=True)
    balanced.add_argument("--screen-root", type=Path, required=True)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--screen-manifest", type=Path, required=True)
    confirm.add_argument("--screen-root", type=Path, required=True)
    confirm.add_argument("--refinement-manifest", type=Path)
    confirm.add_argument("--refinement-root", type=Path)
    confirm.add_argument("--balanced-manifest", type=Path)
    confirm.add_argument("--balanced-root", type=Path)
    confirm.add_argument("--posterior-manifest", type=Path)
    confirm.add_argument("--posterior-root", type=Path)
    posterior = commands.add_parser("posterior-screen")
    for tag in ("screen-manifest", "screen-root", "refinement-manifest", "refinement-root"):
        posterior.add_argument("--" + tag, type=Path, required=True)
    posterior.set_defaults(eval_draws=20, balanced_manifest=None, balanced_root=None)
    confirm.add_argument("--eval-draws", type=int, default=20)
    combine = commands.add_parser("combine")
    combine.add_argument("--summary", type=Path, action="append", required=True)
    for command in (screen, refine, balanced, posterior, confirm, combine):
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
                                    "pilot" if args.pilot else "screen", args.moons_train_size)
        result = write_manifest(args.output, entries)
    elif args.command == "refine-screen":
        entries = read_manifest(args.screen_manifest)
        design = conditions(entries)
        if not variant_grid(entries, grid(design, "screen")):
            raise ValueError("Refinement requires a complete original screen design")
        result = write_manifest(args.output, grid(design, "refine"))
    elif args.command == "balanced-screen":
        original = read_manifest(args.screen_manifest)
        entries = balanced_entries(original, validation_rows(args.screen_root, original))
        result = write_manifest(args.output, entries)
    elif args.command == "task":
        result = run_task(args)
    elif args.command in {"confirm", "posterior-screen"}:
        result = prepare_confirmation(args)
    elif args.command == "combine":
        result = combine_reports(args.summary, args.output)
    else:
        result = report(args.manifest, args.output_root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
