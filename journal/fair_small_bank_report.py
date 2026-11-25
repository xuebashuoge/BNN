"""Verify and publish every outcome of the fixed, paired small-bank campaign."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import re
import shutil
from statistics import fmean

from .campaign import describe, save_json, validate_entries
from .full_fading_comparison import read_results, validate_test
from .run_experiment import file_fingerprint, normalize_config
from .run_task import read_manifest


LABELS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched", "Proposed-Perfect")
DATASETS = ("moons", "mnist", "fashionmnist", "cifar10")
FILES = ("weights.pt", "metadata.json", "task_config.json", "manifest_entry.json",
         "history.json", "test_deployment/evaluation.json")
METHODS = dict(zip(LABELS, ("erm", "l2", "proposed", "joint", "proposed")))
PROTECTED = Path(__file__).resolve().parents[1] / "results" / "final_chosen_results"


def safe_destination(path):
    resolved = Path(path).resolve()
    if resolved == PROTECTED.resolve() or PROTECTED.resolve() in resolved.parents:
        raise ValueError("The frozen final_chosen_results directory cannot be modified")
    return resolved


def target_file(destination, relative):
    target = destination / relative
    resolved = safe_destination(target)
    if destination not in resolved.parents:
        raise ValueError(f"Output escapes destination through a symlink: {target}")
    return target


def copy_verified(source, target):
    expected = file_fingerprint(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_fingerprint(target) != expected:
            raise ValueError(f"Refusing to replace a different archived artifact: {target}")
    else:
        temporary = target.with_name(target.name + ".copying")
        if temporary.is_symlink():
            raise ValueError(f"Unsafe copy temporary: {temporary}")
        shutil.copyfile(source, temporary)
        if file_fingerprint(temporary) != expected:
            raise ValueError(f"Artifact changed while being copied: {source}")
        temporary.replace(target)
    return expected


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.is_symlink():
        raise ValueError(f"Unsafe CSV temporary: {temporary}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(destination, name, value):
    path = target_file(destination, name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.is_symlink():
        raise ValueError(f"Unsafe JSON temporary: {temporary}")
    save_json(path, value)


def exact_sign_flip(differences):
    """Two-sided paired mean test, enumerating all 2**12 sign assignments."""
    if len(differences) != 12 or not all(math.isfinite(x) for x in differences):
        raise ValueError("The exact test requires twelve finite paired differences")
    observed = abs(math.fsum(differences))
    tolerance = 1e-12 * max(1., observed)
    extreme = sum(abs(math.fsum(s * d for s, d in zip(signs, differences))) >= observed - tolerance
                  for signs in product((-1., 1.), repeat=12))
    return extreme / 4096


def holm(rows):
    adjusted = 0.
    for index, row in enumerate(sorted(rows, key=lambda item: item["sign_flip_p"])):
        adjusted = max(adjusted, min(1., (len(rows) - index) * row["sign_flip_p"]))
        row.update(holm_p=adjusted, holm_family_size=len(rows),
                   significant_at_005=adjusted <= .05)


def validate_design(entries, selection):
    validate_entries(entries)
    seeds = selection["confirmation_seeds"]
    if (len(seeds) != 12 or len(set(seeds)) != 12
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise ValueError("Require twelve distinct integer confirmation seeds")
    if not re.fullmatch(r"[0-9a-f]{64}", selection["screen_manifest_sha256"]):
        raise ValueError("Missing screening-manifest SHA256")
    choices = {(row["group"], row["method_label"]): row for row in selection["selected_variants"]}
    pairs = {row["group"]: row for row in selection["primary_pairs"]}
    if len(choices) != 80 or len(selection["selected_variants"]) != 80 or len(pairs) != 16 or len(selection["primary_pairs"]) != 16:
        raise ValueError("Require exactly eighty validation selections and sixteen primary pairs")
    groups, seen, conditions = {}, set(), {}
    for item in entries:
        config = normalize_config(item["config"])
        group, label, seed = item["group"], item["method_label"], config["seed"]
        key = group, label, seed
        if key in seen or label not in LABELS or seed not in seeds:
            raise ValueError(f"Duplicate or undeclared confirmation cell: {key}")
        seen.add(key)
        channel = config["deployment_channel"]
        valid_channel = (channel == {"kind": "outage", "outage_prob": .5}
                         or channel == {"kind": "diagonal_fading", "rho_c": .35})
        if not valid_channel or config["method"] != METHODS[label] or config["predictor"] != "gaussian":
            raise ValueError(f"Unexpected channel or predictor: {key}")
        training = {"kind": "perfect"} if label == "Proposed-Perfect" else channel
        if config["train_channel"] != training:
            raise ValueError(f"Training condition differs from method label: {key}")
        choice = choices[group, label]
        if (item.get("selected_from_validation_variant", item["variant_id"]) != choice["variant_id"]
                or set(choice["seeds"]) & set(seeds)):
            raise ValueError(f"Changed selection or screening/confirmation seed overlap: {key}")
        selected_config = normalize_config(choice["config"])
        varying = {"seed", "channel_seed", "eval_seed", "device", "threads"}
        if any(config[name] != selected_config[name] for name in config if name not in varying):
            raise ValueError(f"Confirmation hyperparameters differ from frozen validation selection: {key}")
        if config["channel_seed"] != seed + 10000 or config["eval_seed"] != seed + 20000:
            raise ValueError(f"Channel/evaluation streams differ from paired seed schedule: {key}")
        condition = config["dataset"], channel["kind"], config["m"]
        if condition in conditions and conditions[condition] != group:
            raise ValueError("Two groups represent the same scientific condition")
        conditions[condition] = group
        groups.setdefault(group, []).append(item)
    expected_conditions = set(product(DATASETS, ("outage", "diagonal_fading"), (50, 100)))
    if set(conditions) != expected_conditions or set(pairs) != set(groups) or len(entries) != 960:
        raise ValueError("Require the complete four-dataset, two-channel, two-bank matrix (960 tasks)")
    if seen != set(product(groups, LABELS, seeds)) or set(choices) != set(product(groups, LABELS)):
        raise ValueError("Missing paired method/seed or validation-selection cell")
    for group, items in groups.items():
        fixed = ("epochs", "eval_draws", "val_draws", "batch_size", "data_seed", "train_size", "val_size")
        if len({tuple(normalize_config(item["config"])[k] for k in fixed) for item in items}) != 1:
            raise ValueError(f"Unequal optimization/data/evaluation budget within {group}")
        for label in LABELS:
            if len({item["variant_id"] for item in items if item["method_label"] == label}) != 1:
                raise ValueError(f"Multiple configurations confirmed for {group}, {label}")
        def order(label):
            choice = choices[group, label]
            accuracy, loss = choice["validation_accuracy"]["mean"], choice["validation_cross_entropy"]["mean"]
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not math.isfinite(loss) or loss < 0:
                raise ValueError("Invalid aggregate validation score")
            return -accuracy, loss, label
        if (pairs[group]["proposed"] != min(LABELS[2:4], key=order)
                or pairs[group]["baseline"] != min(LABELS[:2], key=order)):
            raise ValueError(f"Primary pair is not the frozen validation-selected pair: {group}")
    return sorted(seeds), groups, pairs


def report(manifest, output_root, selection_path, destination, copy_models=False):
    destination = safe_destination(destination)
    entries = read_manifest(manifest)
    selection = json.loads(selection_path.read_text())
    seeds, groups, pairs = validate_design(entries, selection)
    rows = read_results(output_root, entries)
    verified, archive, bank_hashes = [], [], {}
    manifest_sha256 = file_fingerprint(manifest)["sha256"]
    for item, row in zip(entries, rows):
        if row.get("source_manifest_sha256") != manifest_sha256:
            raise ValueError(f"Result belongs to a different manifest: {item['task_id']}")
        verified.append(validate_test(item, row, output_root))
        source = output_root / item["task_id"]
        evaluation = json.loads((source / FILES[-1]).read_text())
        if evaluation["checkpoint_source_fingerprints"]["weights"] != file_fingerprint(source / "weights.pt"):
            raise ValueError(f"Weights differ from the evaluated checkpoint: {item['task_id']}")
        for name in FILES:
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
        if item["method_label"] != "Proposed-Perfect":
            metadata = json.loads((source / "metadata.json").read_text())
            pair = item["group"], item["config"]["seed"]
            bank_hash = metadata["training_channel_bank_hash"]
            if pair in bank_hashes and bank_hashes[pair] != bank_hash:
                raise ValueError(f"Matched algorithms did not share the same finite channel bank: {pair}")
            bank_hashes[pair] = bank_hash
    scores = {(row["group"], row["method_label"], row["seed"]): row["test_accuracy"] * 100 for row in verified}
    means, observations, secondary, primary = [], [], [], []
    for row in verified:
        observations.append({key: row[key] for key in ("group", "dataset", "channel", "m", "method_label", "seed", "task_id", "variant_id")}
                            | dict(accuracy_pct=100 * row["test_accuracy"], bounded_loss=row["test_bounded_loss"],
                                   cross_entropy=row["test_cross_entropy"], validation_accuracy_pct=100 * row["validation_accuracy"]))
    for group in sorted(groups):
        config = groups[group][0]["config"]
        context = dict(group=group, dataset=config["dataset"], channel=config["deployment_channel"]["kind"], m=config["m"])
        for label in LABELS:
            stats = describe(scores[group, label, seed] for seed in seeds)
            means.append(context | dict(method_label=label, n=stats["n"], mean_accuracy_pct=stats["mean"], sd_accuracy_pp=stats["sd"]))
        for proposed, baseline in product(LABELS[2:4], LABELS[:2]):
            differences = [scores[group, proposed, seed] - scores[group, baseline, seed] for seed in seeds]
            stats = describe(differences)
            contrast = context | dict(proposed=proposed, baseline=baseline, n=len(seeds), mean_gap_pp=fmean(differences),
                                      paired_sd_pp=stats["sd"], ci95_low_pp=stats["ci95_low"], ci95_high_pp=stats["ci95_high"],
                                      wins=sum(x > 0 for x in differences), ties=sum(x == 0 for x in differences),
                                      losses=sum(x < 0 for x in differences), sign_flip_p=exact_sign_flip(differences))
            secondary.append(contrast)
            if (proposed, baseline) == (pairs[group]["proposed"], pairs[group]["baseline"]):
                primary.append(dict(contrast))
    holm(secondary)
    holm(primary)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, records in (("accuracy_by_seed.csv", observations), ("accuracy_means.csv", means),
                              ("secondary_contrasts.csv", secondary), ("primary_contrasts.csv", primary)):
        write_csv(target_file(destination, filename), records)
    copy_verified(manifest, target_file(destination, "confirm.jsonl"))
    copy_verified(selection_path, target_file(destination, "selection.json"))
    if copy_models:
        for item in entries:
            for name in FILES:
                relative = Path("models") / item["task_id"] / name
                fingerprint = copy_verified(output_root / item["task_id"] / name, target_file(destination, relative))
                archive.append(dict(path=str(relative), **fingerprint))
        write_json(destination, "model_files.json", archive)
    lookup = {(row["group"], row["method_label"]): row for row in means}
    tex = [r"\begin{tabular}{lllrrrrr}", r"\hline",
           r"Dataset & Channel & $m$ & ERM & L2 & T1 & T2 & Perfect \\", r"\hline"]
    for group in sorted(groups):
        config = groups[group][0]["config"]
        channel = "Erasure" if config["deployment_channel"]["kind"] == "outage" else "Fading"
        cells = [f"${lookup[group, label]['mean_accuracy_pct']:.3f} \\pm {lookup[group, label]['sd_accuracy_pp']:.3f}$" for label in LABELS]
        tex.append(" & ".join([config["dataset"], channel, str(config["m"]), *cells]) + r" \\")
    tex.extend([r"\hline", r"\end{tabular}", "% Accuracy percent +/- sample SD across all 12 paired training seeds; no significance bolding."])
    target_file(destination, "accuracy_table.tex").write_text("\n".join(tex) + "\n")
    summary = dict(tasks=len(rows), groups=len(groups), methods=list(LABELS), confirmation_seeds=seeds,
                   manifest=file_fingerprint(manifest), selection=file_fingerprint(selection_path),
                   screen_manifest_sha256=selection["screen_manifest_sha256"],
                   primary_family_size=len(primary), secondary_family_size=len(secondary),
                   sign_flip_patterns=4096, minimum_two_sided_sign_flip_p=2 / 4096,
                   primary_positive_holm_significant=sum(r["mean_gap_pp"] > 0 and r["significant_at_005"] for r in primary),
                   secondary_positive_holm_significant=sum(r["mean_gap_pp"] > 0 and r["significant_at_005"] for r in secondary),
                   models_copied=copy_models, model_file_count=len(archive),
                   caveats=["Every declared condition and seed is retained; the campaign has a fixed stopping point.",
                            "Inference is conditional on fixed data splits and the reused official test examples.",
                            "Training seeds pair initialization, channel banks and evaluation streams across methods.",
                            "Student-t 95% paired intervals are pointwise and unadjusted; they assume approximately normal seed effects.",
                            "Exact sign-flip inference assumes joint invariance to independent sign flips under the null.",
                            "Holm adjustments cover separate prespecified primary (16) and secondary (64) families.",
                            "The primary method pair is fixed by validation; table rankings are descriptive.",
                            "Finite posterior/channel Monte Carlo error is included in each seed outcome.",
                            "Copied records retain original scratch paths for provenance; local models live under models/."])
    write_json(destination, "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "output-root", "selection", "destination"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--copy-models", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report(args.manifest, args.output_root, args.selection, args.destination, args.copy_models), allow_nan=False))


if __name__ == "__main__":
    main()
