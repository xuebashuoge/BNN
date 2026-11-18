"""Transfer a common budget of mild-law candidates and select on harder-law validation."""
import argparse
import json
import math
from pathlib import Path

from .campaign import select_variants
from .diagonal_fading_comparison import METHODS, entry, validation_rows
from .fading_severity import HARD
from .full_fading_comparison import write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest


SCREEN_SEEDS = (314101, 314102, 314103)
CONFIRM_SEEDS = tuple(range(341201, 341210))
TUNABLES = ("lr", "posterior_std", "prior_std", "rho_lr_scale", "l2", "eta_data", "eta_channel")


def validation_rank(summaries, label):
    eligible = [x for x in summaries if x["method_label"] == label]
    if any(not (0 <= x["validation_accuracy"]["mean"] <= 1 and
                math.isfinite(x["validation_cross_entropy"]["mean"])) for x in eligible):
        raise ValueError("Invalid source validation metrics")
    return sorted(eligible, key=lambda x: (-x["validation_accuracy"]["mean"],
                                          x["validation_cross_entropy"]["mean"], x["variant_id"]))


def candidates(selection, primary):
    selected = {x["method_label"]: x for x in selection["selected_variants"]}
    templates = {x["method_label"]: x for x in primary}
    if set(templates) != set(METHODS) or set(selected) != set(METHODS):
        raise ValueError("Exactly seven original and transferred methods are required")
    conditions = {(x["config"]["dataset"], x["config"]["m"], x["group"]) for x in primary}
    if len(conditions) != 1:
        raise ValueError("Select one matched comparison condition at a time")
    _, _, group = conditions.pop()
    prelim = selection.get("finalist_protocol", {}).get("preliminary_validation_trials", [])
    result, shortlist = [], []
    for label, winner in selected.items():
        template = templates[label]["config"]
        if template["deployment_channel"] != HARD:
            raise ValueError("Expected the harder deployment law")
        if any(template[key] != winner["config"][key] for key in TUNABLES):
            raise ValueError("Primary transfer differs from the mild-law validation selection")
        ordered = [winner, *validation_rank(selection["validation_trials"], label), *validation_rank(prelim, label)]
        seen = set()
        for choice in ordered:
            settings = tuple(choice["config"][key] for key in TUNABLES)
            if settings in seen:
                continue
            seen.add(settings)
            config = {**template, **dict(zip(TUNABLES, settings))}
            for seed in SCREEN_SEEDS:
                item = entry(config, seed, group, label, "severity_validation")
                item.update(role="comparison", source_validation_variant=choice["variant_id"])
                result.append(item)
            shortlist.append(dict(method_label=label, source_variant_id=choice["variant_id"],
                                  severity_variant_id=item["variant_id"], source_rank=len(seen)))
            if len(seen) == 4:
                break
        if len(seen) != 4:
            raise ValueError(f"Four distinct source validation configurations required for {label}")
    return result, shortlist


def prepare_screen(args):
    selected = json.loads(args.source_selection.read_text())
    primary = read_manifest(args.primary)
    jobs, shortlist = candidates(selected, primary)
    if {x["config"]["seed"] for x in primary} & (set(SCREEN_SEEDS) | set(CONFIRM_SEEDS)):
        raise ValueError("New seeds must be disjoint from the primary cohort")
    protocol = dict(candidate_count_per_label=4, shortlist=shortlist, screening_seeds=SCREEN_SEEDS,
                    confirmation_seeds=CONFIRM_SEEDS, primary=file_fingerprint(args.primary),
                    source_selection=file_fingerprint(args.source_selection),
                    rule="Transferred winner plus up to three distinct alternatives by original validation rank; use the preliminary rank only to fill missing alternatives")
    result = write_manifest(args.output, jobs)
    protocol["manifest"] = file_fingerprint(args.output)
    write_once(args.output.with_suffix(".protocol.json"), json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return result


def prepare_confirmation(args):
    entries = read_manifest(args.screen)
    protocol = json.loads(args.screen.with_suffix(".protocol.json").read_text())
    if protocol["manifest"] != file_fingerprint(args.screen):
        raise ValueError("Screening manifest fingerprint changed")
    expected = {(x["method_label"], x["severity_variant_id"], s)
                for x in protocol["shortlist"] for s in SCREEN_SEEDS}
    actual = {(x["method_label"], x["variant_id"], x["config"]["seed"]) for x in entries}
    if len(entries) != 84 or actual != expected or set(protocol["confirmation_seeds"]) != set(CONFIRM_SEEDS):
        raise ValueError("Screen differs from the frozen equal-budget design")
    rows = validation_rows(args.runs, entries)
    variants, winners = select_variants(rows, entries, top=1)
    if len(variants) != 28 or {w["method_label"] for w in winners} != set(METHODS):
        raise ValueError("Every label requires four complete validation variants")
    jobs = []
    for w in winners:
        if tuple(w["seeds"]) != SCREEN_SEEDS:
            raise ValueError("Missing hard validation seed")
        for seed in CONFIRM_SEEDS:
            trial = entry(w["config"], seed, w["group"], w["method_label"], "severity_retuned")
            trial.update(role="severity_confirmation", selected_from_validation_variant=w["variant_id"])
            jobs.append(trial)
    selection = dict(stage="severity_retuned", selected_variants=winners, validation_trials=variants,
                     screening_seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
                     screen_manifest=file_fingerprint(args.screen), protocol=protocol,
                     selection_rule="Highest mean hard-law validation accuracy, then lowest validation cross entropy, then variant ID")
    write_once(args.output.with_suffix(".selection.json"), json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return write_manifest(args.output, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--source-selection", type=Path, required=True)
    screen.add_argument("--primary", type=Path, required=True)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--screen", type=Path, required=True)
    confirm.add_argument("--runs", type=Path, required=True)
    for sub in (screen,confirm):
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_screen(args) if args.command == "screen" else prepare_confirmation(args)))


if __name__ == "__main__":
    main()
