"""Cross learning rates and posterior scales under p=.50, carrying validation-selected penalties."""
import argparse
from itertools import product
import json
from pathlib import Path

from .build_campaign import make_entry
from .campaign import select_variants
from .diagonal_fading_comparison import validation_rows
from .fading_severity_cifar_search import SCREEN_SEEDS, screening
from .full_fading_comparison import METHODS, variant_grid, write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest


RATES = (.002, .0035)
POSTERIOR_SCALES = (.001, .003)
CONFIRM_SEEDS = (175101, 175102, 175103, 175104, 175105)


def first_round(transfer):
    prior = transfer / "retuned"
    manifest = prior / "screen.jsonl"
    entries = read_manifest(manifest)
    if not variant_grid(entries, screening(transfer, "erasure")):
        raise ValueError("Require the complete original erasure validation grid")
    rows = validation_rows(prior / "screen-runs", entries)
    _, winners = select_variants(rows, entries, 1)
    frozen = json.loads((prior / "selection.json").read_text())
    if frozen["screen"] != file_fingerprint(manifest) or winners != frozen["selected_variants"]:
        raise ValueError("The previously frozen validation winners differ from the original records")
    return entries, rows, winners


def new_entries(original, winners):
    by_id = {item["task_id"]:item for item in original}
    old = {(item["method_label"], item["variant_id"]) for item in original}
    new = []
    for choice in winners:
        best = by_id[choice["task_ids"][0]]
        for rate, scale in product(RATES, POSTERIOR_SCALES):
            config = dict(best["config"], lr=rate, posterior_std=scale)
            for seed in SCREEN_SEEDS:
                row = make_entry(config, seed, best["group"], best["method_label"],
                                 "erasure_cifar_scale_screen", "erasure_validation")
                if (row["method_label"], row["variant_id"]) in old:
                    raise ValueError("The expanded tuning budget contains a previous candidate")
                row.update(family=best["family"], transfer_variant=best["transfer_variant"],
                           penalty_from_validation_variant=best["variant_id"])
                new.append(row)
    return new


def select(transfer, directory):
    original, rows, first = first_round(transfer)
    additional = read_manifest(directory / "screen.jsonl")
    if not variant_grid(additional, new_entries(original, first)):
        raise ValueError("The shared learning-rate/posterior-scale grid is incomplete or altered")
    complete = original + additional
    rows.extend(validation_rows(directory / "screen-runs", additional))
    trials, winners = select_variants(rows, complete, 1)
    if len(trials) != 13 * len(METHODS) or len(winners) != len(METHODS):
        raise ValueError("Require all 13 candidates for each label")
    initial = {choice["method_label"]:choice["variant_id"] for choice in first}
    changed = [x["method_label"] for x in winners if initial[x["method_label"]] != x["variant_id"]]
    protocol = dict(candidates_per_method=13, screening_seeds=SCREEN_SEEDS,
                    confirmation_seeds=CONFIRM_SEEDS if changed else [],
                    first_screen=file_fingerprint(transfer/"retuned/screen.jsonl"),
                    first_selection=file_fingerprint(transfer/"retuned/selection.json"),
                    supplemental_screen=file_fingerprint(directory/"screen.jsonl"),
                    selection_rule="Highest pooled accuracy across all three paired validation seeds for each candidate, then lowest mean validation CE, then variant ID",
                    initial_variants=initial, changed_methods=changed, validation_trials=trials,
                    selected_variants=winners, unchanged_confirmation_source=str(transfer/"retuned/runs") if not changed else None)
    write_once(directory/"selection.json",json.dumps(protocol,indent=2,allow_nan=False)+"\n")
    if not changed:
        return dict(tasks=0,changed_methods=[])
    by_id = {x["task_id"]:x for x in complete}
    final = []
    for choice in winners:
        best = by_id[choice["task_ids"][0]]
        for seed in CONFIRM_SEEDS:
            row = make_entry(best["config"], seed, best["group"], best["method_label"],
                             "erasure_cifar_scale_confirm", "erasure_confirmation")
            row.update(family=best["family"], transfer_variant=best["transfer_variant"],
                       selected_from_validation_variant=choice["variant_id"])
            final.append(row)
    return dict(**write_manifest(directory/"manifest.jsonl", final), changed_methods=changed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("screen","select"))
    parser.add_argument("--transfer",type=Path,required=True)
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="screen":
        entries,_,best=first_round(args.transfer)
        result=write_manifest(args.directory/"screen.jsonl",new_entries(entries,best))
    else:
        result=select(args.transfer,args.directory)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main()
