"""Validate prior width at the m=2000 MNIST erasure reference and confirm only a strict validation win."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from .campaign import select_variants
from .diagonal_fading_comparison import METHODS, entry, run_task, validation_rows
from .erasure_bank_comparison import ERASURE
from .full_fading_comparison import write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest


SCREEN_SEEDS = tuple(range(501301, 501306))
TEST_SEEDS = tuple(range(541401, 541406))
PARAMETERS = {"erm": ("posterior_std", (.25, .5, 1., 2.)),
              "l2": ("l2", (.03, .3, 1., 3.)),
              "proposed": ("prior_std", (.125, .35, 1., 3.)),
              "joint": ("prior_std", (.125, .35, 1., 3.))}


def save(path, obj):
    write_once(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def design(source):
    selection = json.loads(source.read_text())
    winners = selection["selected_variants"]
    if (selection["stage"] != "erasure_penalty60" or len(selection["validation_trials"]) != 28
            or len(winners) != len(METHODS) or {w["method_label"] for w in winners} != set(METHODS)):
        raise ValueError("Expected the complete seven-label lower-penalty selection")
    if set(selection["confirmation_seeds"] + selection["screening_seeds"]) & set((*SCREEN_SEEDS, *TEST_SEEDS)):
        raise ValueError("New seeds must be disjoint from the source cohorts")
    tasks, axes = [], {}
    for w in winners:
        label, c = w["method_label"], w["config"]
        expected_training = ERASURE if label.endswith("-Matched") else {"kind": "perfect"}
        if (c["dataset"] != "mnist" or c["m"] != 2000 or c["epochs"] != 60
                or c["train_channel"] != expected_training or c["deployment_channel"] != ERASURE
                or c["selection_metric"] != "accuracy" or c["prior_std"] != 1.):
            raise ValueError("Unexpected MNIST erasure reference anchor")
        axis, factors = PARAMETERS[c["method"]]
        values = [x if axis == "prior_std" else c[axis] * x for x in factors]
        if len(set(values)) != 4 or c[axis] not in values:
            raise ValueError(f"{label} requires four distinct settings including its anchor")
        axes[label] = dict(parameter=axis, values=values, anchor=c[axis],
                           anchor_validation_variant=w["variant_id"])
        for value in values:
            for seed in SCREEN_SEEDS:
                item = entry({**c, axis: value}, seed, w["group"], label, "erasure_prior_validation")
                item.update(role="comparison", selected_from_validation_variant=w["variant_id"])
                tasks.append(item)
    protocol = dict(source=str(source.resolve()), source_selection=file_fingerprint(source), per_label_axis=axes,
                    screening_seeds=SCREEN_SEEDS, confirmation_seeds=TEST_SEEDS, train_epochs=60, method_labels=list(METHODS),
                    selection_rule="Highest five-seed mean validation accuracy, then lowest mean validation cross entropy, then variant ID",
                    gate="Confirm all seven methods once only if the higher matched T1/T2 mean validation accuracy strictly exceeds every other selected label")
    return tasks, protocol


def build(root):
    source = root / "penalty60/retuned/manifest.selection.json"
    out = root / "prior_width_reference"
    tasks, protocol = design(source)
    manifest = out / "validation/manifest.jsonl"
    result = write_manifest(manifest, tasks)
    protocol["validation_manifest"] = file_fingerprint(manifest)
    save(out / "protocol.json", protocol)
    return result


def validate_and_freeze(manifest, runs):
    out = manifest.parent.parent
    protocol = json.loads((out / "protocol.json").read_text())
    expected, frozen = design(Path(protocol["source"]))
    tasks = read_manifest(manifest)
    frozen["validation_manifest"] = file_fingerprint(manifest)
    if tasks != expected or protocol != json.loads(json.dumps(frozen)):
        raise ValueError("The prospective source, grid or seed protocol changed")
    rows = validation_rows(runs, tasks)
    variants, selected = select_variants(rows, tasks, top=1)
    if (len(selected) != len(METHODS) or Counter(v["method_label"] for v in variants) != {label:4 for label in METHODS}
            or any(tuple(v["seeds"]) != SCREEN_SEEDS for v in variants)):
        raise ValueError("Validation coverage is incomplete")
    scores = {w["method_label"]: w["validation_accuracy"]["mean"] for w in selected}
    methods = ("T1-Matched", "T2-Matched")
    best_proposed = max(scores[label] for label in methods)
    best_other = max(v for k, v in scores.items() if k not in methods)
    margin = best_proposed - best_other
    passed = margin > 1e-12
    selection = dict(stage="erasure_prior_reference", validation_trials=variants, selected_variants=selected,
                     screening_seeds=SCREEN_SEEDS, confirmation_seeds=TEST_SEEDS, validation_means=scores,
                     validation_margin=margin, validation_gate_passed=passed, protocol=protocol)
    save(out / "validation_selection.json", selection)
    if passed:
        confirmations = []
        for choice in selected:
            for seed in TEST_SEEDS:
                item = entry(choice["config"], seed, choice["group"], choice["method_label"], "erasure_prior_reference")
                item.update(role="erasure_confirmation", selected_from_validation_variant=choice["variant_id"])
                confirmations.append(item)
        write_manifest(out / "confirmation/manifest.jsonl", confirmations)
        save(out / "confirmation/manifest.selection.json", selection)
    return dict(validation_trials=len(rows), validation_gate_passed=passed, validation_margin=margin, validation_means=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build").add_argument("--root", type=Path, required=True)
    task = commands.add_parser("task")
    task.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    task.add_argument("--threads", type=int, default=1)
    task.add_argument("--index", type=int)
    for p in (task, commands.add_parser("report")):
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.root)
    elif args.command == "task":
        result = run_task(args)
    else:
        result = validate_and_freeze(args.manifest, args.output_root)
        if result["validation_gate_passed"]:
            out = args.manifest.parent.parent / "confirmation"
            (out / "logs").mkdir(parents=True, exist_ok=True)
            subprocess.run(["bash", "journal/submit_bank_array.sh", "journal.erasure_bank_comparison",
                            str(out), "1", "erasure_prior_confirm"], check=True)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
