"""Equal-budget validation grids for feature erasure and the additional hard MNIST comparison."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

from .campaign import select_variants
from .diagonal_fading_comparison import METHODS, entry, validation_rows
from .erasure_bank_comparison import ERASURE
from .fading_severity import HARD
from .full_fading_comparison import write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest
from .severity_validation import SCREEN_SEEDS as PRIOR_HARD_SEEDS


DESIGNS = {
    "erasure": dict(law=ERASURE, rates=(.6, 1., 1.5), scale=(1., 3.), relative_rates=True,
                    screen=(351101, 351102, 351103), confirm=tuple(range(391201, 391210)),
                    role="erasure_confirmation", logdir="validation", out="retuned"),
    "erasure_common": dict(law=ERASURE, rates_by_dataset={"mnist": (.0016, .0035), "fashionmnist": (.0008, .0012)},
                           scale=(.001, .003), posterior_lr_scale=.1, relative_rates=False,
                           screen=(351101, 351102, 351103), confirm=tuple(range(411201, 411210)),
                           role="erasure_confirmation", logdir="common/validation", out="common/retuned"),
    "mnist_hard": dict(law=HARD, rates=(.006, .018), scale=(.3, 6.), relative_rates=False,
                       screen=PRIOR_HARD_SEEDS, confirm=tuple(range(381201, 381210)),
                       role="severity_confirmation", logdir="expansion/validation", out="expansion/retuned"),
}


def inputs(mode, root):
    design = DESIGNS[mode]
    primary = root / "manifest.jsonl"
    entries = read_manifest(primary)
    anchors = {item["method_label"]: item["config"] for item in entries}
    source = primary
    if mode in {"mnist_hard", "erasure_common"}:
        if mode == "mnist_hard" and {c["dataset"] for c in anchors.values()} != {"mnist"}:
            raise ValueError("The additional hard fading grid is for MNIST")
        source = root / "retuned/manifest.selection.json"
        selection = json.loads(source.read_text())
        candidates_per_method = 6 if mode == "erasure_common" else 4
        source_manifest = (selection["sources"][0] if mode == "erasure_common" else selection["screen_manifest"])
        if (len(selection["validation_trials"]) != len(METHODS) * candidates_per_method or selection["screening_seeds"] != list(design["screen"])
                or source_manifest != file_fingerprint(root / "validation/manifest.jsonl")):
            raise ValueError("Missing or altered frozen selection from the preceding validation round")
        anchors = {choice["method_label"]: choice["config"] for choice in selection["selected_variants"]}
    if set(anchors) != set(METHODS) or len({r["group"] for r in entries}) != 1:
        raise ValueError("Expected all seven method labels for exactly one comparison condition")
    for label, cfg in anchors.items():
        train = design["law"] if label.endswith("-Matched") else {"kind": "perfect"}
        if cfg["deployment_channel"] != design["law"] or cfg["train_channel"] != train:
            raise ValueError("Incompatible training or deployment law")
    return anchors, entries[0]["group"], source


def make_grid(mode, root):
    settings = DESIGNS[mode]
    anchors, group, source = inputs(mode, root)
    dataset = next(iter(anchors.values()))["dataset"]
    if mode == "erasure_common" and dataset not in settings["rates_by_dataset"]:
        raise ValueError("The common learning-rate erasure grid is for the two image datasets")
    rates = settings["rates_by_dataset"][dataset] if mode == "erasure_common" else settings["rates"]
    trials = []
    for label, anchor in anchors.items():
        for rate, factor in product(rates, settings["scale"]):
            cfg = dict(anchor)
            cfg["lr"] = rate * anchor["lr"] if settings["relative_rates"] else rate
            if mode == "erasure_common":
                cfg["posterior_std"] = factor
                cfg["rho_lr_scale"] = settings["posterior_lr_scale"]
            elif cfg["method"] == "erm":
                cfg["posterior_std"] *= factor
            elif cfg["method"] == "l2":
                cfg["l2"] *= factor
            else:
                cfg["eta_data"] *= factor
                cfg["eta_channel"] *= factor
            for seed in settings["screen"]:
                task = entry(cfg, seed, group, label, mode + "_validation_expansion")
                task["role"] = "comparison"
                trials.append(task)
    expected = len(rates) * len(settings["scale"])
    for label in METHODS:
        counts = Counter(item["variant_id"] for item in trials if item["method_label"] == label)
        if len(counts) != expected or set(counts.values()) != {len(settings["screen"])}:
            raise ValueError(f"Unequal screening budget for {label}")
    return trials, dict(stage=mode, source=file_fingerprint(source), design=settings,
                        original_primary=file_fingerprint(root / "manifest.jsonl"),
                        settings_per_label=expected, learning_rates=rates)


def build(mode, root):
    trials, protocol = make_grid(mode, root)
    screen = root / DESIGNS[mode]["logdir"] / "manifest.jsonl"
    result = write_manifest(screen, trials)
    protocol["manifest"] = file_fingerprint(screen)
    write_once(screen.with_suffix(".protocol.json"), json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return result


def freeze(mode, root):
    design = DESIGNS[mode]
    screen = root / design["logdir"] / "manifest.jsonl"
    rows = read_manifest(screen)
    expected, protocol = make_grid(mode, root)
    recorded = json.loads(screen.with_suffix(".protocol.json").read_text())
    if rows != expected or recorded["manifest"] != file_fingerprint(screen) or recorded["source"] != protocol["source"]:
        raise ValueError("The prospective grid changed")
    data = validation_rows(screen.parent / "runs", rows)
    sources = [file_fingerprint(screen)]
    if mode in {"mnist_hard", "erasure_common"}:
        previous = root / "validation/manifest.jsonl"
        prior = read_manifest(previous)
        rows.extend(prior)
        data.extend(validation_rows(previous.parent / "runs", prior))
        sources.append(file_fingerprint(previous))
    variants, winners = select_variants(data, rows, top=1)
    prior_count = {"mnist_hard": 4, "erasure_common": 6, "erasure": 0}[mode]
    n = protocol["settings_per_label"] + prior_count
    if len(variants) != len(METHODS) * n or any(tuple(v["seeds"]) != design["screen"] for v in variants):
        raise ValueError("Expected every candidate on every frozen validation seed")
    cohort = mode + "_retuned"
    confirmation = []
    for choice in winners:
        for seed in design["confirm"]:
            item = entry(choice["config"], seed, choice["group"], choice["method_label"], cohort)
            item.update(role=design["role"], selected_from_validation_variant=choice["variant_id"])
            confirmation.append(item)
    result = root / design["out"] / "manifest.jsonl"
    selection = dict(stage=cohort, selected_variants=winners, validation_trials=variants,
                     screening_seeds=design["screen"], confirmation_seeds=design["confirm"],
                     sources=sources, validation_protocol=recorded,
                     selection_rule="Highest mean validation accuracy, then lowest validation cross entropy, then variant ID")
    write_once(result.with_suffix(".selection.json"), json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return write_manifest(result, confirmation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=DESIGNS)
    parser.add_argument("action", choices=("screen", "confirm"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((build if args.action == "screen" else freeze)(args.mode, args.root)))
