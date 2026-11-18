"""Compare lighter erasure regularization with an equal four-setting budget per method."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .campaign import completed_trial, select_variants
from .diagonal_fading_comparison import METHODS, entry, validation_rows
from .erasure_bank_comparison import ERASURE
from .erasure_epoch_confirmation import SEEDS as EPOCH_COHORT
from .full_fading_comparison import write_manifest, write_once
from .run_experiment import file_fingerprint
from .run_task import read_manifest


SCREEN = EPOCH_COHORT[:3]
CONFIRM = tuple(range(451201, 451210))
STRENGTHS = (.03, .1, .3, 1.)
ERM_SCALES = (.25, .5, 1., 2.)


def make_grid(root):
    selected_path = root / "common/retuned/manifest.selection.json"
    selected = json.loads(selected_path.read_text())
    prior_path = root / "epochs60/manifest.jsonl"
    prior = {(x["method_label"], x["config"]["seed"]): x for x in read_manifest(prior_path)}
    winners = selected["selected_variants"]
    if selected["stage"] != "erasure_common_retuned" or {c["method_label"] for c in winners} != set(METHODS):
        raise ValueError("This study requires the shared-learning-rate erasure validation selection")
    fresh, reused = [], []
    for choice in winners:
        label = choice["method_label"]
        anchor = choice["config"]
        expected = ERASURE if label.endswith("-Matched") else dict(kind="perfect")
        if (anchor["dataset"] != "mnist" or anchor["deployment_channel"] != ERASURE
                or anchor["train_channel"] != expected or anchor["epochs"] != 30):
            raise ValueError("The MNIST half-erasure anchors changed")
        factors = ERM_SCALES if anchor["method"] == "erm" else STRENGTHS
        for value in factors:
            c = {**anchor, "epochs": 60}
            if anchor["method"] == "erm":
                c["posterior_std"] *= value
            elif anchor["method"] == "l2":
                c["l2"] *= value
            else:
                c["eta_channel"] *= value
                c["eta_data"] *= value
            for seed in SCREEN:
                item = entry(c, seed, choice["group"], label, "erasure_penalty_validation")
                item.update(role="comparison", regularization_factor=value)
                if value == 1.:
                    previous = prior[label, seed]
                    if previous["config"] != item["config"] or previous["variant_id"] != item["variant_id"]:
                        raise ValueError("A claimed reusable anchor differs from the exact requested configuration")
                    reused.append(previous)
                else:
                    fresh.append(item)
    if len(fresh) != 63 or len(reused) != 21:
        raise ValueError("Expected three new variants plus one reusable variant for every method")
    protocol = dict(stage="erasure_penalty_validation", grid_seed_count=3, grid_settings_per_label=4,
                    training_epochs=60, strength_multipliers=STRENGTHS, erm_posterior_scales=ERM_SCALES,
                    screening_seeds=SCREEN, confirmation_seeds=CONFIRM,
                    source_selection=file_fingerprint(selected_path),
                    anchor_training_manifest=file_fingerprint(prior_path),
                    reuse_rule="First three predefined 60-epoch cohort seeds; identical configs. Read training histories only.",
                    reused_task_ids=[x["task_id"] for x in reused])
    return fresh, reused, protocol


def build(root):
    fresh, _, protocol = make_grid(root)
    screen = root / "penalty60/validation/manifest.jsonl"
    result = write_manifest(screen, fresh)
    protocol["new_training_manifest"] = file_fingerprint(screen)
    write_once(screen.with_suffix(".protocol.json"), json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return result


def freeze(root):
    screen = root / "penalty60/validation/manifest.jsonl"
    expected, reused, protocol = make_grid(root)
    new = read_manifest(screen)
    protocol["new_training_manifest"] = file_fingerprint(screen)
    if new != expected or json.loads(screen.with_suffix(".protocol.json").read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Frozen penalty study changed")
    observations = validation_rows(screen.parent / "runs", new)
    sources = {}
    for item in reused:
        checkpoint = root / "epochs60/runs" / item["task_id"]
        row = completed_trial(checkpoint, item["config"])
        if row is None:
            raise ValueError(f"Anchor training history is incomplete for {item['task_id']}")
        # completed_trial reads training metadata and the validation history, and checks the weights file exists.
        observations.append({**item, **row, "seed": item["config"]["seed"], "status": "complete"})
        sources[item["task_id"]] = {k: file_fingerprint(checkpoint / k) for k in ("metadata.json","history.json","weights.pt")}
    summaries, chosen = select_variants(observations, [*new,*reused], top=1)
    if (Counter(x["method_label"] for x in summaries) != {label:4 for label in METHODS}
            or any(tuple(x["seeds"]) != SCREEN for x in summaries)):
        raise ValueError("Every label must have four complete three-seed validation choices")
    result = root / "penalty60/retuned/manifest.jsonl"
    rows = []
    for choice in chosen:
        for seed in CONFIRM:
            item = entry(choice["config"], seed, choice["group"], choice["method_label"], "erasure_penalty60")
            item.update(role="erasure_confirmation", selected_from_validation_variant=choice["variant_id"])
            rows.append(item)
    selection = dict(stage="erasure_penalty60", validation_trials=summaries, selected_variants=chosen,
                     screening_seeds=SCREEN, confirmation_seeds=CONFIRM, reused_training_sources=sources,
                     validation_protocol=protocol,
                     selection_rule="Highest three-seed validation accuracy, then lowest validation cross entropy, then variant ID")
    write_once(result.with_suffix(".selection.json"), json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return write_manifest(result, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("screen","confirm"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((build if args.action == "screen" else freeze)(args.root)))
