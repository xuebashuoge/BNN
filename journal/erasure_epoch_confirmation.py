"""Confirm the validation-selected MNIST erasure models with a common 60-epoch budget."""
import argparse
import json
from pathlib import Path

from .diagonal_fading_comparison import METHODS, entry
from .erasure_bank_comparison import ERASURE
from .full_fading_comparison import write_manifest, write_once
from .run_experiment import file_fingerprint


SEEDS = tuple(range(431201, 431210))


def prepare(root):
    source = root / "common/retuned/manifest.selection.json"
    selection = json.loads(source.read_text())
    prior = selection["selected_variants"]
    if (selection["stage"] != "erasure_common_retuned" or len(selection["validation_trials"]) != 70
            or len(prior) != 7 or {x["method_label"] for x in prior} != set(METHODS)):
        raise ValueError("A frozen ten-candidate MNIST selection is required")
    if set(SEEDS).intersection(selection["confirmation_seeds"] + selection["screening_seeds"]):
        raise ValueError("The new paired seeds overlap with selection or confirmation")
    rows = []
    for choice in prior:
        cfg = choice["config"]
        training = ERASURE if choice["method_label"].endswith("-Matched") else {"kind":"perfect"}
        if (cfg["dataset"] != "mnist" or cfg["deployment_channel"] != ERASURE or cfg["train_channel"] != training
                or cfg["epochs"] != 30 or cfg["selection_metric"] != "accuracy"
                or cfg["validation_seed_policy"] != "fixed"):
            raise ValueError("Unexpected source model or epoch-selection policy")
        for seed in SEEDS:
            item = entry({**cfg, "epochs": 60}, seed, choice["group"], choice["method_label"], "erasure_60_epochs")
            item.update(role="erasure_confirmation", selected_from_validation_variant=choice["variant_id"])
            rows.append(item)
    result = root / "epochs60" / "manifest.jsonl"
    revised = dict(selection, stage="erasure_60_epochs", confirmation_seeds=SEEDS,
                   source_selection=file_fingerprint(source), original_epochs=30, confirmation_epochs=60,
                   epoch_rule="For every method choose the checkpoint with highest fixed-validation accuracy from epochs 1 through 60; tie-break on cross entropy",
                   training_rule="All other hyperparameters are exactly the validation-selected 30-epoch settings")
    write_once(result.with_suffix(".selection.json"), json.dumps(revised, indent=2, allow_nan=False)+"\n")
    return write_manifest(result, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().root)))
