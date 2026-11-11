"""Run one explicit journal manifest entry; never choose a seed or hyperparameter."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


EVALUATION_OVERRIDES = {"deployment_channel", "eval_draws", "eval_seed", "device", "threads", "batch_size", "download"}
PILOT_MATCH_FIELDS = ("dataset", "architecture", "method", "width", "batch_size", "m", "train_channel",
                      "deployment_channel", "train_size", "val_size", "val_draws")


def effective_config(item):
    """Validate without loading data, constructing models, or requiring CUDA."""
    from journal.run_experiment import normalize_config
    supplied = item["config"]
    if item.get("mode", "train") == "evaluate":
        unknown = set(supplied) - EVALUATION_OVERRIDES
        if unknown:
            raise ValueError(f"Evaluation overrides restricted to {sorted(EVALUATION_OVERRIDES)}; got {sorted(unknown)}")
        checkpoint = Path(item["checkpoint"])
        for name in ("metadata.json", "weights.pt"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(f"Evaluation checkpoint missing {checkpoint / name}")
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        return normalize_config({**metadata["config"], **supplied})
    return normalize_config(supplied)


def require_cpu_pilots(entries, pilot_root):
    """Every task must match a successfully completed representative CPU trial.

    Epoch count may be shorter in the pilot; seeds and tunable coefficients may
    vary. This establishes execution/shape coverage, not GPU performance or a
    walltime guarantee. The final GPU resource estimate still needs measurement.
    """
    root = Path(pilot_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CPU pilot directory not found: {root}")
    completed = []
    for path in sorted(root.rglob("metadata.json")):
        try:
            meta = json.loads(path.read_text())
            config = meta["config"]
            history = json.loads((path.parent / "history.json").read_text())
            finite_positive = all(isinstance(meta.get(key), (int, float)) and
                                  math.isfinite(meta[key]) and meta[key] > 0
                                  for key in ("total_seconds", "peak_rss_mb"))
            if (config.get("device") != "cpu" or not finite_positive or
                    not (path.parent / "weights.pt").is_file() or
                    len(history) != config["epochs"] or not history or
                    history[-1].get("epoch") != config["epochs"] or
                    not 1 <= meta.get("selected_epoch", 0) <= config["epochs"] or
                    not math.isfinite(history[-1]["validation"]["cross_entropy"])):
                continue
            # Normalize channels/defaults consistently with the target configs.
            from journal.run_experiment import normalize_config
            completed.append((path.parent, normalize_config(config)))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    matches = []
    for item in entries:
        target = effective_config(item)
        candidates = [str(path) for path, config in completed
                      if all(config[key] == target[key] for key in PILOT_MATCH_FIELDS)]
        if not candidates:
            signature = {key: target[key] for key in PILOT_MATCH_FIELDS}
            raise ValueError(f"No completed matching CPU pilot for {item['task_id']}: {signature}")
        matches.append({"task_id": item["task_id"], "cpu_pilots": candidates})
    return matches


def verify_submission_snapshot(manifest):
    expected_git = os.environ.get("JOURNAL_GIT_SHA")
    expected_manifest = os.environ.get("JOURNAL_MANIFEST_SHA256")
    if not expected_git and not expected_manifest and "SLURM_JOB_ID" not in os.environ:
        return
    if not expected_git or not expected_manifest:
        raise RuntimeError("Submission git/manifest fingerprints are required inside Slurm")
    actual_git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    actual_manifest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if actual_git != expected_git or dirty or actual_manifest != expected_manifest:
        raise RuntimeError("Checkout or manifest changed after submission; refusing to run a different experiment")


def read_manifest(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest is empty")
    ids = []
    for item in entries:
        if item.get("mode", "train") not in {"train", "evaluate"}:
            raise ValueError("mode must be train or evaluate")
        task_id = item["task_id"]
        if not task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id):
            raise ValueError("task_id must contain only letters, digits, underscores and hyphens")
        ids.append(task_id)
        if not isinstance(item.get("config"), dict):
            raise ValueError("Each task needs a config dictionary")
        if item.get("mode") == "evaluate" and not item.get("checkpoint"):
            raise ValueError("Evaluation requires an existing checkpoint directory")
        if item.get("split", "test") not in {"train", "validation", "test"}:
            raise ValueError("Evaluation split must be train, validation, or test")
        effective_config(item)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in manifest")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output-root", default="results/journal")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cpu-pilot", help="Require matching completed CPU pilots, recursively under this directory")
    args = parser.parse_args()
    entries = read_manifest(args.manifest)
    pilot_matches = require_cpu_pilots(entries, args.cpu_pilot) if args.cpu_pilot else []
    if args.validate_only:
        print(json.dumps({"tasks": len(entries), "ids": [x["task_id"] for x in entries], "cpu_pilot_matches": pilot_matches}))
        return
    verify_submission_snapshot(args.manifest)
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(entries):
        raise ValueError(f"Task index {index} outside manifest of length {len(entries)}")
    entry = entries[index]
    if args.device == "cuda":
        pilot_root = args.cpu_pilot or os.environ.get("JOURNAL_CPU_PILOT")
        if not pilot_root:
            raise RuntimeError("CUDA execution requires JOURNAL_CPU_PILOT or --cpu-pilot")
        pilot_matches = require_cpu_pilots([entry], pilot_root)
    output = Path(args.output_root) / entry["task_id"]
    if (output / "weights.pt").exists() and entry.get("mode", "train") == "train":
        raise FileExistsError(f"Refusing to overwrite existing checkpoint at {output}")
    if entry.get("mode") == "evaluate" and (output / "evaluation.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation at {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(entry["config"]) if entry.get("mode") == "evaluate" else effective_config(entry)
    config["device"] = args.device
    config["threads"] = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    config_path = output / "task_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "manifest_entry.json").write_text(json.dumps(entry, indent=2) + "\n")
    (output / "submission_snapshot.json").write_text(json.dumps({
        "git_sha": os.environ.get("JOURNAL_GIT_SHA"),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "manifest_path": str(Path(args.manifest).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpu_pilot_matches": pilot_matches,
    }, indent=2) + "\n")
    command = [sys.executable, "-m", "journal.run_experiment", "--mode", entry.get("mode", "train"),
               "--config", str(config_path), "--output", str(output)]
    if entry.get("mode") == "evaluate":
        command.extend(["--checkpoint", entry["checkpoint"], "--split", entry.get("split", "test")])
    print(json.dumps({"index": index, "task_id": entry["task_id"], "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
