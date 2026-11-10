"""Run one explicit journal manifest entry; never choose a seed or hyperparameter."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


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
    args = parser.parse_args()
    entries = read_manifest(args.manifest)
    if args.validate_only:
        print(json.dumps({"tasks": len(entries), "ids": [x["task_id"] for x in entries]}))
        return
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(entries):
        raise ValueError(f"Task index {index} outside manifest of length {len(entries)}")
    entry = entries[index]
    output = Path(args.output_root) / entry["task_id"]
    if (output / "weights.pt").exists() and entry.get("mode", "train") == "train":
        raise FileExistsError(f"Refusing to overwrite existing checkpoint at {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(entry["config"])
    config["device"] = args.device
    config["threads"] = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    config_path = output / "task_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "manifest_entry.json").write_text(json.dumps(entry, indent=2) + "\n")
    command = [sys.executable, "-m", "journal.run_experiment", "--mode", entry.get("mode", "train"),
               "--config", str(config_path), "--output", str(output)]
    if entry.get("mode") == "evaluate":
        command.extend(["--checkpoint", entry["checkpoint"], "--split", entry.get("split", "test")])
    print(json.dumps({"index": index, "task_id": entry["task_id"], "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
