"""Copy complete, unchanged training checkpoints into an expanded campaign.

This command neither trains nor evaluates. Source checkpoints and their logs
remain untouched; partial, foreign and corrupt trials are recorded and skipped.
Reusing a checkpoint requires identical committed and working training sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import uuid

import torch

from .campaign import completed_trial, save_json, validate_entries
from .run_experiment import digest, make_models, normalize_config


TRAINING_FILES = ("journal/run_experiment.py", "journal/theorem1.py",
                  "journal/cifar_models.py", "journal/cifar_search_models.py")
CHECKPOINT_FILES = ("metadata.json", "weights.pt", "history.json", "task_config.json",
                    "manifest_entry.json", "train.log")


def fingerprint(path):
    checksum, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            count += len(block)
    return dict(sha256=checksum.hexdigest(), bytes=count)


def verify_training_sources(source_revision, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True,
                                       stderr=subprocess.PIPE).strip()

    try:
        source_sha = git("rev-parse", "--verify", "--end-of-options", source_revision + "^{commit}")
        target_sha = git("rev-parse", "--verify", "HEAD^{commit}")
        files = {}
        for path in TRAINING_FILES:
            source_blob = git("rev-parse", source_sha + ":" + path)
            target_blob = git("rev-parse", target_sha + ":" + path)
            working_blob = git("hash-object", "--", path)
            if not source_blob == target_blob == working_blob:
                raise ValueError(f"Training source changed; checkpoint reuse refused: {path}")
            files[path] = dict(source_blob=source_blob, target_blob=target_blob,
                               working_tree_blob=working_blob)
    except subprocess.CalledProcessError as error:
        raise ValueError("Cannot verify the source revision and all required training files") from error
    return dict(source_revision=source_sha, target_revision=target_sha, files=files)


def validate_checkpoint(path, entry, source_revision):
    """Use the campaign completeness contract, then verify saved tensor integrity."""
    expected = normalize_config(entry["config"])
    if expected["predictor"] != "gaussian" or expected["device"] != "cpu":
        raise ValueError("Staging requires the declared all-Gaussian CPU configuration")
    if not path.is_dir() or path.is_symlink():
        return None, "source_missing_or_symlink"
    if any(not (path / name).is_file() or (path / name).is_symlink() for name in CHECKPOINT_FILES):
        return None, "source_partial_or_missing_provenance"
    summary = completed_trial(path, expected)
    if summary is None:
        return None, "source_incomplete_or_config_mismatch"
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["git_sha"] != source_revision or metadata.get("git_dirty") is not False:
            return None, "source_revision_mismatch_or_dirty_checkout"
        original_entry = json.loads((path / "manifest_entry.json").read_text())
        original_config = json.loads((path / "task_config.json").read_text())
        if (normalize_config(original_config) != expected
                or normalize_config(original_entry["config"]) != expected
                or any(original_entry.get(key) != entry.get(key) for key in
                       ("task_id", "group", "method_label", "family", "stage", "variant_id"))):
            return None, "source_manifest_identity_or_config_mismatch"
        history = json.loads((path / "history.json").read_text())
        selected = metadata["selected_epoch"]
        if (isinstance(selected, bool) or not isinstance(selected, int)
                or not 1 <= selected <= expected["epochs"]
                or not math.isfinite(summary["peak_rss_mb"]) or summary["peak_rss_mb"] <= 0
                or any(record["epoch"] != index for index, record in enumerate(history, 1))
                or any(not math.isfinite(float(record["validation"][metric]))
                       for record in history for metric in ("accuracy", "cross_entropy"))
                or any(not 0 <= record["validation"]["accuracy"] <= 1
                       or record["validation"]["cross_entropy"] < 0 for record in history)):
            return None, "source_corrupt_history"
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all().item()
               for component in ("model", "bank") for value in state[component].values()):
            return None, "source_nonfinite_tensors"
        dimension, classes = (2, 2) if expected["dataset"] == "moons" else (
            3072 if expected["dataset"] == "cifar10" else 784, 10)
        model, bank = make_models(expected, dimension, classes)
        model.load_state_dict(state["model"], strict=True)
        if (set(state["bank"]) != set(bank.state_dict())
                or any(not torch.equal(state["bank"][key], value) for key, value in bank.state_dict().items())):
            return None, "source_channel_bank_mismatch"
        bank_hash = digest({key: state["bank"][key].tolist() for key in ("gain", "noise")})
        if metadata.get("training_channel_bank_hash") != bank_hash:
            return None, "source_channel_bank_hash_mismatch"
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, EOFError,
            AttributeError, pickle.UnpicklingError):
        return None, "source_corrupt_checkpoint_or_provenance"
    return summary, "complete"


def stage_checkpoints(source_root, output_root, source_revision, manifest, stage, repo_root=None):
    verification = verify_training_sources(source_revision, repo_root)
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if (source_root == output_root or source_root in output_root.parents or output_root in source_root.parents):
        raise ValueError("Source and output must be separate, non-nested campaign roots")
    if stage not in {"pilot", "screen"}:
        raise ValueError("Only validation-only pilot and screen checkpoints may be staged")
    source_stage = source_root / stage
    if not source_stage.is_dir() or source_stage.is_symlink():
        raise ValueError("Source campaign stage does not exist or is a symbolic link")
    manifest = Path(manifest).resolve()
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("Staging manifest is empty")
    validate_entries(entries)
    if any(not entry["task_id"] or entry["stage"] != "bayes_" + stage for entry in entries):
        raise ValueError("Manifest entries do not match the requested staging stage")
    target_stage = output_root / stage
    if target_stage.is_symlink():
        raise ValueError("Output stage must not be a symbolic link")
    target_stage.mkdir(parents=True, exist_ok=True)
    audit_directory = output_root / "staging_audits"
    audit_directory.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_path = audit_directory / f"{stage}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    audit = dict(schema_version=1, stage=stage, source_root=str(source_root), output_root=str(output_root),
                 manifest=str(manifest), manifest_fingerprint=fingerprint(manifest),
                 training_source_verification=verification, records=[], status="running",
                 copied_files=list(CHECKPOINT_FILES),
                 policy="Only complete exact-ID/config CPU Gaussian training checkpoints; no training, inference, test results or old-source writes")
    save_json(audit_path, audit)
    try:
        for entry in entries:
            source = source_root / stage / entry["task_id"]
            destination = target_stage / entry["task_id"]
            before = ({name: fingerprint(source / name) for name in CHECKPOINT_FILES}
                      if source.is_dir() and not source.is_symlink()
                      and all((source / name).is_file() and not (source / name).is_symlink()
                              for name in CHECKPOINT_FILES) else None)
            summary, status = validate_checkpoint(source, entry, verification["source_revision"])
            record = dict(task_id=entry["task_id"], source=str(source), destination=str(destination),
                          status=status, source_fingerprints=before)
            audit["records"].append(record)
            if summary is not None:
                provenance = dict(schema_version=1, source_root=str(source_root), source_checkpoint=str(source),
                    source_revision=verification["source_revision"], target_revision=verification["target_revision"],
                    manifest_fingerprint=audit["manifest_fingerprint"], target_entry_sha256=digest(entry),
                    training_source_verification=verification, source_fingerprints=before,
                    validation_summary=summary,
                    reuse_note="Original metadata, weights, history, configuration, manifest entry and training log copied byte-for-byte. No test results reused.")
                if destination.exists() or destination.is_symlink():
                    if (not destination.is_dir() or destination.is_symlink()
                            or {item.name for item in destination.iterdir()} != {*CHECKPOINT_FILES, "staging_provenance.json"}
                            or any(not (destination / name).is_file() or (destination / name).is_symlink()
                                   or fingerprint(destination / name) != before[name] for name in CHECKPOINT_FILES)
                            or not (destination / "staging_provenance.json").is_file()
                            or json.loads((destination / "staging_provenance.json").read_text()) != provenance):
                        raise ValueError(f"Refusing to overwrite an existing mismatched checkpoint: {destination}")
                    record["status"] = "already_staged_identical"
                else:
                    with tempfile.TemporaryDirectory(prefix=".staging-", dir=target_stage) as temporary:
                        candidate = Path(temporary) / "checkpoint"
                        candidate.mkdir()
                        for name in CHECKPOINT_FILES:
                            shutil.copyfile(source / name, candidate / name)
                        after = {name: fingerprint(source / name) for name in CHECKPOINT_FILES}
                        copied = {name: fingerprint(candidate / name) for name in CHECKPOINT_FILES}
                        if before != after or copied != before:
                            raise ValueError(f"Source changed while staging; nothing installed: {source}")
                        save_json(candidate / "staging_provenance.json", provenance)
                        candidate.rename(destination)
                    record["status"] = "staged_complete"
                if before != {name: fingerprint(source / name) for name in CHECKPOINT_FILES}:
                    raise ValueError(f"Source changed while staging: {source}")
            save_json(audit_path, audit)
        if audit["manifest_fingerprint"] != fingerprint(manifest):
            raise ValueError("Manifest changed while checkpoints were staged")
        if verify_training_sources(source_revision, repo_root) != verification:
            raise ValueError("Training source revision changed while checkpoints were staged")
        audit["status"] = "complete"
    except BaseException as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        audit["counts"] = {status: sum(record["status"] == status for record in audit["records"])
                           for status in sorted({record["status"] for record in audit["records"]})}
        save_json(audit_path, audit)
    return dict(audit=str(audit_path), **audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "screen"), required=True)
    args = parser.parse_args()
    result = stage_checkpoints(args.source_root, args.output_root, args.source_revision,
                               args.manifest, args.stage)
    print(json.dumps({key: result[key] for key in ("audit", "status", "counts")}, indent=2))


if __name__ == "__main__":
    main()
