"""Repeat the selected m=2000 configurations on ten common, fixed training seeds."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
import tempfile
import uuid

from .build_campaign import make_entry
from .campaign import completed_trial, evaluate_trial, train_trial
from .full_fading_comparison import write_manifest
from .run_experiment import file_fingerprint
from .run_task import verify_submission_snapshot
from results.final_chosen_results.rebuild_performance_table import CHANNELS, DATA, DATASETS, METHODS

SEEDS = tuple(range(761001, 761011))
TRAINING_FILES = ("run_experiment.py", "theorem1.py", "full_fading.py", "diagonal_fading.py",
                  "cifar_models.py", "cifar_search_models.py", "bounded_mlp.py")
CACHE_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError)


def archive_path(name):
    path = DATA / name
    inventory = json.loads((DATA / "provenance.json").read_text())
    if file_fingerprint(path) != inventory["files"][f"data/performance/{name}"]:
        raise ValueError(f"Archived source differs from its inventory: {path}")
    return path


def build_entries():
    """Get each protocol from every historical fit in its cell, without consulting accuracy."""
    inventory = json.loads((DATA / "provenance.json").read_text())
    cohorts = {(c["channel"], c["dataset"]): c for c in inventory["cohorts"]}
    with tarfile.open(archive_path("selected_trials.tar.gz"), "r|gz") as archive:
        files = {m.name: json.load(archive.extractfile(m)) for m in archive
                 if m.name.endswith(("/task_config.json", "/metadata.json", "/test_deployment/overrides.json"))}
    entries = []
    for channel, law in CHANNELS.items():
        for dataset in DATASETS:
            cohort = cohorts[channel, dataset]
            old_seeds = cohort["confirmation_seeds"]
            if set(old_seeds).intersection(SEEDS):
                raise ValueError("Replication seeds overlap the original cohort")
            for method in METHODS:
                source = None
                for seed in old_seeds:
                    key = f"{cohort['archive_directory']}/{method}/{seed}"
                    cfg = files[f"{key}/task_config.json"]
                    meta = files[f"{key}/metadata.json"]
                    override = files[f"{key}/test_deployment/overrides.json"]
                    if (cfg != meta["config"] or cfg["dataset"] != dataset or cfg["deployment_channel"] != law
                            or cfg["train_channel"] != ({"kind": "perfect"} if method == "Proposed-Perfect" else law)
                            or (cfg["m"], cfg["device"], cfg["seed"], cfg["channel_seed"], cfg["eval_seed"]) !=
                               (2000, "cpu", seed, seed + 10000, seed + 20000)
                            or override != dict(eval_draws=cfg["eval_draws"], eval_seed=cfg["eval_seed"] + 1000003,
                                                device="cpu", threads=cfg["threads"])):
                        raise ValueError(f"Inconsistent archive record: {key}")
                    protocol = {k: v for k, v in cfg.items() if k not in {"seed", "channel_seed", "eval_seed"}}
                    current = (protocol, meta["git_sha"], meta["split"])
                    if source is not None and source != current:
                        raise ValueError(f"The archived fits use different protocols or source revisions: {key}")
                    source = current
                protocol, revision, split = source
                for seed in SEEDS:
                    entry = make_entry(protocol, seed, f"{dataset}_{channel.lower()}_m2000",
                                       method, "paired_replication")
                    entry.update(channel=channel, source_revision=revision, source_split=split,
                                 source_archive_directory=f"{cohort['archive_directory']}/{method}",
                                 source_seeds=old_seeds)
                    entries.append(entry)
    return entries


def unpack_trainer(revision, directory):
    """Replay the exact per-cohort trainer and its dependencies in a private task directory."""
    package = directory / "journal"
    package.mkdir()
    (package / "__init__.py").touch()
    with tarfile.open(archive_path("training_source.tar.gz"), "r:gz") as archive:
        for name in TRAINING_FILES:
            source = archive.extractfile(f"{revision}/journal/{name}")
            (package / name).write_bytes(source.read())


def training_metadata(item, run_dir):
    cfg = item["config"]
    saved = json.loads((run_dir / "manifest_entry.json").read_text())
    meta = json.loads((run_dir / "metadata.json").read_text())
    if (saved != item or meta["config"] != cfg or meta["split"] != item["source_split"]
            or not 1 <= meta["selected_epoch"] <= cfg["epochs"]
            or completed_trial(run_dir, cfg) is None):
        raise ValueError(f"Incomplete or mismatched training record: {item['task_id']}")
    return meta


def validate_evaluation(item, run_dir):
    """Verify a complete fit and frozen deployment evaluation before accepting its score."""
    cfg = item["config"]
    meta = training_metadata(item, run_dir)
    report = json.loads((run_dir / "test_deployment/evaluation.json").read_text())
    if (report["selected_epoch"] != meta["selected_epoch"]
            or report["evaluation_config"] != dict(cfg, eval_seed=cfg["eval_seed"] + 1000003)):
        raise ValueError(f"Mismatched evaluation record: {item['task_id']}")
    for kind, filename in (("metadata", "metadata.json"), ("weights", "weights.pt")):
        if report["checkpoint_source_fingerprints"][kind] != file_fingerprint(run_dir / filename):
            raise ValueError(f"Changed checkpoint {filename}: {item['task_id']}")
    mc = report["heldout_gibbs_mc"]
    draws = [r["accuracy"] for r in mc["replicate_statistics"]]
    verified = report["checkpoint_verification"]
    if (report["split"] != "test" or report["risk_scope"] != "deployment"
            or mc["channel_source"] != "deployment" or mc["examples"] != 10000
            or mc["draws"] != cfg["eval_draws"] or len(draws) != mc["draws"]
            or not all(0 <= accuracy <= 1 for accuracy in draws)
            or not math.isclose(statistics.fmean(draws), mc["accuracy"], abs_tol=1e-12, rel_tol=0)
            or verified["bank_hash_matches_metadata"] is not True
            or verified["saved_channel_bank_hash"] != meta["training_channel_bank_hash"]
            or verified["split_hash_matches"] != {"train_indices_hash": True, "validation_indices_hash": True}):
        raise ValueError(f"Invalid deployment evaluation: {item['task_id']}")
    return report


def preserve_interrupted(path, root, task_id):
    """Keep partial work for diagnosis and permit deterministic retries on the same seed."""
    dest = root / "interrupted" / f"{task_id}-{uuid.uuid4().hex}"
    dest.parent.mkdir(exist_ok=True)
    path.rename(dest)


def run_task(root, index):
    manifest = root / "manifest.jsonl"
    verify_submission_snapshot(manifest)
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    if not 0 <= index < len(entries) or entries != build_entries():
        raise ValueError("The index or manifest differs from the frozen 400-fit design")
    item = entries[index]
    cfg, task_id = item["config"], item["task_id"]
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is not None and int(allocation) < cfg["threads"]:
        raise ValueError(f"The fit requires {cfg['threads']} CPU threads")
    run_dir = root / "runs" / task_id
    locks = root / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / f"{task_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if run_dir.exists():
            try:
                training_metadata(item, run_dir)
            except CACHE_ERRORS:
                preserve_interrupted(run_dir, root, task_id)
        evaluation = run_dir / "test_deployment"
        if evaluation.exists():
            try:
                return validate_evaluation(item, run_dir)
            except CACHE_ERRORS:
                preserve_interrupted(evaluation, root, task_id)
        with tempfile.TemporaryDirectory(prefix="bnn-paired-trainer-") as tmp:
            source = Path(tmp)
            unpack_trainer(item["source_revision"], source)
            # The campaign launches both phases using "python -m journal.run_experiment".
            # Its child interpreters import the archived modules from this working directory.
            with contextlib.chdir(source):
                row = train_trial(item, run_dir.parent, cfg["threads"], resume=True)
                if row["status"] != "complete":
                    raise RuntimeError(f"{task_id}: {row['error']}")
                row = evaluate_trial(row, cfg["eval_draws"], full_risks=False, resume=False)
                if row["test_status"] != "complete":
                    raise RuntimeError(f"{task_id}: {row['test_error']}")
        return validate_evaluation(item, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    task = sub.add_parser("task")
    for command in (build, task):
        command.add_argument("--output-root", type=Path, required=True)
    task.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "build":
        print(json.dumps(write_manifest(root / "manifest.jsonl", build_entries())))
    else:
        report = run_task(root, args.index)
        print(json.dumps(dict(index=args.index, test_accuracy=report["heldout_gibbs_mc"]["accuracy"])))


if __name__ == "__main__":
    main()
