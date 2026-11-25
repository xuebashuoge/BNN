"""Frozen full-data comparison at two finite channel-bank sizes.

Prepare -> CPU pilots -> validation screen -> unconditional confirmation -> report.
The scientific grid and stopping rule do not depend on test performance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from .build_campaign import make_entry, variant_config
from .campaign import select_variants
from .full_fading_comparison import read_results, write_manifest, write_once
from .run_task import read_manifest


REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "results/final_chosen_results"
DESTINATION = REPO / "results/fair_small_bank_20261123"
SCRATCH = Path("/data/gpfs/projects/punim2792/yangshuo/fair_small_bank_20261123")
DATASETS = ("moons", "mnist", "fashionmnist", "cifar10")
LABELS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched", "Proposed-Perfect")
LAWS = {"Fading": {"kind": "diagonal_fading", "rho_c": .35},
        "Erasure": {"kind": "outage", "outage_prob": .5}}
BANKS = (50, 100)
SCREEN_SEEDS = (26112301, 26112302)
CONFIRM_SEEDS = tuple(range(26112401, 26112413))
DATA_SEED = 20261123
FTZ = Path("/data/gpfs/projects/punim2792/yangshuo/matrix_cpu_runtime_20261118/libbnn_ftz.so")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(path, value):
    write_once(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_configs():
    path = REFERENCE / "data/performance/selected_trials.tar.gz"
    inventory = json.loads((REFERENCE / "data/performance/provenance.json").read_text())
    if sha(path) != inventory["files"]["data/performance/selected_trials.tar.gz"]["sha256"]:
        raise ValueError("Reference archive fingerprint differs")
    configs = {}
    with tarfile.open(path, "r:gz") as archive:
        for cohort in inventory["cohorts"]:
            for label in LABELS:
                variants = []
                for seed in cohort["confirmation_seeds"]:
                    key = f"{cohort['archive_directory']}/{label}/{seed}/task_config.json"
                    variants.append(variant_config(json.load(archive.extractfile(key))))
                if any(v != variants[0] for v in variants):
                    raise ValueError("Reference cohort mixes hyperparameters")
                configs[cohort["dataset"], cohort["channel"], label] = variants[0]
    return configs


def candidates(original, label, m):
    config = dict(original, m=m, data_seed=DATA_SEED, val_draws=5, eval_draws=20,
                  device="cpu", threads=2, download=False)
    old_m, n = original["m"], original["train_size"]
    if label == "T1-Matched":
        config["eta_data"] *= math.sqrt((old_m - 1) / (m - 1))
    elif label == "T2-Matched":
        config["eta_data"] *= math.sqrt((1 + (old_m - 1) / (n - 1)) /
                                         (1 + (m - 1) / (n - 1)))
    result = [dict(config, lr=config["lr"] * factor) for factor in (.5, 1., 2.)]
    if label == "ERM-Matched":
        result.extend(dict(config, posterior_std=config["posterior_std"] * factor)
                      for factor in (.1, 3., 10.))
    else:
        for factor in (0., .1, 10.):
            candidate = dict(config)
            for field in (("l2",) if label == "L2-Matched" else ("eta_channel", "eta_data")):
                candidate[field] *= factor
            result.append(candidate)
    if len({json.dumps(v, sort_keys=True) for v in result}) != 6:
        raise ValueError("Each method requires six distinct candidates")
    return result


def prepare():
    configs = source_configs()
    screen, pilot = [], []
    for dataset in DATASETS:
        for channel, law in LAWS.items():
            for m in BANKS:
                group = f"{dataset}_{channel.lower()}_m{m}"
                for label in LABELS:
                    base = configs[dataset, channel, label]
                    if base["deployment_channel"] != law:
                        raise ValueError("Reference deployment law differs")
                    for candidate in candidates(base, label, m):
                        screen.extend(make_entry(candidate, seed, group, label, "fair_small_bank_screen")
                                      for seed in SCREEN_SEEDS)
                    if m == 50 and label == "L2-Matched":
                        cfg = dict(candidates(base, label, m)[1], epochs=1)
                        trial = make_entry(cfg, 26112201, group, label, "fair_small_bank_pilot")
                        trial["scientific_result"] = False
                        pilot.append(trial)
    validate_screen(screen)
    screen_info = write_manifest(DESTINATION / "screen.jsonl", screen)
    pilot_info = write_manifest(DESTINATION / "pilot.jsonl", pilot)
    freeze(DESTINATION / "protocol.json", dict(
        campaign="fair_small_bank_20261123", datasets=DATASETS, channels=LAWS, banks=BANKS,
        methods=LABELS, screening_seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
        data_seed=DATA_SEED, candidates_per_method=6, validation_draws=5, test_draws=20,
        screen=screen_info, pilot=pilot_info, screening_fits=960, confirmation_fits=960,
        source_archive_sha256=sha(REFERENCE / "data/performance/selected_trials.tar.gz"),
        source_configs=[dict(dataset=d, channel=c, method_label=l, config=configs[d, c, l])
                        for d in DATASETS for c in LAWS for l in LABELS],
        tuning="Each method starts from its archived validation-selected settings: LR factors 0.5,1,2; plus ERM posterior-SD factors 0.1,3,10 or regularizer factors 0,0.1,10 at the original LR. Zero regularization is available to every regularized method.",
        bank_rescaling="Keep dimensionless strengths fixed: T1 eta_data times sqrt((old_m-1)/(m-1)); T2 eta_data times sqrt((1+(old_m-1)/(n-1))/(1+(m-1)/(n-1))). Same rule as the existing bank-size implementation.",
        selection="Highest mean validation accuracy over both screening seeds, then lowest mean validation cross-entropy, then variant ID. Every group and method proceeds to confirmation, regardless of ordering.",
        primary="Per group, choose the higher-validation T1/T2 and higher-validation ERM/L2 before confirmation; ties use lower validation CE then label. Two-sided paired sign-flip tests, Holm across 16 primary contrasts.",
        secondary="All 64 T1/T2-versus-ERM/L2 contrasts, separately Holm-adjusted. Pointwise paired t intervals are sensitivity summaries. Training seed is the independent analysis unit; fixed official test examples.",
        stopping="Complete exactly the frozen grid and all twelve confirmation seeds. No test-driven extension, seed exclusion, or requirement that a particular method wins. Infrastructure failures block aggregation until resolved; finite low-accuracy runs remain included.",
        interpretation="Prospectively frozen follow-up conditional on previously explored benchmarks and chosen hyperparameter centers; not an independent new benchmark. Moons erasure includes replication of previously studied bank sizes.",
        scratch=str(SCRATCH), final_models=str(DESTINATION / "models"),
        protected_reference="final_chosen_results is read-only throughout this campaign",
    ))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "logs").mkdir(exist_ok=True)
    print(json.dumps(dict(screening=len(screen), pilots=len(pilot), confirmation=960,
                          protocol=str(DESTINATION / "protocol.json"))))


def validate_screen(entries):
    expected_groups = {f"{d}_{c.lower()}_m{m}" for d in DATASETS for c in LAWS for m in BANKS}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    fixed_by_group = {}
    fixed_fields = ("dataset", "architecture", "predictor", "training_loss", "loss_epsilon",
                    "epochs", "width", "batch_size", "train_size", "val_size", "data_seed",
                    "m", "deployment_channel", "eval_draws", "val_draws", "selection_metric",
                    "validation_seed_policy", "rho_lr_scale")
    for e in entries:
        if e["stage"] != "fair_small_bank_screen":
            raise ValueError("Wrong screening stage")
        groups[e["group"]][e["method_label"]][e["variant_id"]].add(e["config"]["seed"])
        fixed = {key: e["config"][key] for key in fixed_fields}
        if fixed_by_group.setdefault(e["group"], fixed) != fixed:
            raise ValueError("Mixed architecture, training, or evaluation budget within a group")
    if set(groups) != expected_groups or len(entries) != 960:
        raise ValueError("The full 16-group screening grid is required")
    for methods in groups.values():
        if set(methods) != set(LABELS):
            raise ValueError("Incomplete method coverage")
        for variants in methods.values():
            if len(variants) != 6 or any(seeds != set(SCREEN_SEEDS) for seeds in variants.values()):
                raise ValueError("Unequal candidate or seed coverage")


def select():
    manifest = DESTINATION / "screen.jsonl"
    protocol = json.loads((DESTINATION / "protocol.json").read_text())
    if sha(manifest) != protocol["screen"]["sha256"]:
        raise ValueError("Frozen screening manifest changed")
    entries = read_manifest(manifest)
    validate_screen(entries)
    rows = read_results(SCRATCH / "screen", entries)
    if any(r.get("source_manifest_sha256") != sha(manifest) or
           any(k.startswith("test_") for k in r) or
           not all(math.isfinite(r[k]) for k in ("validation_accuracy", "validation_cross_entropy"))
           for r in rows):
        raise ValueError("Selection requires complete finite validation-only observations")
    summaries, selected = select_variants(rows, entries, 1)
    if len(selected) != 80:
        raise ValueError("Every method in every group must be selected")
    primary_pairs = []
    for group in sorted({e["group"] for e in entries}):
        subset = {r["method_label"]: r for r in selected if r["group"] == group}
        def rank(label):
            row = subset[label]
            return (-row["validation_accuracy"]["mean"], row["validation_cross_entropy"]["mean"], label)
        primary_pairs.append(dict(group=group, proposed=min(("T1-Matched", "T2-Matched"), key=rank),
                                  baseline=min(("ERM-Matched", "L2-Matched"), key=rank)))
    selection = dict(selected_variants=selected, primary_pairs=primary_pairs,
                     confirmation_seeds=CONFIRM_SEEDS, screening_seeds=SCREEN_SEEDS,
                     screen_manifest_sha256=sha(manifest), validation_trials=summaries,
                     protocol_sha256=sha(DESTINATION / "protocol.json"))
    freeze(DESTINATION / "selection.json", selection)
    confirm = []
    for choice in selected:
        for seed in CONFIRM_SEEDS:
            entry = make_entry(choice["config"], seed, choice["group"], choice["method_label"],
                               "fair_small_bank_confirm")
            entry["selected_from_validation_variant"] = choice["variant_id"]
            confirm.append(entry)
    write_manifest(DESTINATION / "confirm.jsonl", confirm)
    print(json.dumps(dict(selected=len(selected), confirmation=len(confirm))))


def clean_revision():
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip():
        raise RuntimeError("Commit scientific code and inputs before submitting")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def submit_job(name, command, manifest=None, tasks=None, dependency=None, limit=64):
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("SLURM_", "SBATCH_", "SRUN_"))}
    environment.update(JOURNAL_GIT_SHA=clean_revision(), DATASET="/home/yangshuo/Datasets",
                       FULL_FADING_FTZ_LIBRARY=str(FTZ), FULL_FADING_FTZ_SHA256=sha(FTZ),
                       OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
    environment.pop("JOURNAL_MANIFEST_SHA256", None)
    if manifest:
        environment["JOURNAL_MANIFEST_SHA256"] = sha(manifest)
    args = ["sbatch", "--parsable", f"--job-name={name}", "--time=00:40:00",
            "--kill-on-invalid-dep=yes", f"--chdir={REPO}",
            f"--output={SCRATCH}/logs/%x-%A_%a.log", "--export=ALL"]
    if tasks:
        args.append(f"--array=0-{tasks - 1}%{limit}")
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    args.extend([str(REPO / "journal/full_fading_spartan.slurm"), *command])
    job = subprocess.check_output(args, cwd=REPO, env=environment, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected submission response: {job}")
    return job


def worker_command(stage):
    return ["python", "-u", "-m", "journal.fair_small_bank_worker", "--manifest",
            str(DESTINATION / f"{stage}.jsonl"), "--output-root", str(SCRATCH / stage), "--threads", "2"]


def submit():
    path = SCRATCH / "submission.json"
    # Record each submission immediately so a partial submission is recoverable.
    from .campaign import save_json
    state = json.loads(path.read_text()) if path.exists() else dict(revision=clean_revision(), jobs={})
    if state["revision"] != clean_revision():
        raise ValueError("Existing submission used a different source revision")
    for stage, tasks, dependency in (("pilot", 8, None), ("screen", 960, "pilot")):
        if stage in state["jobs"]:
            continue
        job = submit_job("fsb_" + stage, worker_command(stage), DESTINATION / f"{stage}.jsonl", tasks,
                         state["jobs"].get(dependency))
        state["jobs"][stage] = job
        save_json(path, state)
    if "select" not in state["jobs"]:
        job = submit_job("fsb_select", ["python", "-u", "-m", "journal.fair_small_bank", "advance"],
                         dependency=state["jobs"]["screen"])
        state["jobs"]["select"] = job
        save_json(path, state)
    print(json.dumps(state))


def advance():
    select()
    path = SCRATCH / "confirmation_submission.json"
    from .campaign import save_json
    state = json.loads(path.read_text()) if path.exists() else {}
    if "confirm" not in state:
        state["confirm"] = submit_job("fsb_confirm", worker_command("confirm"), DESTINATION / "confirm.jsonl", 960)
        save_json(path, state)
    if "report" not in state:
        state["report"] = submit_job("fsb_report", ["python", "-u", "-m", "journal.fair_small_bank", "finalize"],
                                     dependency=state["confirm"])
        save_json(path, state)
    print(json.dumps(state))


def finalize():
    """Archive every confirmation model, then sync only this campaign's reports."""
    from .fair_small_bank_report import report
    revision = clean_revision()
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True).strip():
        raise RuntimeError("Leave existing staged work untouched; result sync needs an empty index")
    summary = report(DESTINATION / "confirm.jsonl", SCRATCH / "confirm",
                     DESTINATION / "selection.json", DESTINATION, copy_models=True)
    freeze(DESTINATION / "submission.json", dict(
        training_revision=revision,
        screening=json.loads((SCRATCH / "submission.json").read_text()),
        confirmation=json.loads((SCRATCH / "confirmation_submission.json").read_text()),
        checkpoint_sync="All confirmation checkpoints stored locally under Git/bnn; scripts, manifests and reports synced by Git. Binary models are excluded from ordinary Git."))
    filenames = ("confirm.jsonl", "selection.json", "submission.json", "accuracy_by_seed.csv",
                 "accuracy_means.csv", "primary_contrasts.csv", "secondary_contrasts.csv",
                 "accuracy_table.tex", "model_files.json", "summary.json")
    paths = [str((DESTINATION / name).relative_to(REPO)) for name in filenames]
    subprocess.run(["git", "add", "--", *paths], cwd=REPO, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
        subprocess.run(["git", "commit", "-m", "Report every outcome of the frozen small-bank comparison"], cwd=REPO, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPO, check=True)
    print(json.dumps(summary))


def status():
    for stage in ("pilot", "screen", "confirm"):
        paths = list((SCRATCH / stage / "reports").glob("*.json"))
        counts = Counter()
        for path in paths:
            row = json.loads(path.read_text())
            counts[row.get("test_status", row.get("status", "unknown"))] += 1
        print(json.dumps(dict(stage=stage, reports=len(paths), statuses=counts)))
    for name in ("submission.json", "confirmation_submission.json"):
        if (SCRATCH / name).exists():
            print((SCRATCH / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "select", "submit", "advance", "finalize", "status"))
    args = parser.parse_args()
    if args.command in {"submit", "advance", "finalize"}:
        with (SCRATCH / (args.command + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            globals()[args.command]()
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
