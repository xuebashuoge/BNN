"""Run the separate small-model pilot, frozen-risk evaluation and certificates."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .build_campaign import make_entry
from .campaign import save_json


def pilot_entries(entries):
    selected = {}
    for entry in entries:
        selected.setdefault((entry["group"], entry["method_label"]), entry)
    if len(entries) != 48 or len(selected) != 16:
        raise ValueError("Require the complete 48-case small-model family")
    pilots = []
    for entry in selected.values():
        config = {**entry["config"], "epochs": 1}
        item = make_entry(config, 76100, entry["group"], entry["method_label"], "bayes_certificate_pilot")
        item.update(family=entry["family"], scientific_result=False, role="resource_pilot")
        pilots.append(item)
    return pilots


def command(arguments, log, environment):
    with Path(log).open("a") as stream:
        subprocess.run([sys.executable, "-u", "-m", *arguments], stdout=stream,
                       stderr=subprocess.STDOUT, env=environment, check=True)


def run(manifest, output, workers=4, threads=2):
    from .batch_campaign import validate_cpu_budget
    from .run_task import read_manifest, verify_submission_snapshot
    if threads != 2 or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("The fixed small-model protocol uses two threads and one to four workers")
    validate_cpu_budget(workers, threads)
    verify_submission_snapshot(manifest)
    entries = read_manifest(manifest)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prior_status = root / "status.json"
    if prior_status.exists() and json.loads(prior_status.read_text()).get("git_sha") != os.environ.get("JOURNAL_GIT_SHA"):
        raise ValueError("Use the original committed source to resume this certificate output")
    state = dict(status="running", stage="pilot", manifest=str(Path(manifest).resolve()),
                 git_sha=os.environ.get("JOURNAL_GIT_SHA"), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                 scientific_setup="separate width8/n512/m2000 bounded-loss diagnostic",
                 main_accuracy_table=False, delta=.025, bank_draws=1000,
                 workers=workers, threads=threads)
    save_json(root / "status.json", state)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    try:
        pilot = root / "pilot.jsonl"
        content = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in pilot_entries(entries))
        if pilot.exists() and pilot.read_text() != content:
            raise ValueError("Resource pilot manifest changed")
        pilot.write_text(content)
        environment["JOURNAL_MANIFEST_SHA256"] = hashlib.sha256(pilot.read_bytes()).hexdigest()
        command(["journal.campaign", "--manifest", str(pilot), "--output-root", str(root / "pilot"),
                 "--workers", str(workers), "--threads", str(threads), "--evaluate-top", "0", "--resume"],
                root / "pilot.log", environment)
        rows = json.loads((root / "pilot" / "training_results.json").read_text())
        if len(rows) != 16 or any(row["status"] != "complete" for row in rows):
            raise ValueError("Incomplete small-model CPU pilot")
        state["pilot"] = dict(trials=16, max_worker_rss_mib=max(row["peak_rss_mb"] for row in rows),
                              max_trial_seconds=max(row["seconds"] for row in rows))
        projected_memory = 1024 + 1.25 * workers * state["pilot"]["max_worker_rss_mib"]
        if os.environ.get("SLURM_MEM_PER_NODE") and projected_memory > int(os.environ["SLURM_MEM_PER_NODE"]):
            raise RuntimeError("Measured pilot pool with headroom exceeds allocated memory")
        state["stage"] = "train_and_evaluate"
        save_json(root / "status.json", state)
        environment["JOURNAL_MANIFEST_SHA256"] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        campaign = root / "campaign"
        campaign_complete = False
        if (campaign / "test_results.json").exists():
            from .evaluate_bayesian_certificate import declared_family
            try:
                declared_family(manifest, campaign)
                campaign_complete = True
            except (OSError, ValueError, KeyError):
                if (root / "bounds" / "certificate_family.json").exists():
                    raise ValueError("The frozen certificate source campaign changed on resume")
        if not campaign_complete:
            command(["journal.campaign", "--manifest", str(manifest), "--output-root", str(campaign),
                     "--workers", str(workers), "--threads", str(threads), "--evaluate-top", "1",
                     "--eval-draws", "20", "--full-risks", "--resume"], root / "campaign.log", environment)
        state["resumed_complete_campaign"] = campaign_complete
        summary = json.loads((campaign / "summary.json").read_text())
        if summary["completed_trials"] != 48 or summary["failed_trials"] or summary.get("failed_evaluations"):
            raise ValueError("Incomplete small-model training or evaluation")
        state["stage"] = "bank_certificates"
        save_json(root / "status.json", state)
        bounds = root / "bounds"
        bound_command = ["journal.evaluate_bayesian_certificate", "--manifest", str(manifest), "--root", str(campaign),
                         "--output", str(bounds), "--draws", "1000", "--delta", ".025", "--threads", str(threads)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(command, [*bound_command, "--index", str(index), "--resume"],
                                   root / f"bank_{index}.log", environment) for index in range(48)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        command([*bound_command, "--aggregate-only"], root / "aggregate.log", environment)
        state["stage"] = "figures"
        save_json(root / "status.json", state)
        command(["journal.plot_bayesian_certificate", "--source", str(bounds / "risk_bounds.json"),
                 "--output", str(root / "figures")], root / "figures.log", environment)
        state.update(status="complete", stage=None)
        save_json(root / "status.json", state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(root / "status.json", state)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("journal/configs/bayesian_certificate.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/journal_bayesian_certificate/20261117"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, args.workers, args.threads), indent=2))


if __name__ == "__main__":
    main()
