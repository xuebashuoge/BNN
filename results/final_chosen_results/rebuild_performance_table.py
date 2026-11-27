"""Rebuild the fixed-prior (Q_W) m=2000 deployment table from the archived held-out evaluations.

The manuscript table now uses the fresh Q_{W|S}; see rebuild_final_tables.py.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "performance"
REPORTS = DATA / "selected_trials.tar.gz"
WEIGHTS = ROOT / "models" / "performance" / "weights.tar"
METHODS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched", "Proposed-Perfect")
DATASETS = {"moons": "Two moons", "mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
CHANNELS = {"Fading": {"kind": "diagonal_fading", "rho_c": .35}, "Erasure": {"kind": "outage", "outage_prob": .5}}


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed_observations(index):
    if file_sha256(REPORTS) != index["files"][str(REPORTS.relative_to(ROOT))]["sha256"]:
        raise ValueError("The selected trial archive fingerprint differs from its inventory")
    with tarfile.open(REPORTS, "r|gz") as trials:
        files = {member.name: trials.extractfile(member).read() for member in trials if member.isfile()}

    rows = []
    for cohort in index["cohorts"]:
        channel, dataset = cohort["channel"], cohort["dataset"]
        for method in METHODS:
            for seed in cohort["confirmation_seeds"]:
                key = f"{cohort['archive_directory']}/{method}/{seed}"
                entry = json.loads(files[key + "/manifest_entry.json"])
                metadata = files[key + "/metadata.json"]
                report = json.loads(files[key + "/test_deployment/evaluation.json"])
                cfg, mc = report["evaluation_config"], report["heldout_gibbs_mc"]
                law = CHANNELS[channel]
                train = {"kind": "perfect"} if method == "Proposed-Perfect" else law
                if (entry["method_label"], cfg["seed"], cfg["dataset"], cfg["m"], cfg["deployment_channel"],
                    cfg["train_channel"], report["split"], mc["channel_source"]) != (
                    method, seed, dataset, 2000, law, train, "test", "deployment"):
                    raise ValueError(f"Unexpected trial conditions for {key}")
                fingerprint = report["checkpoint_source_fingerprints"]
                if hashlib.sha256(metadata).hexdigest() != fingerprint["metadata"]["sha256"]:
                    raise ValueError(f"Checkpoint metadata differ for {key}")
                draws = [r["accuracy"] for r in mc["replicate_statistics"]]
                if len(draws) != mc["draws"] or not math.isclose(statistics.mean(draws), mc["accuracy"], abs_tol=1e-12):
                    raise ValueError(f"Gibbs replicate mean differs for {key}")
                rows.append(dict(dataset=dataset, channel=channel, method=method, seed=seed, m=cfg["m"],
                                 test_accuracy_pct=100 * mc["accuracy"], draws=mc["draws"], examples=mc["examples"],
                                 task_id=entry["task_id"], training_revision=json.loads(metadata)["git_sha"],
                                 archive_key=key, weights_sha256=fingerprint["weights"]["sha256"]))
    return rows


def render_table(rows):
    lines = [r"\begin{table*}[htb]", r"\centering",
             r"\caption{Deployment accuracy (\%) comparison at channel-set size $m=2000$.}",
             r"\label{tab:performance}", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{}}",
             r"\toprule", r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Channel}",
             r"& \multicolumn{4}{c}{Matched training} & \multirow{2}{*}{Perfect training} \\",
             r"\cmidrule(lr){3-6}", r"& & ERM & L2 & T1 & T2 &  \\"]
    for dataset, label in DATASETS.items():
        lines.extend([r"\midrule", r"\multirow{2}{*}{" + label + "}"])
        for channel in CHANNELS:
            values = [[r["test_accuracy_pct"] for r in rows if (r["dataset"], r["channel"], r["method"]) ==
                       (dataset, channel, method)] for method in METHODS]
            means = [statistics.fmean(v) for v in values]
            cells = []
            for values_by_seed, avg in zip(values, means):
                printed = f"{avg:.2f}"
                if max(means) - avg <= 1e-10:
                    printed = r"\mathbf{" + printed + "}"
                cells.append("$" + printed + r"\pm" + f"{statistics.stdev(values_by_seed):.2f}$")
            lines.extend(["& " + channel, "& " + " & ".join(cells) + r" \\"])
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])


def verify_weights(rows, index):
    if file_sha256(WEIGHTS) != index["files"][str(WEIGHTS.relative_to(ROOT))]["sha256"]:
        raise ValueError("The weights archive fingerprint differs from its inventory")
    expected = {r["archive_key"] + "/weights.pt": r["weights_sha256"] for r in rows}
    with tarfile.open(WEIGHTS, "r|") as weights:
        for member in weights:
            if hashlib.file_digest(weights.extractfile(member), "sha256").hexdigest() != expected.pop(member.name):
                raise ValueError(f"Checkpoint differs for {member.name}")
    if expected:
        raise ValueError(f"Missing checkpoints: {sorted(expected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare the table and per-seed CSV to the saved results")
    parser.add_argument("--verify-models", action="store_true", help="Also verify every checkpoint in the Spartan archive")
    args = parser.parse_args()
    index = json.loads((DATA / "provenance.json").read_text())
    rows = seed_observations(index)
    if args.verify_models:
        verify_weights(rows, index)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    for path, content in [(DATA / "deployment_by_seed.csv", stream.getvalue()),
                          (DATA / "performance_table_fixed_prior.tex", render_table(rows))]:
        if args.check:
            if path.read_text() != content:
                raise ValueError(f"Regenerated data differ from {path.name}")
        else:
            path.write_text(content)
    print(f"Verified 40 means and sample SDs from {len(rows)} held-out evaluations.")


if __name__ == "__main__":
    main()
