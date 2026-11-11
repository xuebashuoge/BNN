"""Publish CIFAR-10 tables/plot from complete saved campaigns only.

This wrapper performs metadata validation and reporting. It never trains a
network, loads model weights, runs checkpoint inference, or submits jobs.
Both stages must be complete before any report or publication file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
DEFAULT_SCREEN = REPO / "results/journal_cifar/screen"
DEFAULT_CONFIRMATION = REPO / "results/journal_cifar/confirm_cpu"
DEFAULT_REPORTS = REPO / "results/journal_cifar/paper_assets"


def relative(path):
    """Use lexical repo-relative paths, preserving source directory links."""
    return Path(os.path.relpath(str(path), REPO)).as_posix()


def normalize_paths(value):
    if isinstance(value, dict):
        return {normalize_paths(key): normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return relative(value)
    return value


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def single_column_table(tex, label):
    """Keep numerical cells/caption unchanged while fitting an IEEE column."""
    if tex.count(r"\begin{tabular}{lcc}") != 1:
        raise ValueError("A stage report must have exactly three table columns")
    if tex.count(r"\label{tab:sim_cifar}") != 1:
        raise ValueError("Unexpected CIFAR report table label")
    tex, count = re.subn(r"\\begin\{table\*?\}\[t\]", r"\\begin{table}[t]", tex)
    if count != 1:
        raise ValueError("Expected one complete report table")
    tex, count = re.subn(r"\\end\{table\*?\}", r"\\end{table}", tex)
    if count != 1:
        raise ValueError("Expected one report table ending")
    tex, count = re.subn(
        r"\\centering\\(?:small|footnotesize)",
        r"\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}", tex)
    if count != 1:
        raise ValueError("Expected the report table's centering/font declaration")
    tex = tex.replace(r"\label{tab:sim_cifar}", rf"\label{{{label}}}")
    if label == "tab:app_cifar_screen":
        tex = tex.replace(r"\begin{table}[t]", r"\begin{table}[!htbp]")
    return tex


def write_bytes_atomic(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def publish(screen, confirmation, report_root):
    # Import only the metadata reader; plotting remains in the two subprocesses.
    # Direct script execution otherwise places journal/paper, not REPO, on sys.path.
    sys.path.insert(0, str(REPO))
    from journal.report_cifar import load_campaign

    stages = (("screen", "Screen", Path(screen).resolve()),
              ("confirmation", "Full data", Path(confirmation).resolve()))
    verified = {}
    for stage, title, campaign_root in stages:
        sources = {}
        try:
            methods = load_campaign(campaign_root, sources)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{title} campaign is missing or incomplete: {error}") from error
        verified[stage] = dict(methods=methods, source_hashes=sources)

    # No directories/publication assets are created until BOTH preflights pass.
    report_root = Path(report_root).resolve()
    reports = {}
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(report_root / "matplotlib-cache")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    for stage, title, campaign_root in stages:
        output = report_root / stage
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "report.log"
        command = [sys.executable, "-m", "journal.report_cifar", "--campaign",
                   f"{title}={campaign_root}", "--output", str(output)]
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=REPO, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{title} report failed; see {relative(log_path)}")
        payload = json.loads((output / "cifar_results.json").read_text())
        if payload.get("schema_version", 0) < 2:
            raise ValueError("Expected the compact, source-hashed CIFAR report schema")
        if payload["source_hashes"] != verified[stage]["source_hashes"]:
            raise ValueError(f"{title} source files changed while generating the report")
        reports[stage] = dict(payload=payload, output=output, title=title,
                              campaign_root=campaign_root)

    publication = {}
    specifications = (
        ("confirmation", "cifar_comparison", "tab:sim_cifar"),
        ("screen", "appendix_cifar_screen", "tab:app_cifar_screen"),
    )
    for stage, basename, label in specifications:
        output = reports[stage]["output"]
        tex = single_column_table((output / "cifar_results.tex").read_text(), label)
        publication[PAPER / "tables" / f"{basename}.tex"] = tex.encode()
        publication[PAPER / "tables" / f"{basename}.csv"] = (output / "cifar_results.csv").read_bytes()
    for extension in ("pdf", "png"):
        publication[PAPER / "assets" / f"fig_cifar_validation.{extension}"] = (
            reports["confirmation"]["output"] / f"cifar_validation.{extension}").read_bytes()

    source_hashes, compact_stages = {}, {}
    for stage, report in reports.items():
        payload, title = report["payload"], report["title"]
        for path, description in payload["source_hashes"].items():
            name = relative(path)
            if name in source_hashes and source_hashes[name] != description:
                raise ValueError(f"Conflicting source hash: {name}")
            source_hashes[name] = description
        compact_stages[stage] = dict(
            campaign_root=relative(report["campaign_root"]),
            report_json=relative(report["output"] / "cifar_results.json"),
            report_sha256=sha256((report["output"] / "cifar_results.json").read_bytes()),
            protocol=payload["protocols"][title], methods=payload["campaigns"][title])
    manifest = dict(
        schema_version=1, generated_by=relative(Path(__file__)),
        stages=normalize_paths(compact_stages), source_hashes=source_hashes,
        publication_files={relative(path): dict(sha256=sha256(contents), bytes=len(contents))
                           for path, contents in publication.items()},
        selection="Aggregate-validation-selected configurations; every selected seed retained",
        uncertainty="Sample SD over initialization/channel-bank seeds conditional on fixed data; not MC draws",
        source_policy="Screen and full-data stages stay separate; source paths are repo-relative; full histories stay at hashed inputs",
        computations="Metadata validation/reporting/plotting only; no model weights, inference, training or job submission")
    manifest_path = PAPER / "cifar_data_manifest.json"
    publication[manifest_path] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    # Materialize all bytes above before publishing any paper destination.
    for path, contents in publication.items():
        write_bytes_atomic(path, contents)
    print(json.dumps(dict(publication_files=[relative(path) for path in publication],
                          source_files=len(source_hashes),
                          report_roots={stage: relative(report["output"])
                                        for stage, report in reports.items()}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS,
                        help="Writable ignored directory for the two complete metadata reports")
    args = parser.parse_args()
    try:
        publish(args.screen, args.confirmation, args.report_root)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"CIFAR publication assets not generated: {error}\n")


if __name__ == "__main__":
    main()
