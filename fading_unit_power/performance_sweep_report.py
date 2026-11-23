"""Report paired held-out performance from a completed unit-power sweep."""

import csv
import io
import json
import random
import statistics
from pathlib import Path

from fading_unit_power import performance


METHODS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched")
BASELINES, PROPOSED = METHODS[:2], METHODS[2:]
DATASETS = performance.historical_table.DATASETS
SEED_FIELDS = {"seed", "channel_seed", "eval_seed"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 18171


def file_record(path: Path) -> dict:
    return dict(sha256=performance.historical_table.file_sha256(path), bytes=path.stat().st_size)


def read_checkpoint_inventory(path: Path) -> dict:
    by_dataset = {dataset: {} for dataset in DATASETS}
    task_ids = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        entry = record["manifest_entry"]
        task_id = entry["task_id"]
        dataset = entry["config"]["dataset"]
        if task_id in task_ids or dataset not in by_dataset:
            raise ValueError(f"Unexpected checkpoint inventory entry for {task_id}")
        task_ids.add(task_id)
        by_dataset[dataset][task_id] = record
    return by_dataset


def paired_difference(proposed: dict, baseline: dict) -> dict:
    by_seed = {seed: proposed[seed] - baseline[seed] for seed in proposed}
    values = tuple(by_seed.values())
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrapped = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    quantiles = statistics.quantiles(bootstrapped, n=40, method="inclusive")
    return dict(
        mean_pp=statistics.fmean(values),
        sample_sd_pp=statistics.stdev(values),
        paired_bootstrap_95_interval_pp=[quantiles[0], quantiles[-1]],
        wins=sum(value > 0 for value in values),
        ties=sum(value == 0 for value in values),
        losses=sum(value < 0 for value in values),
        by_seed_pp=by_seed,
    )


def method_results(rows: list[dict], entries: list[dict]) -> dict:
    results = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        method_entries = [entry for entry in entries if entry["method"] == method]
        configs = [
            {key: value for key, value in entry["config"].items() if key not in SEED_FIELDS}
            for entry in method_entries
        ]
        candidates = {entry["candidate"] for entry in method_entries}
        if len(candidates) != 1 or any(config != configs[0] for config in configs):
            raise ValueError(f"Confirmation settings changed across paired seeds for {method}")
        values = [row["test_accuracy_pct"] for row in method_rows]
        results[method] = dict(
            mean_accuracy_pct=statistics.fmean(values),
            sample_sd_accuracy_pct=statistics.stdev(values),
            candidate=method_entries[0]["candidate"],
            selected_config=configs[0],
            selected_epoch_by_seed={row["seed"]: row["selected_epoch"] for row in method_rows},
        )
    return results


def read_cohort(output: Path, dataset: str, recorded: dict | None = None) -> tuple[dict, list[dict], list[dict]]:
    selection_path = output / "manifests" / f"{dataset}-sweep-selection.json"
    selection = performance.read_json(selection_path)
    manifest = performance.manifest_path(output, dataset, "confirmation")
    manifest_sha256 = performance.historical_table.file_sha256(manifest)
    bank_size = selection["chosen_bank_size"]
    proposed, baseline = selection["chosen_proposed"], selection["chosen_baseline"]
    if (selection["dataset"] != dataset or selection["confirmation_manifest_sha256"] != manifest_sha256
            or proposed not in PROPOSED or baseline not in BASELINES):
        raise ValueError(f"Invalid frozen selection for {dataset}")

    entries = performance.read_manifest(manifest)
    seeds = sorted({entry["config"]["seed"] for entry in entries})
    keys = {(entry["method"], entry["config"]["seed"]) for entry in entries}
    if (len(seeds) < 2 or len(entries) != len(keys)
            or keys != {(method, seed) for method in METHODS for seed in seeds}):
        raise ValueError(f"Expected complete paired confirmation for {dataset}")
    if sorted(selection.get("confirmation_seeds", seeds)) != seeds:
        raise ValueError(f"Confirmation seeds differ from the frozen selection for {dataset}")
    if recorded is not None and recorded.keys() != {entry["task_id"] for entry in entries}:
        raise ValueError(f"Checkpoint inventory differs from the confirmation manifest for {dataset}")

    rows, checksums = [], []
    eval_settings = {}
    for entry in entries:
        config = entry["config"]
        seed = config["seed"]
        if (entry["phase"], config["dataset"], config["m"]) != ("confirmation", dataset, bank_size):
            raise ValueError(f"Unexpected confirmation conditions for {entry['task_id']}")
        report = performance.evaluate_result(output, entry)
        evaluated = report["evaluation_config"]
        mc = report["heldout_gibbs_mc"]
        settings = (config["data_seed"], config["channel_seed"], evaluated["eval_seed"],
                    mc["draws"], mc["examples"])
        if settings != eval_settings.setdefault(seed, settings):
            raise ValueError(f"Unpaired data, channel, or evaluation settings for {dataset}/seed {seed}")

        directory = performance.task_dir(output, entry)
        weights = directory / "weights.pt"
        metadata_path = directory / "metadata.json"
        expected = report["checkpoint_source_fingerprints"]
        metadata_record = file_record(metadata_path)
        if recorded is None or weights.exists() or weights.is_symlink():
            weights_record = file_record(weights)
        else:
            weights_record = expected["weights"]
        if expected != dict(weights=weights_record, metadata=metadata_record):
            raise ValueError(f"Checkpoint fingerprints differ from the held-out report for {entry['task_id']}")
        metadata = performance.read_json(metadata_path)
        epoch = report["selected_epoch"]
        if epoch != metadata["selected_epoch"] or not 1 <= epoch <= config["epochs"]:
            raise ValueError(f"Unexpected selected epoch for {entry['task_id']}")

        archive_key = directory.relative_to(output).as_posix()
        row = dict(
            dataset=dataset, channel="Fading", method=entry["method"], seed=seed, m=bank_size,
            candidate=entry["candidate"], learning_rate=config["lr"], lr_decay=evaluated["lr_decay"],
            epochs=config["epochs"], selected_epoch=epoch, train_mc=evaluated["train_mc"],
            val_draws=config["val_draws"], eval_seed=evaluated["eval_seed"], channel_seed=config["channel_seed"],
            draws=mc["draws"], examples=mc["examples"], test_accuracy_pct=100 * mc["accuracy"],
            test_cross_entropy=mc["cross_entropy"], task_id=entry["task_id"],
            training_revision=metadata["git_sha"], archive_key=archive_key,
            weights_sha256=weights_record["sha256"], weights_bytes=weights_record["bytes"],
        )
        rows.append(row)
        record = dict(
            manifest_entry=entry, archive_key=archive_key,
            weights=dict(path=weights.relative_to(output).as_posix(), **weights_record),
            metadata=dict(path=metadata_path.relative_to(output).as_posix(), **metadata_record),
        )
        if recorded is not None and record != recorded[entry["task_id"]]:
            raise ValueError(f"Checkpoint inventory differs from the held-out report for {entry['task_id']}")
        checksums.append(record)

    stats = method_results(rows, entries)
    accuracies = {
        method: {row["seed"]: row["test_accuracy_pct"]
                 for row in sorted(rows, key=lambda row: row["seed"]) if row["method"] == method}
        for method in METHODS
    }
    differences = {
        method: {control: paired_difference(accuracies[method], accuracies[control]) for control in BASELINES}
        for method in PROPOSED
    }
    test_best_proposed = max(PROPOSED, key=lambda method: stats[method]["mean_accuracy_pct"])
    test_best_baseline = max(BASELINES, key=lambda method: stats[method]["mean_accuracy_pct"])
    frozen_gaps = differences[proposed]
    summary = dict(
        bank_size=bank_size, paired_test_seeds=seeds, methods=stats, paired_differences=differences,
        validation_selection_path=selection_path.relative_to(output).as_posix(),
        validation_selection_sha256=performance.historical_table.file_sha256(selection_path),
        confirmation_manifest_sha256=manifest_sha256,
        validation_selected=dict(
            proposed=proposed, baseline=baseline, paired_gap=frozen_gaps[baseline],
            strongest_test_baseline=test_best_baseline,
            gap_to_strongest_test_baseline_pp=frozen_gaps[test_best_baseline]["mean_pp"],
            beats_both_baselines=all(gap["mean_pp"] > 0 for gap in frozen_gaps.values()),
        ),
        descriptive_test_best=dict(
            proposed=test_best_proposed, baseline=test_best_baseline,
            mean_gap_pp=differences[test_best_proposed][test_best_baseline]["mean_pp"],
        ),
    )
    return summary, rows, checksums


def render_table(datasets: dict) -> str:
    lines = [
        r"\begin{table*}[htb]",
        r"\centering",
        r"\caption{Held-out deployment accuracy (\%) under unit-power fading. "
        r"Entries show mean and sample standard deviation across paired training seeds. "
        r"The final column gives the difference in percentage points between the validation-selected "
        r"proposed method and the stronger baseline on the test set.}",
        r"\label{tab:unit-power-sweep}",
        r"\begin{tabular}{lrccccc}",
        r"\toprule",
        r"Dataset & $m$ & ERM & L2 & T1 & T2 & Gap (pp) \\",
        r"\midrule",
    ]
    for dataset, result in datasets.items():
        cells = [DATASETS[dataset], str(result["bank_size"])]
        top = max(scores["mean_accuracy_pct"] for scores in result["methods"].values())
        for method in METHODS:
            scores = result["methods"][method]
            mean = scores["mean_accuracy_pct"]
            printed = f"{mean:.2f}"
            if mean == top:
                printed = r"\mathbf{" + printed + "}"
            cells.append("$" + printed + r"\pm" + f"{scores['sample_sd_accuracy_pct']:.2f}$")
        gap = result["validation_selected"]["gap_to_strongest_test_baseline_pp"]
        cells.append(f"${gap:+.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def summarize(output: Path, dataset: str | None = None, *, recorded_checkpoints: Path | None = None) -> None:
    """Validate local weights or replay reports using an archived checkpoint inventory."""
    output = output.resolve()
    datasets = (dataset,) if dataset else tuple(DATASETS)
    inventory = None if recorded_checkpoints is None else read_checkpoint_inventory(recorded_checkpoints)
    results, rows, checkpoints = {}, [], []
    for current in datasets:
        recorded = None if inventory is None else inventory[current]
        summary, seed_rows, checksums = read_cohort(output, current, recorded)
        results[current] = summary
        rows.extend(seed_rows)
        checkpoints.extend(checksums)
        selected = summary["validation_selected"]
        gap = selected["gap_to_strongest_test_baseline_pp"]
        print(f"{current}: {selected['proposed']} vs {selected['strongest_test_baseline']}, {gap:+.3f} pp")

    suffix = f"-{dataset}" if dataset else ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    performance.write_text(output / f"deployment_by_seed{suffix}.csv", stream.getvalue())
    performance.write_json(output / f"performance_summary{suffix}.json", dict(
        channel="Unit-power diagonal fading",
        paired_interval=dict(
            level=0.95, method="percentile bootstrap of paired training seeds",
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, multiplicity_adjusted=False,
        ),
        all_reported_datasets_have_validation_selected_proposed_wins=all(
            result["validation_selected"]["beats_both_baselines"] for result in results.values()
        ),
        datasets=results,
    ))
    performance.write_text(output / f"performance_table{suffix}.tex", render_table(results))
    performance.write_text(output / f"selected_checkpoints{suffix}.jsonl", "".join(
        json.dumps(record, allow_nan=False) + "\n" for record in checkpoints
    ))
