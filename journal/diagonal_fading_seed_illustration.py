"""Freeze a validation-selected CIFAR-10 seed illustration and evaluate both screen seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .campaign import completed_trial
from .diagonal_fading_comparison import ranking
from .full_fading_comparison import METHODS, write_once
from .run_experiment import evaluate_checkpoint, file_fingerprint, save_json
from .run_task import read_manifest

SEEDS = (122101, 122102)
DEPLOYMENT = {"kind": "diagonal_fading", "rho_c": .35}
DRAWS = 100
FRESH_OFFSET = 3_000_019
SUBDIR = "seed-illustration"


def prepare(root):
    selection_path = root / "confirm.selection.json"
    screen_path = root / "screen.jsonl"
    selection = json.loads(selection_path.read_text())
    if (selection["stage"] != "confirm" or selection["screening_seeds"] != list(SEEDS)
            or selection["sources"]["screen"] != file_fingerprint(screen_path)):
        raise ValueError("Require the original validation selection and matching screen manifest")
    selected = {item["method_label"]: item["variant_id"] for item in selection["selected_variants"]}
    if len(selection["selected_variants"]) != len(METHODS) or set(selected) != set(METHODS):
        raise ValueError("Require seven original selected variants")
    scores = {seed: {} for seed in SEEDS}
    tasks = []
    for item in read_manifest(screen_path):
        label, config = item["method_label"], item["config"]
        if item["variant_id"] != selected[label]:
            continue
        seed = config["seed"]
        if (seed not in SEEDS or label in scores[seed] or config["dataset"] != "cifar10"
                or config["m"] != 2000 or config["deployment_channel"] != DEPLOYMENT
                or config["train_channel"] != ({"kind": "perfect"} if METHODS[label][1] else DEPLOYMENT)):
            raise ValueError(f"Unexpected or duplicate screening case: {item['task_id']}")
        checkpoint = root / "screen" / item["task_id"]
        validation = completed_trial(checkpoint, config)
        if not validation or not 0 <= validation["validation_accuracy"] <= 1:
            raise ValueError(f"Missing completed validation checkpoint: {checkpoint}")
        scores[seed][label] = validation["validation_accuracy"]
        tasks.append(dict(seed=seed, method=label, variant_id=item["variant_id"], task_id=item["task_id"],
                          validation_accuracy=validation["validation_accuracy"],
                          eval_seed=config["eval_seed"]+FRESH_OFFSET,
                          source={key: file_fingerprint(checkpoint / filename) for key, filename in
                                  (("weights", "weights.pt"), ("metadata", "metadata.json"))}))
    if any(set(cohort) != set(METHODS) for cohort in scores.values()):
        raise ValueError("Incomplete common screening cohort")
    ranks = {seed: ranking(scores[seed]) for seed in SEEDS}
    winner = max(SEEDS, key=lambda s: (ranks[s]["margin"], -s))
    protocol = dict(dataset="cifar10", m=2000, deployment_channel=DEPLOYMENT, eval_draws=DRAWS,
                    selection_rule="Maximize validation accuracy of the better matched T1/T2 minus the best of the other five methods; smaller seed breaks a tie.",
                    validation_selected_seed=winner, validation_ranking=ranks, validation_accuracy=scores,
                    screen=file_fingerprint(screen_path), original_selection=file_fingerprint(selection_path),
                    selected_variants=selected, seeds=SEEDS, fresh_evaluation_seed_offset=FRESH_OFFSET,
                    tasks=sorted(tasks, key=lambda row: (SEEDS.index(row["seed"]), tuple(METHODS).index(row["method"]))))
    output = root / SUBDIR
    output.mkdir(exist_ok=True)
    write_once(output / "protocol.json", json.dumps(protocol, indent=2, allow_nan=False)+"\n")
    return dict(validation_selected_seed=winner, validation_ranking=ranks)


def read_protocol(root):
    path = root / SUBDIR / "protocol.json"
    protocol = json.loads(path.read_text())
    if protocol["seeds"] != list(SEEDS) or protocol["eval_draws"] != DRAWS or len(protocol["tasks"]) != 14:
        raise ValueError("The frozen protocol does not match this experiment")
    return protocol


def evaluate(root, index, threads):
    protocol = read_protocol(root)
    if not 0 <= index < len(protocol["tasks"]):
        raise ValueError("Task index must be in 0..13")
    item = protocol["tasks"][index]
    checkpoint = root / "screen" / item["task_id"]
    case = root / SUBDIR / "trials" / item["task_id"]
    case.mkdir(parents=True, exist_ok=True)
    if not (case / "evaluation.json").is_file():
        overrides = case / "overrides.json"
        write_once(overrides, json.dumps(dict(eval_draws=DRAWS, eval_seed=item["eval_seed"],
                                              device="cpu", threads=threads))+"\n")
        evaluate_checkpoint(checkpoint, str(overrides), case, "test", "deployment")
    return checked_result(root, item)


def checked_result(root, item):
    path = root / SUBDIR / "trials" / item["task_id"] / "evaluation.json"
    r = json.loads(path.read_text())
    config, risk = r["evaluation_config"], r["heldout_gibbs_mc"]
    if (r["split"] != "test" or r["risk_scope"] != "deployment" or risk["channel_source"] != "deployment"
            or r["checkpoint_source_fingerprints"] != item["source"]
            or r["checkpoint_verification"]["bank_hash_matches_metadata"] is not True
            or config["seed"] != item["seed"] or config["eval_seed"] != item["eval_seed"]
            or config["deployment_channel"] != DEPLOYMENT or config["m"] != 2000
            or risk["examples"] != 10000 or risk["draws"] != DRAWS or not risk["predictor"]["stochastic"]
            or not 0 <= risk["accuracy"] <= 1):
        raise ValueError(f"Frozen test evaluation differs from protocol: {path}")
    return dict(method=item["method"], seed=item["seed"], validation_accuracy=item["validation_accuracy"],
                test_accuracy=risk["accuracy"], evaluation_sha256=file_fingerprint(path)["sha256"])


def report(root):
    protocol = read_protocol(root)
    selected = protocol["validation_selected_seed"]
    cells = [checked_result(root, item) for item in protocol["tasks"]]
    by_seed = {seed: {r["method"]: r["test_accuracy"] for r in cells if r["seed"] == seed} for seed in SEEDS}
    means = {label: statistics.fmean(by_seed[s][label] for s in SEEDS) for label in METHODS}
    rankings = {seed: ranking(by_seed[seed]) for seed in SEEDS}
    rows = []
    for label in METHODS:
        row = dict(method=label, variant_id=protocol["selected_variants"][label])
        for seed in SEEDS:
            row[f"validation_pct_s{seed}"] = 100*protocol["validation_accuracy"][str(seed)][label]
            row[f"test_pct_s{seed}"] = 100*by_seed[seed][label]
        row["test_mean_pct_two_seeds"] = 100*means[label]
        rows.append(row)
    output = root / SUBDIR
    result = dict(scope="Exploratory original-screen seed illustration. The mean describes both screened seeds. The independent three-seed confirmation is a separate comparison.",
                  units="CSV accuracies are percentages; JSON accuracy scores and margins are fractions.",
                  validation_selected_seed=selected, validation_ranking=protocol["validation_ranking"],
                  test_ranking_by_seed=rankings, test_ranking_two_seed_mean=ranking(means),
                  m=2000, deployment_channel=DEPLOYMENT, seeds=SEEDS, test_draws=DRAWS, test_examples=10000,
                  protocol=file_fingerprint(output / "protocol.json"), rows=rows, evaluations=cells)
    save_json(output / "seed_illustration.json", result)
    with (output / "seed_illustration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {key: result[key] for key in ("validation_selected_seed", "test_ranking_by_seed", "test_ranking_two_seed_mean")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "evaluate":
        if args.index is None or args.threads < 1:
            parser.error("Evaluation requires --index and positive --threads")
        result = evaluate(args.root, args.index, args.threads)
    else:
        result = report(args.root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
