"""Build the all-Gaussian experiment matrix without opening data or training.

Selection consumes complete validation training records and their original
manifest. Test scores never enter selection. All declared seeds are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .build_campaign import make_entry


DATASETS = ("moons", "mnist", "fashionmnist", "cifar10")
CHANNELS = {
    "analog": dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.),
    "outage": dict(kind="outage", outage_prob=.5),
}
METHODS = (("erm", "ERM"), ("l2", "L2"), ("proposed", "T1"), ("joint", "T2"))
REGIMES = ("matched", "perfect")
M_GRID = (2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 200)
P_GRID = (0., .025, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.)
SEEDS = {
    "pilot": (72000,),
    "screen": (72101, 72102),
    "confirm": (73101, 73102, 73103, 73104, 73105),
    "m_sweep": (74101, 74102, 74103),
    "p_train_sweep": (75101, 75102, 75103),
}


def validate_seeds(seeds):
    if (not seeds or len(seeds) != len(set(seeds)) or
            any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        raise ValueError("Require distinct nonnegative integer seeds")
    return tuple(seeds)


def base_config(dataset, channel_name, stage="screen"):
    """Full-data settings avoid selecting with a different sample-size penalty."""
    if dataset not in DATASETS or channel_name not in CHANNELS:
        raise ValueError("Unknown dataset or channel")
    sizes = {"moons": (800, 200), "mnist": (55000, 5000),
             "fashionmnist": (55000, 5000), "cifar10": (45000, 5000)}
    epochs = {"moons": 150, "mnist": 30, "fashionmnist": 30, "cifar10": 25}
    # Screen/confirmation use the same training budget. Confirmation changes
    # only initialization, finite-bank and evaluation random-number streams.
    config = dict(dataset=dataset, architecture="cnn_spatial3" if dataset == "cifar10" else "mlp",
                  predictor="gaussian", method="erm", training_loss="bounded_ce_smooth",
                  loss_epsilon=.01, clip_ce=10., selection_metric="accuracy",
                  validation_seed_policy="fixed", prior_std=1., posterior_std=.001,
                  rho_lr_scale=.1, width=16 if dataset == "moons" else 128,
                  train_size=sizes[dataset][0], val_size=sizes[dataset][1],
                  epochs=1 if stage == "pilot" else epochs[dataset],
                  batch_size=128 if dataset in {"moons", "cifar10"} else 256,
                  m=20, lr=.001, eta_channel=.001, eta_data=.01, l2=1e-5,
                  data_seed=20261117, eval_draws=100, val_draws=5, delta=.05,
                  train_channel=dict(CHANNELS[channel_name]),
                  deployment_channel=dict(CHANNELS[channel_name]),
                  device="cpu", threads=2, cpu_flush_denormal=True,
                  download=False, moons_sampler="iid")
    return config


def method_variants(base, method):
    """Eight distinct trials per method and training regime, never test-tuned.

    ERM spends its eight trials on learning rate and initial posterior scale;
    the regularized methods divide the same budget across their coefficient.
    All methods optimize both Gaussian means and scales with rho LR factor .1.
    """
    if method == "erm":
        for lr in (.0003, .001, .003, .01):
            for posterior_std in (.001, .003):
                yield {**base, "method": method, "lr": lr, "posterior_std": posterior_std}
        return
    for lr in (.001, .003):
        for posterior_std in (.001, .003):
            for index in range(2):
                config = {**base, "method": method, "lr": lr, "posterior_std": posterior_std}
                if method == "l2":
                    config["l2"] = (1e-6, 1e-4)[index]
                else:
                    strength = (.001, .003)[index]
                    ratio = math.sqrt((base["train_size"] - 1) / (base["m"] - 1))
                    if base["train_channel"]["kind"] == "perfect":
                        # Both objectives specialize to the data term. Their
                        # eight candidates intentionally coincide in this case.
                        config.update(eta_channel=0., eta_data=(.003, .03)[index])
                    elif method == "proposed":
                        config.update(eta_channel=strength, eta_data=strength * ratio)
                    else:
                        factor = 2 / math.sqrt(1 + (base["m"] - 1) / (base["train_size"] - 1))
                        config.update(eta_channel=0., eta_data=strength * factor)
                yield config


def entry(config, seed, dataset, channel_name, label, stage, regime,
          experiment="main", axis=None, value=None, group=None):
    value_group = group or f"{dataset}_{channel_name}"
    item = make_entry(config, seed, value_group, label, f"bayes_{stage}",
                      "resource_pilot" if stage == "pilot" else
                      "mechanism" if experiment == "m_sweep" else
                      "mismatch" if experiment == "p_train_sweep" else "comparison")
    # Perfect-training T1/T2 remain proposed methods, not competing baselines.
    item["family"] = "proposed" if config["method"] in {"proposed", "joint"} else "baseline"
    item.update(experiment=experiment, channel_name=channel_name,
                training_regime=regime, sweep_axis=axis, sweep_value=value,
                predictor="single diagonal Gaussian; Gibbs posterior prediction")
    if stage == "pilot":
        item.update(scientific_result=False, cpu_pilot_target="screen_and_confirm_full_shape")
    return item


def screen_entries(seeds=SEEDS["screen"], datasets=DATASETS, channels=tuple(CHANNELS), pilot=False):
    validate_seeds(seeds)
    entries = []
    for dataset in datasets:
        for channel_name in channels:
            for regime in REGIMES:
                base = base_config(dataset, channel_name, "pilot" if pilot else "screen")
                if regime == "perfect":
                    base["train_channel"] = {"kind": "perfect"}
                for method, name in METHODS:
                    variants = list(method_variants(base, method))
                    if pilot:
                        variants = variants[:1]
                    label = f"{name}-{regime.title()}"
                    for config in variants:
                        for seed in seeds:
                            entries.append(entry(config, seed, dataset, channel_name, label,
                                                 "pilot" if pilot else "screen", regime,
                                                 "resource_pilot" if pilot else "main"))
    return entries


def read_entries(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("Empty screening manifest")
    return entries


def select_validation(entries, rows):
    """Reject partial/foreign/unequal screens before selecting every method."""
    from .campaign import select_variants, validate_entries
    validate_entries(entries)
    expected = {item["task_id"]: item for item in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate screening tasks")
    actual = {row["task_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("Selection requires exactly every screening training result")
    counts, seed_sets = defaultdict(Counter), defaultdict(dict)
    for task_id, source in expected.items():
        row = actual[task_id]
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete screening trial {task_id}")
        # Compare material settings, including the Gaussian predictor, rather
        # than trusting a result's label or a high validation score alone.
        from .run_experiment import normalize_config
        source_config = normalize_config(source["config"])
        result_config = normalize_config(row["config"])
        # Campaign launchers override execution resources. They cannot change
        # any scientific setting, including predictor, seeds or loss units.
        material = lambda config: {key: value for key, value in config.items()
                                   if key not in {"device", "threads"}}
        if material(source_config) != material(result_config):
            raise ValueError(f"Screening result config mismatch: {task_id}")
        if source.get("stage") != "bayes_screen" or source["config"].get("predictor") != "gaussian":
            raise ValueError("Selection requires the explicit all-Gaussian screening manifest")
        for key in ("group", "method_label", "variant_id", "family", "seed"):
            expected_value = source["config"]["seed"] if key == "seed" else source[key]
            if row.get(key) != expected_value:
                raise ValueError(f"Screening result {key} mismatch: {task_id}")
        if not all(math.isfinite(float(row.get(key, float("nan"))))
                   for key in ("validation_accuracy", "validation_cross_entropy")):
            raise ValueError("Nonfinite validation metric")
        group, label, variant = source["group"], source["method_label"], source["variant_id"]
        counts[group][label] += 1
        seed_sets[group].setdefault((label, variant), set()).add(source["config"]["seed"])
    expected_labels = {f"{name}-{regime.title()}" for _, name in METHODS for regime in REGIMES}
    for group, labels in counts.items():
        if set(labels) != expected_labels or len(set(labels.values())) != 1:
            raise ValueError(f"Unequal or incomplete method tuning budget in {group}")
        common = None
        variants_by_label = Counter(label for label, _ in seed_sets[group])
        if set(variants_by_label.values()) != {8}:
            raise ValueError(f"Require eight variants per method/regime in {group}")
        for seeds in seed_sets[group].values():
            if len(seeds) < 2 or (common is not None and seeds != common):
                raise ValueError("All variants require the same complete paired screening seeds")
            common = seeds
    _, selected = select_variants(rows, entries, 1)
    if len(selected) != len(counts) * len(expected_labels):
        raise ValueError("Missing validation-selected methods")
    return selected


def selected_entries(selected, seeds, stage, m_grid=M_GRID, p_grid=P_GRID):
    validate_seeds(seeds)
    screening_seeds = {seed for item in selected for seed in item["seeds"]}
    if screening_seeds & set(seeds):
        raise ValueError("Confirmation/sweep seeds must be disjoint from every screening seed")
    output = []
    for source in selected:
        config = dict(source["config"])
        dataset = config["dataset"]
        channel_name = next((name for name, law in CHANNELS.items() if config["deployment_channel"] == law), None)
        if channel_name is None:
            raise ValueError("Unknown deployment channel in selection")
        label = source["method_label"]
        regime = "perfect" if label.endswith("Perfect") else "matched"
        if stage != "confirm" and regime != "matched":
            continue
        if stage == "p_train_sweep" and channel_name != "outage":
            continue
        points = [None] if stage == "confirm" else m_grid if stage == "m_sweep" else p_grid
        for point in points:
            candidate = dict(config)
            if stage != "confirm":
                # Mechanism curves deliberately use smaller image-data budgets;
                # within each curve n, epochs and optimizer updates stay fixed.
                if dataset != "moons":
                    candidate.update(train_size=10000, val_size=5000,
                                     epochs=25 if dataset == "cifar10" else 30)
            group = f"{dataset}_{channel_name}"
            experiment, axis, item_label = "main", None, label
            item_regime = regime
            if stage == "m_sweep":
                candidate["m"] = int(point)
                group += f"_m{int(point)}"
                experiment, axis = stage, "m"
            elif stage == "p_train_sweep":
                candidate["train_channel"] = ({"kind": "perfect"} if point == 0 else
                                               {"kind": "outage", "outage_prob": float(point)})
                group += f"_ptr{round(float(point) * 1000):04d}"
                experiment, axis = stage, "p_train"
                item_label = label.removesuffix("-Matched")
                item_regime = "perfect" if point == 0 else "matched" if point == .5 else "mismatch"
            # Hyperparameters stay fixed over each curve, including eta_data
            # and eta_channel, so each x point has identical tuning budget.
            for seed in seeds:
                item = entry(candidate, seed, dataset, channel_name, item_label, stage,
                             item_regime, experiment, axis, point, group)
                item.update(selected_from_validation_variant=source["variant_id"],
                            selection_source="mean validation accuracy over every declared screening seed; CE tiebreak",
                            screening_seed_set=source["seeds"])
                output.append(item)
    if not output:
        raise ValueError("No selected tasks for requested stage")
    return output


def make_plan(entries, stage, seeds, source_hashes=None):
    return dict(stage=stage, tasks=len(entries), seeds=list(seeds),
                seed_streams="initialization=seed; bank=seed+10000; evaluation=seed+20000",
                datasets=sorted({item["config"]["dataset"] for item in entries}),
                channels=sorted({item["channel_name"] for item in entries}),
                groups=sorted({item["group"] for item in entries}),
                methods=sorted({item["method_label"] for item in entries}),
                predictor="All methods use a single diagonal Gaussian posterior",
                tuning_variants_per_method_regime=8 if stage == "screen" else None,
                tuning_seeds=list(seeds) if stage == "screen" else sorted({seed for item in entries for seed in item.get("screening_seed_set", SEEDS["screen"])}),
                confirmation_seeds=list(SEEDS["confirm"]),
                m_grid=list(M_GRID), p_train_grid=list(P_GRID),
                selection="Complete paired screening validation only; no test-score or favorable-seed selection",
                all_trial_outcomes_retained=True, test_evaluation_during_screen=False,
                source_sha256=source_hashes or {},
                sweep_controls="Selected matched hyperparameters, data split and optimizer updates fixed across each curve; image sweeps use n=10000 and 30 epochs (CIFAR 25), full-data main comparison is separate",
                training_loss="Smooth bounded CE; tuned eta values are bound-inspired surrogates, not eta=1 bound minimization",
                training_budget="Fixed budget: moons150 epochs; grayscale30; CIFAR25. Convergence is not assumed.",
                perfect_specialization="T1 and T2 have identical candidate objectives under perfect training; retain both labels transparently",
                bound_note="Finite KL permits evaluation; nonvacuity and mismatch shift validity require separate checking",
                resource_note="Full-shape one-epoch CPU pilot first. Historical CNN full-data epochs ~7-9s on 2 CPUs; cnn_spatial3 and new Gaussian ERM/L2 costs require the matching pilot. Screen contains 1024 full-data trials; use bounded CPU array concurrency and measured pilot time/RSS. No GPU requested.",
                figure_formats=["png", "pdf", "eps"], confidence_shading=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(SEEDS), default="pilot")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--channels", default=",".join(CHANNELS))
    parser.add_argument("--seeds", help="Explicit paired seeds; defaults are preregistered by stage")
    parser.add_argument("--from-manifest", type=Path, help="Original all-Gaussian screening JSONL")
    parser.add_argument("--from-results", type=Path, help="Complete training_results.json; test_results.json is never read")
    parser.add_argument("--m-grid", help="Optional explicit comma-separated bank sizes >1")
    parser.add_argument("--p-grid", help="Optional explicit comma-separated outage probabilities in [0,1]")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets, channels = args.datasets.split(","), args.channels.split(",")
    if len(datasets) != len(set(datasets)) or not set(datasets) <= set(DATASETS):
        parser.error("Datasets must be distinct names from " + ",".join(DATASETS))
    if len(channels) != len(set(channels)) or not set(channels) <= set(CHANNELS):
        parser.error("Channels must be distinct names from " + ",".join(CHANNELS))
    seeds = validate_seeds(tuple(int(value) for value in args.seeds.split(",")) if args.seeds else SEEDS[args.stage])
    source_hashes = {}
    if args.stage in {"pilot", "screen"}:
        entries = screen_entries(seeds, datasets, channels, args.stage == "pilot")
    else:
        if not args.from_manifest or not args.from_results:
            parser.error("Selected stages require --from-manifest and --from-results")
        selected = select_validation(read_entries(args.from_manifest), json.loads(args.from_results.read_text()))
        selected = [item for item in selected if item["config"]["dataset"] in datasets and
                    any(item["config"]["deployment_channel"] == CHANNELS[name] for name in channels)]
        m_grid = tuple(int(value) for value in args.m_grid.split(",")) if args.m_grid else M_GRID
        p_grid = tuple(float(value) for value in args.p_grid.split(",")) if args.p_grid else P_GRID
        if not m_grid or any(value <= 1 for value in m_grid) or len(set(m_grid)) != len(m_grid):
            parser.error("m grid requires distinct integers >1")
        if not p_grid or any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_grid) or len(set(p_grid)) != len(p_grid):
            parser.error("p grid requires distinct probabilities in [0,1]")
        entries = selected_entries(selected, seeds, args.stage, m_grid, p_grid)
        source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.from_manifest, args.from_results)}
    from .campaign import validate_entries
    validate_entries(entries)
    output = args.output or Path(f"journal/configs/bayesian_{args.stage}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in entries))
    plan = make_plan(entries, args.stage, seeds, source_hashes)
    if args.m_grid:
        plan["m_grid"] = [int(value) for value in args.m_grid.split(",")]
    if args.p_grid:
        plan["p_train_grid"] = [float(value) for value in args.p_grid.split(",")]
    plan["manifest_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
