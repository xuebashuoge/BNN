"""Declare a separate small-model all-Gaussian bound/decomposition diagnostic.

This builder never opens datasets, trains, selects hyperparameters, or reads
results. Its width8/n512/m2000 setting follows the earlier toy certificate's
scale, with a newly declared common Gaussian architecture for all four methods.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .build_bayesian_campaign import CHANNELS, METHODS, REGIMES
from .build_campaign import make_entry


SEEDS = (76101, 76102, 76103)
SCIENTIFIC_SETUP = "small_model_bound_diagnostic"


def diagnostic_config(channel_name, regime, method):
    if channel_name not in CHANNELS or regime not in REGIMES or method not in {item[0] for item in METHODS}:
        raise ValueError("Unknown method, channel law or training regime")
    channel = dict(CHANNELS[channel_name])
    config = dict(dataset="moons", architecture="mlp", predictor="gaussian", method=method,
                  training_loss="bounded_ce_smooth", loss_epsilon=.01, clip_ce=10.,
                  selection_metric="accuracy", validation_seed_policy="fixed",
                  width=8, train_size=512, val_size=512, m=2000,
                  epochs=150, batch_size=128, lr=.003, posterior_std=.1, prior_std=1.,
                  rho_lr_scale=.1, l2=1e-4, eta_channel=0., eta_data=0.,
                  eval_draws=20, val_draws=5, delta=.05, data_seed=20261117,
                  device="cpu", threads=2, cpu_flush_denormal=True, download=False,
                  moons_sampler="iid", moon_noise=.3,
                  train_channel={"kind": "perfect"} if regime == "perfect" else channel,
                  deployment_channel=dict(channel))
    if method == "proposed":
        config.update(eta_channel=.01, eta_data=.03)
    elif method == "joint":
        config["eta_data"] = .03
    return config


def diagnostic_entries():
    entries = []
    for channel_name in CHANNELS:
        group = f"moons_{channel_name}_certificate_w8_n512_m2000"
        for regime in REGIMES:
            for method, name in METHODS:
                config = diagnostic_config(channel_name, regime, method)
                for seed in SEEDS:
                    item = make_entry(config, seed, group, f"{name}-{regime.title()}",
                                      "bayes_certificate", "small_model_certificate")
                    item.update(family="proposed" if method in {"proposed", "joint"} else "baseline",
                                experiment="main", channel_name=channel_name, training_regime=regime,
                                sweep_axis=None, sweep_value=None, scientific_setup=SCIENTIFIC_SETUP,
                                main_accuracy_table=False, hyperparameter_search=False,
                                prior_policy="Fixed standard Gaussian, independent of all data")
                    entries.append(item)
    return entries


def write_manifest(output):
    from .campaign import validate_entries
    entries = diagnostic_entries()
    validate_entries(entries)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in entries))
    plan = dict(stage="bayes_certificate", scientific_setup=SCIENTIFIC_SETUP, tasks=len(entries),
                seeds=list(SEEDS), channel_seed_rule="seed+10000", eval_seed_rule="seed+20000",
                groups=sorted({item["group"] for item in entries}),
                method_counts=dict(sorted(Counter(item["method_label"] for item in entries).items())),
                predictor="One diagonal Gaussian posterior over every MLP weight and bias, including ERM/L2",
                dataset="IID two moons", width=8, n=512, validation_n=512, m=2000,
                epochs=150, lr=.003, posterior_std=.1, prior_std=1., rho_lr_scale=.1,
                l2=1e-4, t1=dict(eta_channel=.01, eta_data=.03), t2=dict(eta_data=.03),
                bounded_loss="-log(.01+.99*p_y)/log(100)",
                evaluation_draws=20, validation_draws=5,
                checkpoint_selection="Validation accuracy with CE tiebreak; no configuration or seed selection",
                hyperparameter_search=False, test_based_selection=False, all_declared_seeds_retained=True,
                main_accuracy_table=False,
                rationale="Revisit the earlier width8/n512/m2000 proof-scale diagnostic using a common Gaussian predictor for every method; earlier selected stochastic-posterior nonvacuity is not evidence for deterministic ERM/L2",
                interpretation="Bound/decomposition diagnostic, separate from main accuracy comparisons. Neither proposed-method superiority nor nonvacuity is promised. Perfect-to-noisy shift and simultaneous finite-MC allocation may make bounds vacuous.",
                certificate_scope="Conditional on the supplied theorems and assumptions; normalized smooth Gibbs loss, not raw CE or classification error",
                execution="Generate only. Training/evaluation belongs on Spartan after a matching CPU resource pilot.",
                figure_formats=["png", "pdf", "eps"], confidence_shading=False,
                manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("journal/configs/bayesian_certificate.jsonl"))
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output), indent=2))


if __name__ == "__main__":
    main()
