"""Two-moons risk guarantees for diagonal correlated fading and feature erasure."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .c1_full_fading import perfect_shift as erasure_shift
from .diagonal_fading import diagonal_gain_kl
from .full_fading_bounds import REGIMES, draw_figure, evaluate_case
from .run_experiment import normalize_config, save_json

FAMILIES = ("fading", "erasure")
WIDTHS = (16, 8)
FAMILY_SIZE = len(FAMILIES) * len(REGIMES) * len(WIDTHS)


def perfect_shift(deployment, width, model, loss_epsilon):
    """C1(ii) under the common iid-gain/unit-AWGN reference, or C1(i) for erasures.

    Stack the d diagonal gain differences and d noise differences. The requested
    metric is the Euclidean norm of this vector. For ||f||_2 <= R, the operator
    [diag(f), I] has norm <= sqrt(R^2 + 1). Multiplying the receiver/loss
    Lipschitz constant by this factor gives a uniform product-metric constant K.

    Under Q=P_iid, X=(h-1,N) is a standard-covariance Gaussian translated by
    (-1/2,...,-1/2,0,...,0). For 0<a<1, use
    K||X|| <= K^2/(2a) + a||X||^2/2 and the noncentral chi-square MGF.
    The specified a is analytic, common to every posterior with these caps.
    """
    if deployment["kind"] == "outage":
        return erasure_shift(deployment, width, model, loss_epsilon)
    if deployment["kind"] != "diagonal_fading":
        raise ValueError("Expected diagonal fading or erasure deployment")
    k = model.channel_lipschitz_constant(loss_epsilon) * math.hypot(model.feature_norm, 1.)
    noncentrality_sq = width / 4.
    dimension = 2 * width
    a = min(.5, k / math.sqrt(dimension + noncentrality_sq))
    log_moment_upper = (k*k/(2*a) - dimension/2*math.log1p(-a)
                        + a*noncentrality_sq/(2*(1-a))) if k else 0.
    reference_kl = diagonal_gain_kl(deployment, dict(kind="diagonal_fading", rho_c=0.), width)
    return dict(method="C1 second", metric="sqrt(||H-H'||_F^2 + ||N-N'||_2^2)",
                reference="P_iid", reference_kl=reference_kl, uniform_lipschitz=k,
                feature_norm_limit=model.feature_norm, receiver_norm_limit=model.receiver_norm,
                quadratic_parameter=a, noncentrality_squared=noncentrality_sq,
                reference_dimension=dimension, reference_log_mgf_upper=log_moment_upper,
                analytic=True, upper=reference_kl + log_moment_upper)


def design(width, family, regime, m, threads, epochs):
    deploy = (dict(kind="diagonal_fading", rho_c=.35) if family == "fading"
              else dict(kind="outage", outage_prob=.5))
    if regime == "matched":
        training = deploy.copy()
    elif regime == "perfect":
        training = dict(kind="perfect")
    else:
        training = (dict(kind="diagonal_fading", rho_c=0.) if family == "fading"
                    else dict(kind="outage", outage_prob=.45))
    return normalize_config(dict(
        dataset="moons", architecture="bounded_mlp", width=width, train_size=800, val_size=200,
        moons_sampler="iid", train_channel=training, deployment_channel=deploy, m=m,
        method="proposed", predictor="gaussian", training_loss="bounded_ce_smooth", loss_epsilon=.01,
        bounded_feature_norm=1., bounded_receiver_norm=.075, prior_std=.24, posterior_std=.1,
        rho_lr_scale=0., posterior_mean_clip=.25, eta_data=.02, eta_channel=.02,
        batch_size=128, lr=.006, epochs=epochs, selection_metric="accuracy", val_draws=8,
        seed=92601, channel_seed=92602, eval_seed=92603, data_seed=92604,
        device="cpu", threads=threads))


def run(args, width, family, regime):
    config = design(width, family, regime, args.m, args.threads, args.epochs)
    labels = dict(dataset="moons", family=family, regime=regime, fallback=width == 8)
    case = args.output / f"d{width}" / family / regime
    return evaluate_case(config, case, args.empirical_draws, args.test_draws, True, FAMILY_SIZE,
                         labels, perfect_shift_fn=perfect_shift)


def aggregate(output, width):
    root = output / f"d{width}"
    rows = []
    for family in FAMILIES:
        for regime in REGIMES:
            path = root / family / regime / "bound.json"
            row = json.loads(path.read_text())
            config = row["config"]
            if (row["dataset"] != "moons" or row["family"] != family or row["regime"] != regime
                    or config != design(width, family, regime, config["m"], config["threads"], config["epochs"])
                    or row["confidence"]["family_size"] != FAMILY_SIZE):
                raise ValueError(f"Case does not match the bound design: {path}")
            rows.append(row)
    if len({(r["config"]["m"], r["config"]["epochs"], r["test"]["draws"],
             r["empirical"]["iid_observations"]) for r in rows}) != 1:
        raise ValueError("Cases must use a common training and evaluation budget")
    report = dict(width=width, m=rows[0]["config"]["m"], cases=len(rows),
                  nonvacuous=sum(row["nonvacuous"] for row in rows), rows=rows)
    draw_figure(rows, root / "bound_decomposition", FAMILIES, moons_only=True)
    save_json(root / "bound_decomposition.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cell = sub.add_parser("run")
    cell.add_argument("--family", choices=FAMILIES, required=True)
    cell.add_argument("--regime", choices=REGIMES, required=True)
    summary = sub.add_parser("aggregate")
    fallback = sub.add_parser("fallback", help="Run width eight if any width-sixteen bound is vacuous")
    for command in (cell, summary, fallback):
        command.add_argument("--output", type=Path, required=True)
    for command in (cell, summary):
        command.add_argument("--width", type=int, choices=WIDTHS, default=16)
    for command in (cell, fallback):
        command.add_argument("--m", type=int, default=2000)
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--epochs", type=int, default=120)
        command.add_argument("--empirical-draws", type=int, default=20000)
        command.add_argument("--test-draws", type=int, default=20)
    args = parser.parse_args()
    if args.command != "aggregate" and min(args.empirical_draws, args.test_draws) < 1:
        parser.error("Require positive empirical and test draw counts")
    if args.command == "run":
        row = run(args, args.width, args.family, args.regime)
        print(json.dumps({key: row[key] for key in ("family", "regime", "uncapped_risk_upper", "nonvacuous")}))
        return
    report = aggregate(args.output, args.width if args.command == "aggregate" else 16)
    if args.command == "fallback" and report["nonvacuous"] < report["cases"]:
        first = report["rows"][0]
        if (args.m, args.epochs, args.empirical_draws, args.test_draws) != (
                report["m"], first["config"]["epochs"], first["empirical"]["iid_observations"],
                first["test"]["draws"]):
            parser.error("Pass the same training and evaluation budget as the width-sixteen cases")
        for family in FAMILIES:
            for regime in REGIMES:
                run(args, 8, family, regime)
        report = aggregate(args.output, 8)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
