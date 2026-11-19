"""Train each two-moons fading predictor with its designated objective and certify its own posterior."""
import argparse
import json
from pathlib import Path

from .diagonal_fading_bounds import MISMATCH_RHO, WIDTHS, design, perfect_shift
from .full_fading_bounds import evaluate_case

# Each width has one deterministic-channel posterior and two learners per stochastic law.
CASES = {
    "perfect": ("perfect", "proposed", "C1"),
    "t1_matched": ("matched", "proposed", "T1"),
    "t2_matched": ("matched", "joint", "T2"),
    "t1_mismatch": ("mismatch", "proposed", "T1"),
    "t2_mismatch": ("mismatch", "joint", "T2"),
}
FAMILY_SIZE = len(WIDTHS) * len(CASES)


def learner_config(width, case, mismatch_rho=MISMATCH_RHO):
    regime, method, certificate = CASES[case]
    cfg = design(width, "fading", regime, 2000, 1, 120, mismatch_rho)
    cfg["method"] = method
    if certificate in {"C1", "T2"}:
        cfg["eta_channel"] = 0.  # Only T1 has a separate trainable channel complexity.
    return cfg


def run(output, width, case, mismatch_rho=MISMATCH_RHO):
    regime, _, certificate = CASES[case]
    cfg = learner_config(width, case, mismatch_rho)
    labels = dict(dataset="moons", family="fading", regime=regime, certificate=certificate, fallback=width == 8)
    return evaluate_case(cfg, output/f"d{width}"/case, 20000, 20, True, FAMILY_SIZE, labels,
                         perfect_shift_fn=perfect_shift, stochastic_theorem="T2" if certificate == "T2" else "T1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--mismatch-rho", type=float, default=MISMATCH_RHO)
    args = parser.parse_args()
    result = run(args.output, args.width, args.case, args.mismatch_rho)
    print(json.dumps({k: result[k] for k in ("regime", "certificate", "uncapped_risk_upper", "nonvacuous")}))
