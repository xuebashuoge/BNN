"""Train each two-moons fading predictor with its designated objective and certify its own posterior."""
import argparse
import fcntl
import json
from pathlib import Path

from .diagonal_fading import DEPLOYMENT_RHO
from .diagonal_fading_bounds import ERASURE_MISMATCH_P, FAMILIES, MISMATCH_RHO, WIDTHS, design, perfect_shift
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
JOINT_FAMILY_SIZE = len(FAMILIES) * FAMILY_SIZE


def learner_config(width, case, mismatch_rho=MISMATCH_RHO, *, family="fading", mismatch_p=ERASURE_MISMATCH_P):
    if family not in FAMILIES:
        raise ValueError(f"Unknown bound-study channel family: {family}")
    if family == "fading" and (isinstance(mismatch_rho, bool) or not isinstance(mismatch_rho, (int, float))
            or not 0 < mismatch_rho < 1 or mismatch_rho == DEPLOYMENT_RHO):
        raise ValueError("The training mismatch must have a nonzero correlation distinct from deployment")
    regime, method, certificate = CASES[case]
    cfg = design(width, family, regime, 2000, 1, 120, mismatch_rho, mismatch_p=mismatch_p)
    cfg["method"] = method
    if certificate in {"C1", "T2"}:
        cfg["eta_channel"] = 0.  # Only T1 has a separate trainable channel complexity.
    return cfg


def run(output, width, case, mismatch_rho=MISMATCH_RHO, *, family="fading", both_channels=False, mismatch_p=ERASURE_MISMATCH_P):
    regime, _, certificate = CASES[case]
    cfg = learner_config(width, case, mismatch_rho, family=family, mismatch_p=mismatch_p)
    labels = dict(dataset="moons", family=family, regime=regime, certificate=certificate, fallback=width == 8)
    family_size = JOINT_FAMILY_SIZE if both_channels else FAMILY_SIZE
    directory = (output / family if both_channels else output) / f"d{width}" / case
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return evaluate_case(cfg, directory, 20000, 20, True, family_size, labels,
                             perfect_shift_fn=perfect_shift, stochastic_theorem="T2" if certificate == "T2" else "T1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="fading")
    parser.add_argument("--both-channels", action="store_true", help="Cover all 20 cases in one confidence family")
    parser.add_argument("--mismatch-p", type=float, default=ERASURE_MISMATCH_P, help="Training erasure probability in the mismatch case")
    parser.add_argument("--mismatch-rho", type=float, default=MISMATCH_RHO)
    args = parser.parse_args()
    result = run(args.output, args.width, args.case, args.mismatch_rho, family=args.family, both_channels=args.both_channels, mismatch_p=args.mismatch_p)
    print(json.dumps({k: result[k] for k in ("regime", "certificate", "uncapped_risk_upper", "nonvacuous")}))
