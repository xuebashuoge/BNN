"""Build the targeted follow-up to the first recorded journal screen.

All conditions and seeds are explicit. The screen found a near tie between T1
and sample-only PAC on MNIST and lower erasure accuracy for the noisy posterior.
This follow-up fixes learning rate/posterior noise independently of penalty
strength and examines smaller reusable channel banks. It does not select seeds
using their test performance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from journal.build_campaign import base_config, make_entry


def penalty_variants(base, strengths):
    """Give PAC, T1 and T2 comparable large-KL penalty strengths.

    T1 assigns equal leading contributions to channel and sample complexity.
    PAC and T2 coefficients match their sum to first order when KL dominates
    confidence logarithms. This removes a regularization-strength confound;
    the actual finite-KL objectives remain distinct.
    """
    ratio = math.sqrt((base["train_size"] - 1) / (base["m"] - 1))
    joint_ratio = 2 / math.sqrt(1 + (base["m"] - 1) / (base["train_size"] - 1))
    for strength in strengths:
        yield ({**base, "method": "pac_bayes", "eta_channel": 0.,
                "eta_data": 2 * strength * ratio}, "PAC-Matched")
        yield ({**base, "method": "proposed", "eta_channel": strength,
                "eta_data": strength * ratio}, "T1-Matched")
        yield ({**base, "method": "joint", "eta_channel": 0.,
                "eta_data": strength * joint_ratio}, "T2-Matched")


def refine_entries(seeds=(1003, 1004)):
    entries = []
    analog = dict(kind="gaussian", gain_mean=.5, gain_std=1., noise_std=1.)
    outage = dict(kind="outage", outage_prob=.5)
    for m in (20, 50):
        base = base_config("mnist", analog)
        base.update(train_size=10000, val_size=5000, width=128, epochs=35,
                    m=m, lr=.003, posterior_std=.01)
        variants = [({**base, "method": "erm"}, "ERM-Matched"),
                    ({**base, "method": "erm", "training_loss": "raw_ce"}, "ERM-RawCE-Matched"),
                    ({**base, "method": "stochastic_erm"}, "StochERM-Matched"),
                    ({**base, "method": "l2", "l2": 1e-5}, "L2-Matched")]
        variants.extend(penalty_variants(base, (.001, .003, .01)))
        for config, label in variants:
            for seed in seeds:
                entries.append(make_entry(config, seed, f"mnist_analog_m{m}",
                                          label, "refine", "adaptive_refinement"))
    for m in (5, 20, 50):
        base = base_config("moons", outage)
        base.update(train_size=800, val_size=200, width=64, epochs=150,
                    m=m, lr=.003, posterior_std=.003)
        variants = [({**base, "method": "erm"}, "ERM-Matched"),
                    ({**base, "method": "stochastic_erm"}, "StochERM-Matched")]
        variants.extend(penalty_variants(base, (.0003, .003)))
        for config, label in variants:
            for seed in seeds:
                entries.append(make_entry(config, seed, f"moons_outage_m{m}",
                                          label, "refine", "adaptive_refinement"))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("journal/configs/refine.jsonl"))
    parser.add_argument("--seeds", default="1003,1004")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Require distinct explicit seeds")
    entries = refine_entries(seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries))
    plan = dict(stage="refine", tasks=len(entries), seeds=seeds,
                groups=sorted({entry["group"] for entry in entries}),
                based_on="Initial 104-trial screen; MNIST T1/PAC effectively tied, moons erasure below ERM",
                controls="Fixed lr=.003; lower posterior noise for moons; common channel and bank settings within group",
                penalty_matching="PAC/T1/T2 leading sqrt(KL) coefficients matched at each bank size",
                selection="All outcomes retained; aggregate validation selects finalists; test/seed winners exploratory",
                perfect_training="Existing screen/confirmation references retained; no redundant perfect retraining across m")
    args.output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
