"""Plot explicitly designated diagonal-fading confirmation summaries, one cohort per dataset."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .full_fading_comparison import DATASET_NAMES, IMAGE_DATASETS, METHODS

STEM = "diagonal_fading_accuracy"
ORDER = (*IMAGE_DATASETS, "moons")


def settings(group):
    dataset = group["dataset"]
    if dataset not in ORDER or group["channel"] != "diagonal_fading":
        raise ValueError(f"Unsupported dataset/channel in {group['group']}")
    m = group["m"]
    if type(m) is not int or m < 2:
        raise ValueError(f"Invalid channel-bank size in {group['group']}")
    # The first campaign wrote rho in the group ID. Later reports also store the field.
    encoded = re.fullmatch(rf"{dataset}_diagonal_fading_r(\d+(?:p\d+)?)_m(\d+)(?:_n(\d+))?",
                           group["group"])
    rho = group.get("rho_c")
    train_size = group.get("train_size")
    if encoded:
        encoded_rho = float(encoded[1].replace("p", "."))
        if (rho is not None and rho != encoded_rho) or int(encoded[2]) != m:
            raise ValueError(f"Inconsistent channel settings in {group['group']}")
        rho = encoded_rho
        if encoded[3]:
            if train_size is not None and train_size != int(encoded[3]):
                raise ValueError(f"Inconsistent training size in {group['group']}")
            train_size = int(encoded[3])
    if type(rho) not in (float, int) or not 0 <= rho < 1:
        raise ValueError(f"Missing or invalid deployment correlation in {group['group']}")
    return dict(dataset=dataset, channel=group["channel"], rho_c=rho, m=m,
                train_size=train_size, width=group.get("width"))


def collect(paths):
    sources, rows, seen = [], [], set()
    for path in paths:
        data = path.read_bytes()
        summary = json.loads(data)
        seeds, groups, stage = summary["confirmation_seeds"], summary["groups"], summary["stage"]
        if (not isinstance(stage, str) or not stage or len(seeds) < 2
                or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
                or not groups or summary["trials"] != len(seeds)*len(groups)*len(METHODS)):
            raise ValueError(f"Incomplete confirmation cohort in {path}")
        source = str(path.resolve())
        sources.append(dict(path=source, sha256=hashlib.sha256(data).hexdigest(),
                            bytes=len(data), summary=summary))
        for group in groups:
            config = settings(group)
            dataset = config["dataset"]
            if dataset in seen:
                raise ValueError(f"Supply one designated cohort per dataset: {dataset}")
            seen.add(dataset)
            if set(group["methods"]) != set(METHODS):
                raise ValueError(f"Require all seven matched/perfect methods in {group['group']}")
            seed_rankings = group.get("descriptive_ranking_by_seed")
            if seed_rankings is not None and set(seed_rankings) != {str(s) for s in seeds}:
                raise ValueError(f"Unpaired seed identities in {group['group']}")
            cohort = []
            for label in METHODS:
                metric = group["methods"][label]["test_accuracy"]
                mean, sd = metric["mean"], metric["sd"]
                if (metric["n"] != len(seeds) or type(mean) not in (float, int)
                        or type(sd) not in (float, int) or not 0 <= mean <= 1
                        or not 0 <= sd <= math.sqrt(len(seeds)/(len(seeds)-1)*mean*(1-mean)) + 1e-12):
                    raise ValueError(f"Invalid fractional accuracy or seed count for {dataset}/{label}")
                cohort.append(dict(**config, stage=stage, confirmation_seeds=" ".join(map(str, seeds)),
                                   n_seeds=len(seeds), method=label, mean_accuracy=mean, sd_accuracy=sd,
                                   mean_accuracy_pct=100*mean, sd_accuracy_pp=100*sd, source=source))
            best = max(row["mean_accuracy"] for row in cohort)
            winners = {row["method"] for row in cohort if best-row["mean_accuracy"] <= 1e-12}
            if winners != set(group["highest_mean_methods"]):
                raise ValueError(f"Winner labels disagree with reported means in {group['group']}")
            rows.extend(cohort)
    rows.sort(key=lambda row: (ORDER.index(row["dataset"]), tuple(METHODS).index(row["method"])))
    return rows, sources


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(1.5 + 2.65*len(datasets), 3.5),
                             squeeze=False, sharey=True, sharex=True)
    styles = {False: dict(fmt="o", color="#1d6480", markerfacecolor="#1d6480"),
              True: dict(fmt="s", color="#8c522d", markerfacecolor="white")}
    for ax, dataset in zip(axes[0], datasets):
        cohort = [row for row in rows if row["dataset"] == dataset]
        sample = cohort[0]
        for y, row in enumerate(cohort):
            ax.errorbar(row["mean_accuracy_pct"], y, xerr=row["sd_accuracy_pp"], markersize=4.8,
                        capsize=2.2, elinewidth=1.2, **styles[METHODS[row["method"]][1]])
        ax.axhline(3.5, color=".8", linewidth=.7)
        seeds = [int(s) for s in sample["confirmation_seeds"].split()]
        seed_label = (f"{seeds[0]}–{seeds[-1]}" if all(b == a+1 for a, b in zip(seeds, seeds[1:]))
                      else ", ".join(map(str, seeds)))
        title = f"{DATASET_NAMES[dataset]}\n$m={sample['m']}$, $\\rho_c={sample['rho_c']:g}$\nSeeds {seed_label}"
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(range(len(METHODS)), ["ERM", "L2", "T1", "T2", "ERM", "L2", "T1/T2"])
    axes[0, 0].set_ylim(len(METHODS)-.5, -.5)
    legend = [Line2D([], [], marker=st["fmt"], color=st["color"], markerfacecolor=st["markerfacecolor"],
                     markersize=5, linestyle="none", label=label)
              for label, st in zip(("Matched training", "Perfect training"), styles.values())]
    fig.legend(handles=legend, ncol=2, loc="lower center", bbox_to_anchor=(.5, .003),
               frameon=False, fontsize=8, handlelength=1.2)
    fig.supxlabel("Gibbs deployment accuracy (%, mean ± 1 SD)", y=.11, fontsize=9.5)
    fig.tight_layout(rect=(0, .15, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{STEM}.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True,
                        help="Confirmation accuracy_summary.json, repeat for each input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = collect(args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(rows, args.output)
    with (args.output / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(metric="Held-out Gibbs deployment accuracy across training seeds",
                      uncertainty="One sample standard deviation across the confirmation seeds",
                      units={"mean_accuracy": "fraction", "sd_accuracy": "fraction",
                             "mean_accuracy_pct": "percent", "sd_accuracy_pp": "percentage points"},
                      datasets=list(dict.fromkeys(row["dataset"] for row in rows)), sources=sources)
    (args.output / f"{STEM}_sources.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    print(f"Wrote {len(rows)} method summaries and figures for {', '.join(provenance['datasets'])} to {args.output}")


if __name__ == "__main__":
    main()
