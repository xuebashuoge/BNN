"""Write the manuscript deployment tables from the published per-seed CSVs.

* performance_table.tex: m = 2000, T1 with the fresh data-dependent prior Q_{W|S}
  (data_dependent/only_fresh/performance).
* bank_size_table.tex: best baseline max(ERM, L2) against best proposed max(T1, T2) at
  m = 20, 200, 1000 (higher_accuracy/) and 2000 (only_fresh).

Pass --check to compare against the saved files instead of writing them.
"""
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "data_dependent" / "only_fresh" / "performance" / "deployment_by_seed.csv"
SWEEP = {20: ROOT / "higher_accuracy" / "performance" / "deployment_by_seed.csv",
         200: ROOT / "higher_accuracy" / "m200" / "performance" / "deployment_by_seed.csv",
         1000: ROOT / "higher_accuracy" / "m1000" / "performance" / "deployment_by_seed.csv",
         2000: MAIN}
METHODS = ("ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched", "Proposed-Perfect")
DATASETS = {"moons": "Two moons", "mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
CHANNELS = ("Fading", "Erasure")


def accuracies(path, m):
    """{(dataset, channel, method): [accuracy % per paired seed, in seed order]}."""
    table = {}
    for row in csv.DictReader(path.open()):
        # The perfect-training column does not depend on m; every campaign copies the m = 2000 fits.
        if int(row["m"]) != (2000 if row["method"] == "Proposed-Perfect" else m):
            raise ValueError(f"{path} has m = {row['m']}, expected {m}")
        table.setdefault((row["dataset"], row["channel"], row["method"]), []).append(
            (int(row["seed"]), float(row["test_accuracy_pct"])))
    if len(table) != len(DATASETS) * len(CHANNELS) * len(METHODS):
        raise ValueError(f"{path} does not cover every row and method")
    for key, values in table.items():
        table[key] = [acc for _, acc in sorted(values)]
    seeds = {(d, c): len(table[d, c, METHODS[0]]) for d in DATASETS for c in CHANNELS}
    if any(len(table[d, c, method]) != seeds[d, c] for d, c, method in table):
        raise ValueError(f"{path} has unpaired seeds")
    return table


def header(caption, label, spec):
    return [r"\begin{table*}[htb]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + spec + "@{}}", r"\toprule"]


FOOTER = [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]


def main_table(acc):
    lines = header(r"Deployment accuracy (\%) comparison at channel-set size $m=2000$.", "tab:performance",
                   "llccccc")
    lines += [r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Channel}",
              r"& \multicolumn{4}{c}{Matched training} & \multirow{2}{*}{Perfect training} \\",
              r"\cmidrule(lr){3-6}", r"& & ERM & L2 & T1 & T2 &  \\"]
    for dataset, label in DATASETS.items():
        lines += [r"\midrule", r"\multirow{2}{*}{" + label + "}"]
        for channel in CHANNELS:
            values = [acc[dataset, channel, method] for method in METHODS]
            means = [statistics.fmean(v) for v in values]
            cells = []
            for by_seed, mean in zip(values, means):
                printed = f"{mean:.2f}"
                if max(means) - mean <= 1e-10:
                    printed = r"\mathbf{" + printed + "}"
                cells.append("$" + printed + r"\pm" + f"{statistics.stdev(by_seed):.2f}$")
            lines += ["& " + channel, "& " + " & ".join(cells) + r" \\"]
    return "\n".join(lines + FOOTER)


def best(acc, dataset, channel, methods):
    """(method, mean, per-seed values) of the method with the highest mean."""
    return max(((m, statistics.fmean(acc[dataset, channel, m]), acc[dataset, channel, m]) for m in methods),
               key=lambda item: item[1])


def sweep_table(sweep):
    sizes = list(sweep)
    lines = header(r"Deployment accuracy (\%) of the best baseline, $\max(\text{ERM}, \text{L2})$, and the best "
                   r"proposed method, $\max(\text{T1}, \text{T2})$, for channel-set sizes $m$. $^\dagger$ marks "
                   r"rows where T2 is the better proposed method; elsewhere it is T1. The $m=2000$ "
                   r"columns repeat Table~\ref{tab:performance}; the other sizes were tuned at their own $m$.",
                   "tab:bank_size", "ll" + "cc" * len(sizes))
    lines += [r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Channel}",
              "".join(r" & \multicolumn{2}{c}{$m=" + str(m) + "$}" for m in sizes) + r" \\",
              "".join(r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}" for i in range(len(sizes))),
              "& " + " & Baseline & Proposed" * len(sizes) + r" \\"]
    leads = {m: [] for m in sizes}
    for dataset, label in DATASETS.items():
        lines += [r"\midrule", r"\multirow{2}{*}{" + label + "}"]
        for channel in CHANNELS:
            cells = []
            for m in sizes:
                _, base, _ = best(sweep[m], dataset, channel, METHODS[:2])
                method, ours, _ = best(sweep[m], dataset, channel, METHODS[2:4])
                leads[m].append(ours - base)
                base_cell, ours_cell = f"{base:.2f}", f"{ours:.2f}"
                if ours > base:
                    ours_cell = r"\mathbf{" + ours_cell + "}"
                else:
                    base_cell = r"\mathbf{" + base_cell + "}"
                dagger = r"^\dagger" if method == "T2-Matched" else ""
                cells += ["$" + base_cell + "$", "$" + ours_cell + dagger + "$"]
            lines += ["& " + channel, "& " + " & ".join(cells) + r" \\"]
    lines += [r"\midrule",
              r"\multicolumn{2}{@{}l}{Mean gain (points)}" + "".join(
                  r" & \multicolumn{2}{c}{$" + f"{statistics.fmean(leads[m]):+.2f}" + "$}" for m in sizes) + r" \\",
              r"\multicolumn{2}{@{}l}{Rows won}" + "".join(
                  r" & \multicolumn{2}{c}{" + f"{sum(g > 0 for g in leads[m])}/{len(leads[m])}" + "}"
                  for m in sizes) + r" \\"]
    return "\n".join(lines + FOOTER)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Compare with the saved tables instead of writing")
    args = parser.parse_args()
    sweep = {m: accuracies(path, m) for m, path in SWEEP.items()}
    for path, content in [(ROOT / "performance_table.tex", main_table(sweep[2000])),
                          (ROOT / "bank_size_table.tex", sweep_table(sweep))]:
        if args.check:
            if path.read_text() != content:
                raise ValueError(f"Regenerated table differs from {path.name}")
        else:
            path.write_text(content)
    print("Wrote the m = 2000 table and the bank-size table" if not args.check else "Both tables match")


if __name__ == "__main__":
    main()
