"""Generate journal tables from frozen campaign JSON; no training or inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
OUT = PAPER / "tables"
RESULTS = REPO / "results" / "journal_campaign"
SOURCES = {}
TABLES = {}
LABELS = {
    "ERM-Perfect": "ERM (perfect)",
    "ERM-Matched": "ERM (matched)",
    "L2-Matched": r"$\ell_2$ (matched)",
    "PAC-Perfect": "Proposed (perfect)",
    "T1-Matched": "Proposed T1 (matched)",
    "T2-Matched": "Proposed T2 (matched)",
}


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[relative(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw)


def describe(values):
    values = list(map(float, values))
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Every table statistic needs finite observations")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None)


def campaign(folder):
    root = RESULTS / folder
    summary = read_json(root / "summary.json")
    selected = {task for item in summary["selected_variants"] if item.get("rank", 1) == 1
                for task in item["task_ids"]}
    rows = [row for row in read_json(root / "test_results.json")
            if row.get("test_status") == "complete" and row["task_id"] in selected]
    if len(rows) != len({row["task_id"] for row in rows}):
        raise ValueError(f"Duplicate selected trials in {folder}")
    return root, rows


def aggregate(folder, group, method):
    root, rows = campaign(folder)
    rows = [row for row in rows if row["group"] == group and row["method_label"] == method]
    if not rows:
        raise ValueError(f"No observations for {folder}/{group}/{method}")
    if len({row["variant_id"] for row in rows}) != 1:
        raise ValueError("Do not pool distinct hyperparameter configurations")
    observations = []
    for row in sorted(rows, key=lambda x: x["seed"]):
        paths = [root / row["task_id"] / name / "evaluation.json"
                 for name in ("test_deployment", "test_full")]
        path = next((path for path in paths if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(row["task_id"])
        report = read_json(path)
        heldout = report["heldout_gibbs_mc"]
        if abs(heldout["accuracy"] - row["test_accuracy"]) > 1e-12:
            raise ValueError("Campaign accuracy differs from frozen inference report")
        if not 0 <= heldout["bounded_loss"] <= 1:
            raise ValueError("Bounded loss outside [0,1]")
        observations.append(dict(task_id=row["task_id"], seed=row["seed"],
                                 source=relative(path), accuracy=heldout["accuracy"],
                                 bounded_loss=heldout["bounded_loss"],
                                 cross_entropy=heldout["cross_entropy"],
                                 draws=heldout["draws"], report=report))
    config = rows[0]["config"]
    return dict(source_campaign=folder, group=group, original_method_label=method,
                display_method=LABELS.get(method, method), config=config,
                accuracy=describe(x["accuracy"] for x in observations),
                bounded_loss=describe(x["bounded_loss"] for x in observations),
                cross_entropy=describe(x["cross_entropy"] for x in observations),
                seeds=[x["seed"] for x in observations],
                draws=sorted({x["draws"] for x in observations}), observations=observations)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items() if key != "report"}
    if isinstance(value, list):
        return [clean(val) for val in value]
    return value


def metric(summary, percent=False, bold=False):
    scale, digits = (100., 2) if percent else (1., 4)
    text = f"{summary['mean'] * scale:.{digits}f}"
    if summary["sd"] is not None:
        text += rf"\pm{summary['sd'] * scale:.{digits}f}"
    return "$" + (rf"\mathbf{{{text}}}" if bold else text) + "$"


def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else val
                             for key, val in row.items()})


def record(name, data, csv_rows, tex):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(tex + "\n")
    write_csv(name, csv_rows)
    TABLES[name] = clean(data)


def flat(row, condition):
    config = row["config"]
    return dict(condition=condition, method=row["original_method_label"],
                display_method=row["display_method"], n_train=config["train_size"],
                n_validation=config["val_size"], m=config["m"], width=config["width"],
                epochs=config["epochs"], batch_size=config["batch_size"],
                seeds=row["seeds"], evaluation_draws=row["draws"],
                accuracy_mean=row["accuracy"]["mean"], accuracy_sample_sd=row["accuracy"]["sd"],
                bounded_loss_mean=row["bounded_loss"]["mean"], bounded_loss_sample_sd=row["bounded_loss"]["sd"],
                raw_ce_mean=row["cross_entropy"]["mean"], raw_ce_sample_sd=row["cross_entropy"]["sd"],
                source_tasks=[x["task_id"] for x in row["observations"]])


def accuracy_and_loss():
    methods = list(LABELS)
    data = {
        "mnist_full": [aggregate("full_refine", "mnist_analog", method) for method in methods],
        "fashion": [aggregate("fashion_controls", "fashion_analog", method) for method in methods],
    }
    for rows in data.values():
        if any(row["draws"] != [10] for row in rows):
            raise ValueError("Main table requires the common ten-draw evaluation")
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             r"\caption{Deployment performance under matched Gaussian fading and additive noise. Accuracy is in percent; bounded risk uses the smooth loss in~\eqref{equ:sim_bounded_loss}. Entries are mean $\pm$ sample standard deviation over three initialization/channel-bank seeds for MNIST and two for FashionMNIST, with ten fresh Gibbs/channel draws per checkpoint. Perfect refers to training only; all models face the same deployment law. Bold identifies the best reported mean within each metric.}",
             r"\label{tab:sim_accuracy_risk}", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{MNIST ($n=55{,}000$, $m=100$)} & \multicolumn{2}{c}{FashionMNIST ($n=10{,}000$, $m=100$)}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Training method & Accuracy (\%) & Bounded risk & Accuracy (\%) & Bounded risk\\", r"\midrule"]
    for index, method in enumerate(methods):
        cells = [LABELS[method]]
        for rows in data.values():
            row = rows[index]
            cells.extend([metric(row["accuracy"], True, row["accuracy"]["mean"] == max(x["accuracy"]["mean"] for x in rows)),
                          metric(row["bounded_loss"], False, row["bounded_loss"]["mean"] == min(x["bounded_loss"]["mean"] for x in rows))])
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    record("accuracy_and_loss", data,
           [flat(row, name) for name, rows in data.items() for row in rows], "\n".join(lines))


def active_hyperparameters(config):
    """Omit inactive defaults, e.g. posterior noise for deterministic ERM."""
    method = config["method"]
    result = dict(lr=config["lr"],
                  posterior_std=None if method in {"erm", "l2"} else config["posterior_std"])
    if method == "l2":
        result["lambda_2"] = config["l2"]
    elif method == "proposed":
        result.update(eta_channel=config["eta_channel"], eta_data=config["eta_data"])
    elif method == "pac_bayes":
        result["eta_data"] = config["eta_data"]
    elif method == "joint":
        result["eta_joint"] = config["eta_data"]
    return result


def latex_number(value):
    if value and abs(value) < 0.001:
        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / 10**exponent
        return (rf"10^{{{exponent}}}" if abs(mantissa - 1) < 1e-12 else
                rf"{mantissa:g}\times10^{{{exponent}}}")
    return f"{value:g}"


def selected_hyperparameters():
    datasets = [("MNIST", "full_refine", "mnist_analog"),
                ("FashionMNIST", "fashion_controls", "fashion_analog")]
    data, csv_rows = dict(selected=[], conditional_search_grids={}), []
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             r"\caption{Validation-selected settings for Table~\ref{tab:sim_accuracy_risk}. The active coefficients are the L2 weight $\lambda_2$, T1 channel/data weights $(\eta_1,\eta_2)$, T2 joint weight $\eta$, or the perfect-training data weight $\eta_2$. The initial posterior standard deviation is $s_0$; dashes denote deterministic models or no penalty. Posterior means use fan-in-scaled uniform initialization with zero biases, separately from $s_0$; the fixed prior has standard deviation one. These are selected configurations from conditional grids, not an exhaustive or full-factorial search.}",
             r"\label{tab:app_hyperparameters}", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Training method & Learning rate & $s_0$ & Active coefficient(s)\\", r"\midrule"]
    for dataset_index, (dataset, folder, group) in enumerate(datasets):
        if dataset_index:
            lines.append(r"\addlinespace")
        for method in LABELS:
            row = aggregate(folder, group, method)
            cfg = row["config"]
            active = active_hyperparameters(cfg)
            coefficient = "--"
            if cfg["method"] == "l2":
                coefficient = rf"$\lambda_2={latex_number(cfg['l2'])}$"
            elif cfg["method"] == "pac_bayes":
                coefficient = rf"$\eta_2={cfg['eta_data']:g}$"
            elif cfg["method"] == "proposed":
                coefficient = rf"$(\eta_1,\eta_2)=({cfg['eta_channel']:g},{cfg['eta_data']:g})$"
            elif cfg["method"] == "joint":
                coefficient = rf"$\eta={cfg['eta_data']:g}$"
            std = "--" if active["posterior_std"] is None else f"{active['posterior_std']:g}"
            lines.append(" & ".join([dataset, LABELS[method], f"{cfg['lr']:g}", std, coefficient]) + r"\\")
            item = dict(dataset=dataset, method=method, display_method=LABELS[method],
                        selected_active_parameters=active, prior_std=cfg["prior_std"],
                        initial_mean="fan-in-scaled uniform; biases zero", config=cfg,
                        seeds=row["seeds"], task_ids=[x["task_id"] for x in row["observations"]])
            data["selected"].append(item)
            csv_rows.append(dict(dataset=dataset, method=method, display_method=LABELS[method],
                                 **active, prior_std=cfg["prior_std"], epochs=cfg["epochs"],
                                 n_train=cfg["train_size"], m=cfg["m"], width=cfg["width"],
                                 seeds=row["seeds"], source_tasks=item["task_ids"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    for name in ("screen", "full_refine", "fashion_controls"):
        path = REPO / "journal" / "configs" / f"{name}.jsonl"
        raw = path.read_bytes()
        SOURCES[relative(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        grouped = {}
        for line in raw.decode().splitlines():
            entry = json.loads(line)
            key = entry["group"] + "/" + entry["method_label"]
            active = active_hyperparameters(entry["config"])
            if active not in grouped.setdefault(key, []):
                grouped[key].append(active)
        data["conditional_search_grids"][relative(path)] = grouped
    record("selected_hyperparameters", data, csv_rows, "\n".join(lines))


def certificate():
    value = read_json(RESULTS / "certificate_m2000" / "certificate.json")
    empirical = value["empirical_gibbs_mc"]["bounded_loss"]
    radius = value["confidence_allocation"]["empirical_mc_upper_radius"]
    rows = [dict(component="Empirical bounded risk estimate", t1=empirical, t2=empirical),
            dict(component="Monte Carlo upper correction", t1=radius, t2=radius),
            dict(component="Channel shift", t1=value["channel_shift"]["channel_shift_complexity"], t2=value["channel_shift"]["channel_shift_complexity"]),
            dict(component="Channel complexity", t1=value["channel_complexity"], t2=None),
            dict(component="Data complexity", t1=value["sample_complexity"], t2=None),
            dict(component="Joint complexity", t1=None, t2=value["joint_complexity"]),
            dict(component="Population-risk upper bound", t1=value["theorem_1"]["population_risk_upper_uncapped"], t2=value["theorem_2"]["population_risk_upper_uncapped"])]
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize",
             r"\caption{Two population-risk bounds evaluated at the same frozen T1-trained posterior ($d=8$, $n=512$, $m=2000$). A fresh 100-draw empirical-risk estimate is shared by both bounds. The total failure budget is $0.05$: $0.04$ is split across both theorems and 21 declared setups, and $0.01$ controls Monte Carlo error. Values concern normalized bounded loss, not classification error.}",
             r"\label{tab:sim_certificate}", r"\begin{tabular}{lcc}", r"\toprule",
             r"Component & Theorem 1 & Theorem 2\\", r"\midrule"]
    for index, row in enumerate(rows):
        values = ["--" if row[key] is None else f"{row[key]:.6f}" for key in ("t1", "t2")]
        if index == len(rows) - 1:
            lines.append(r"\midrule")
        lines.append(row["component"] + " & " + " & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    record("certificate_table", value, rows, "\n".join(lines))


def grouped_performance(name, cases, methods, caption, label, draws):
    data = {condition: [aggregate(folder, group, method) for method in methods]
            for condition, folder, group in cases}
    if any(row["draws"] != [draws] for rows in data.values() for row in rows):
        raise ValueError(f"Unexpected inference draw count in {name}")
    environment = "table*" if name == "appendix_toy_erasure" else "table"
    placement = "t" if environment == "table*" else "!ht"
    lines = [rf"\begin{{{environment}}}[{placement}]", r"\centering\footnotesize", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\begin{tabular}{llcc}", r"\toprule",
             r"Condition & Method & Accuracy (\%) & Bounded risk\\", r"\midrule"]
    for group_index, (condition, rows) in enumerate(data.items()):
        if group_index:
            lines.append(r"\addlinespace")
        best_accuracy = max(row["accuracy"]["mean"] for row in rows)
        best_risk = min(row["bounded_loss"]["mean"] for row in rows)
        for row in rows:
            short_label = row["display_method"].replace(" (matched)", "")
            lines.append(" & ".join([condition, short_label,
                metric(row["accuracy"], True, row["accuracy"]["mean"] == best_accuracy),
                metric(row["bounded_loss"], False, row["bounded_loss"]["mean"] == best_risk)]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}"]
    record(name, data, [flat(row, condition) for condition, rows in data.items() for row in rows], "\n".join(lines))


def data_size():
    data = [aggregate("followup", f"moons_outage_n{n}_m100", "T1-Matched") for n in (128, 512, 2048)]
    rows = []
    for row in data:
        cfg = row["config"]
        signed = lambda key: describe(x["report"]["signed_gap_decomposition_mc"]["bounded_loss"][key] for x in row["observations"])
        rows.append({**flat(row, f"n={cfg['train_size']}"),
                     "optimizer_updates": cfg["epochs"] * math.ceil(cfg["train_size"] / cfg["batch_size"]),
                     "channel_gap_mean": signed("channel_generalization")["mean"],
                     "data_gap_mean": signed("data_generalization")["mean"],
                     "data_complexity_mean": statistics.fmean(x["report"]["final_posterior_complexities"]["sample_complexity"] for x in row["observations"])})
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             r"\caption{Exploratory data-size slice for T1 on IID two moons ($d=8$, $m=100$). Accuracy is mean $\pm$ sample standard deviation over three initialization/channel-bank seeds. The same 150 epochs imply different optimizer-update counts. Negative observed data gaps and the changing optimization budget preclude interpreting this slice as a quantitative validation of the theoretical $n$-rate.}",
             r"\label{tab:app_data_size}", r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$n$ & Updates & Accuracy (\%) & Channel gap & Data gap & Data complexity\\", r"\midrule"]
    for row, raw in zip(rows, data):
        lines.append(f"{row['n_train']} & {row['optimizer_updates']} & {metric(raw['accuracy'], True)} & {row['channel_gap_mean']:.6f} & {row['data_gap_mean']:.6f} & {row['data_complexity_mean']:.6f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    record("appendix_data_size", dict(aggregates=data, rows=rows), rows, "\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_and_loss()
    selected_hyperparameters()
    certificate()
    grouped_performance("appendix_reduced_mnist",
        [(f"$m={m}$", "followup", f"mnist_analog_m{m}") for m in (20, 50)],
        ["ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched"],
        "Reduced-data MNIST under the same analog deployment law ($n=10{,}000$, $d=128$, 35 epochs). Entries use all two paired initialization/channel-bank seeds and ten fresh draws per checkpoint; settings were selected on validation data.",
        "tab:app_reduced_mnist", 10)
    grouped_performance("appendix_screen_mnist", [("$m=100$", "screen", "mnist_analog")],
        ["ERM-Matched", "L2-Matched", "T1-Matched", "T2-Matched"],
        "Initial reduced-data MNIST screen ($n=10{,}000$, $d=128$, $m=100$, 25 epochs). These entries use five inference draws and two seeds, and are kept separate from the ten-draw final comparisons.",
        "tab:app_screen_mnist", 5)
    grouped_performance("appendix_toy_erasure",
        [(f"$m={m}$", "followup", f"moons_outage_m{m}") for m in (5, 20, 50)],
        ["ERM-Matched", "T1-Matched", "T2-Matched"],
        "Width-64 IID two-moons feature-erasure refinement ($n=800$, erasure probability $0.5$, 150 epochs). Entries use two seeds and ten inference draws. A matched L2 run is unavailable for this cohort and is not substituted from another experiment.",
        "tab:app_toy_erasure", 10)
    data_size()
    manifest = dict(description="Frozen-source journal tables; no training or inference", tables=TABLES,
                    source_files=dict(sorted(SOURCES.items())),
                    conventions=dict(accuracy="fraction internally; percent in LaTeX", uncertainty="sample SD across initialization/channel-bank seeds, conditional on fixed dataset",
                        bounded_loss="-log(epsilon+(1-epsilon)*p_y)/log(1/epsilon), epsilon=0.01",
                        proposed_perfect_alias="Source PAC-Perfect is the exact proposed perfect-channel specialization",
                        missing="L2 Perfect absent from the new protocol; no synthetic or legacy substitution"))
    (PAPER / "data_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tables": list(TABLES), "source_files": len(SOURCES), "manifest": str(PAPER / "data_manifest.json")}))


if __name__ == "__main__":
    main()
