"""Evaluate C1's two inequalities on current frozen Gaussian posteriors.

Erasure has an exact first-distance expression. Centered-lognormal output
noise has no simple known closed-form mean norm; the bound instead uses a
fixed Gaussian reference with an analytic moment and exact KL. Independent
distance MC is diagnostic only and never enters a risk upper bound.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .build_bayesian_certificate import diagnostic_entries
from .certify_checkpoint import bank_hash, verify_sampler_source
from .run_experiment import (bounded_losses_from_ce, check_device, load_data,
                            make_models, normalize_config, save_json)
from .stage_bayesian_checkpoints import verify_training_sources
from .theorem1 import gaussian_kl, theorem1_terms


SEVERITIES = (0., .1, .25, .5, 1., 2.)
ERASURE_PROBABILITIES = (0., .01, .05, .1, .25, .5)
RUNS = 12
OFFSET = 13_000_057
TAU = .1
SOURCE_FILES = ("metadata.json", "weights.pt", "history.json", "task_config.json", "manifest_entry.json")
BENCHMARK_REQUEST = dict(shape=TAU, samples=2_000_000, seed=879513, batch_size=100_000)


def fingerprint(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def confidence(delta=.05, empirical_draws=1000, test_draws=100):
    if (not 0 < delta < 1 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                for x in (empirical_draws, test_draws))):
        raise ValueError("Require 0<delta<1 and positive integer draw counts")
    data_failure, mc_failure = .8 * delta / RUNS, .2 * delta / RUNS
    return dict(total_failure=delta, runs=RUNS, data_failure_total=.8 * delta,
        empirical_mc_failure_total=.2 * delta, data_failure_per_run=data_failure,
        empirical_mc_failure_per_run=mc_failure, empirical_draws=empirical_draws,
        test_draws=test_draws, empirical_mc_radius=math.sqrt(math.log(1 / mc_failure) / (2 * empirical_draws)),
        scope="Separate C1 family of twelve fixed posteriors; deterministic analytic shifts apply to all displayed severities. Held-out metrics are descriptive MC estimates.")


def family_entries():
    entries = [entry for entry in diagnostic_entries()
               if entry["channel_name"] == "outage" and entry["training_regime"] == "perfect"]
    if len(entries) != RUNS:
        raise ValueError("Expected twelve current perfect-trained Bayesian checkpoints")
    return entries


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_budget(budget):
    try:
        expected = confidence(budget["total_failure"], budget["empirical_draws"], budget["test_draws"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid C1 confidence allocation") from error
    if budget != expected:
        raise ValueError("C1 confidence allocation differs from its declared twelve-run family")


def checkpoint_context(entry, checkpoint_root):
    """Validate family membership, completeness and the exact frozen source files."""
    prescribed = {item["task_id"]: item for item in family_entries()}
    if entry != prescribed.get(entry.get("task_id")):
        raise ValueError("C1 checkpoint is outside the declared current perfect-trained family")
    checkpoint = Path(checkpoint_root) / entry["task_id"]
    files = {name: checkpoint / name for name in SOURCE_FILES}
    before = {name: fingerprint(path) for name, path in files.items()}
    metadata = json.loads(files["metadata.json"].read_text())
    saved = normalize_config(metadata["config"])
    expected = normalize_config(entry["config"])
    task_config = normalize_config(json.loads(files["task_config.json"].read_text()))
    manifest = json.loads(files["manifest_entry.json"].read_text())
    if (saved != expected or task_config != expected or manifest != entry
            or saved["predictor"] != "gaussian" or saved["train_channel"] != {"kind": "perfect"}):
        raise ValueError("Checkpoint differs from the declared current Gaussian perfect-trained family")
    from .campaign import completed_trial
    if completed_trial(checkpoint, saved) is None or metadata.get("git_dirty") is not False:
        raise ValueError("Require a complete checkpoint from a clean committed training source")
    return checkpoint, files, before, metadata, saved


def validate_benchmark(benchmark):
    if any(benchmark.get(key) != BENCHMARK_REQUEST[key] for key in ("shape", "samples", "seed")):
        raise ValueError("C1 distance diagnostic has a different declared identity")
    for key in ("unit_scale_mean_norm", "unit_scale_mean_norm_standard_error"):
        value = benchmark.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("C1 distance diagnostic must contain finite positive descriptive statistics")
    if benchmark.get("status") != "Independent diagnostic MC only; not a certified first-inequality bound":
        raise ValueError("C1 distance diagnostic must not claim to be a certified first bound")


def prepare_protocol(entries, checkpoint_root, output, budget, threads):
    """Freeze sources, family, budgets and diagnostic identity before parallel work.

    A process lock and atomic JSON writes serialize first-time creation. Every
    later worker must match this immutable protocol; it cannot silently replace
    the diagnostic, budget, source checkpoint or evaluator implementation.
    """
    from .c1_reference import lognormal_distance_benchmark
    validate_budget(budget)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if entries != family_entries():
        raise ValueError("C1 protocol requires the exact declared twelve-posterior family")
    sources = []
    verified = set()
    for entry in entries:
        checkpoint, _, hashes, metadata, saved = checkpoint_context(entry, checkpoint_root)
        if metadata["git_sha"] not in verified:
            verify_training_sources(metadata["git_sha"])
            verify_sampler_source(metadata["git_sha"])
            verified.add(metadata["git_sha"])
        sources.append(dict(task_id=entry["task_id"], checkpoint=str(checkpoint.resolve()),
                            config=saved, source_fingerprints=hashes, source_revision=metadata["git_sha"]))
    science_files = ("evaluate_c1_bayesian.py", "c1_reference.py", "run_experiment.py", "theorem1.py",
                     "build_bayesian_certificate.py", "certify_checkpoint.py", "stage_bayesian_checkpoints.py", "campaign.py")
    expected = dict(schema_version=1, budget=budget, threads=threads, tau=TAU, epsilon=.01,
                    sigmas=list(SEVERITIES), erasure_probabilities=list(ERASURE_PROBABILITIES),
                    benchmark_request=BENCHMARK_REQUEST, sources=sources,
                    evaluator_sources={name: fingerprint(Path(__file__).with_name(name)) for name in science_files},
                    confidence_scope="Standalone C1 family at its declared delta; no joint confidence claim with the main campaign or its separate certificate family.")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        protocol_path, benchmark_path = root / "protocol.json", root / "distance_benchmark.json"
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            benchmark = json.loads(benchmark_path.read_text())
            validate_benchmark(benchmark)
            expected["benchmark_sha256"] = content_sha(benchmark)
            if protocol != expected:
                raise ValueError("Existing C1 protocol has different sources, evaluator, confidence, severities or diagnostic identity")
        else:
            if benchmark_path.exists():
                raise ValueError("C1 diagnostic exists without its immutable protocol; use a fresh output directory")
            benchmark = lognormal_distance_benchmark(**BENCHMARK_REQUEST)
            validate_benchmark(benchmark)
            expected["benchmark_sha256"] = content_sha(benchmark)
            save_json(benchmark_path, benchmark)
            save_json(protocol_path, expected)
            protocol = expected
    return protocol, benchmark


def metrics(logits, labels, config):
    ce = F.cross_entropy(logits, labels, reduction="none")
    return dict(bounded_risk=float(bounded_losses_from_ce(ce, config).double().mean()),
                accuracy=float((logits.argmax(-1) == labels).double().mean()),
                cross_entropy=float(ce.double().mean()))


def average(replicates):
    return {name: statistics.fmean(row[name] for row in replicates)
            for name in ("bounded_risk", "accuracy", "cross_entropy")}


def fresh_streams(config):
    seed = config["eval_seed"] + OFFSET
    previously_used = {config["eval_seed"], config["eval_seed"] + 1_000_003,
                       config["eval_seed"] + 5_000_011, config["eval_seed"] + 9_000_019}
    streams = dict(empirical_weights=seed, test_weights=seed + 100_000, test_channels=seed + 200_000)
    if set(streams.values()) & previously_used or len(set(streams.values())) != 3:
        raise ValueError("C1 streams must be fresh and distinct")
    return streams


def validate_replicates(replicates, count):
    if not isinstance(replicates, list) or len(replicates) != count:
        raise ValueError("C1 cached MC replicate count differs from the declared draws")
    for row in replicates:
        if set(row) != {"bounded_risk", "accuracy", "cross_entropy"}:
            raise ValueError("C1 cached MC replicate has unexpected metrics")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in row.values()):
            raise ValueError("C1 cached MC replicate contains nonfinite metrics")
        if not 0 <= row["bounded_risk"] <= 1 or not 0 <= row["accuracy"] <= 1 or row["cross_entropy"] < 0:
            raise ValueError("C1 cached MC metrics violate their loss ranges")


def derive_channels(config, baseline, benchmark, erasure_draws, noise_draws):
    from .c1_reference import erasure_first_shift, lognormal_logit_channel_terms
    if len(erasure_draws) != len(ERASURE_PROBABILITIES) or len(noise_draws) != len(SEVERITIES):
        raise ValueError("C1 channel rows differ from the declared severity grid")
    erasure, lognormal = [], []
    for probability, replicates in zip(ERASURE_PROBABILITIES, erasure_draws):
        values = average(replicates)
        shift = erasure_first_shift(config["width"], probability)
        erasure.append(dict(severity=probability, test_accuracy=values["accuracy"],
            test_bounded_risk=values["bounded_risk"], test_cross_entropy=values["cross_entropy"],
            shift_first_exact=shift, risk_upper_uncapped=baseline + shift,
            population_risk_upper=min(1., baseline + shift), nonvacuous=baseline + shift < 1., replicates=replicates))
    for sigma, replicates in zip(SEVERITIES, noise_draws):
        values = average(replicates)
        terms = lognormal_logit_channel_terms(sigma, shape=TAU, epsilon=config["loss_epsilon"])
        shift = terms["c1_second_shift_upper"]
        lognormal.append(dict(severity=sigma, test_accuracy=values["accuracy"],
            test_bounded_risk=values["bounded_risk"], test_cross_entropy=values["cross_entropy"],
            shift_first_mc_diagnostic=terms["lipschitz"] * sigma * benchmark["unit_scale_mean_norm"],
            shift_first_mc_se=terms["lipschitz"] * sigma * benchmark["unit_scale_mean_norm_standard_error"],
            shift_second_closed=shift, log_reference_moment=terms["log_exponential_moment"],
            reference_kl=terms["channel_kl"], jensen_shift_upper=terms["jensen_shift_upper"],
            risk_upper_uncapped=baseline + shift, population_risk_upper=min(1., baseline + shift),
            nonvacuous=baseline + shift < 1., first_risk_upper=None,
            channel_guarantee_uses_distance_mc=False, replicates=replicates))
    return erasure, lognormal


def validate_cached_result(result, entry, request, metadata, saved, state, benchmark,
                           code_verification, sampler_verification):
    """Check checksum, frozen identity, MC counts and every derived bound value."""
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if result.get("result_sha256") != content_sha(payload):
        raise ValueError("C1 cached result checksum does not match its contents")
    if result.get("request") != request:
        raise ValueError("Existing C1 evaluation has different sources, confidence, streams or diagnostic")
    fixed = dict(task_id=entry["task_id"], method_label=entry["method_label"], seed=saved["seed"],
                 checkpoint=request["checkpoint"], source_config=saved,
                 source_hashes=request["source_fingerprints"], source_revision=metadata["git_sha"], no_training=True,
                 split_hashes=metadata["split"], saved_bank_hash=metadata["training_channel_bank_hash"],
                 sampler_source_verification=sampler_verification)
    if any(result.get(key) != value for key, value in fixed.items()):
        raise ValueError("C1 cached result disagrees with its frozen source identity")
    old_code = result.get("training_source_verification", {})
    if any(old_code.get(key) != code_verification.get(key) for key in ("source_revision", "files")):
        raise ValueError("C1 cached result has different verified training sources")
    budget = request["budget"]
    validate_replicates(result["empirical_replicates"], budget["empirical_draws"])
    empirical = average(result["empirical_replicates"])["bounded_risk"]
    kl = gaussian_kl(state["model"]["mu"].double(), state["model"]["rho"].double(), saved["prior_std"]).item()
    data_term = theorem1_terms(torch.tensor(kl, dtype=torch.float64), n=512, m=saved["m"],
                               delta=budget["data_failure_per_run"], perfect_training=True)["sample_complexity"].item()
    empirical_upper = min(1., empirical + budget["empirical_mc_radius"])
    baseline = empirical_upper + data_term
    derived = dict(empirical_bounded_risk_mc=empirical, empirical_mc_radius=budget["empirical_mc_radius"],
                   empirical_bounded_risk_upper=empirical_upper, data_complexity=data_term,
                   posterior_kl_float64=kl, baseline_risk_upper_uncapped=baseline)
    if any(result.get(key) != value for key, value in derived.items()):
        raise ValueError("C1 cached result has inconsistent empirical risk, KL or confidence terms")
    erasure_draws = [row["replicates"] for row in result["erasure"]]
    noise_draws = [row["replicates"] for row in result["lognormal"]]
    for replicates in erasure_draws + noise_draws:
        validate_replicates(replicates, budget["test_draws"])
    erasure, lognormal = derive_channels(saved, baseline, benchmark, erasure_draws, noise_draws)
    if result["erasure"] != erasure or result["lognormal"] != lognormal:
        raise ValueError("C1 cached result has inconsistent channel terms or risk bounds")
    # Both zero-severity channels are exactly the same noiseless predictor on
    # the same full-loader weight draws, so these MC lists must coincide.
    if erasure_draws[0] != noise_draws[0]:
        raise ValueError("C1 zero-severity channels disagree on the noiseless predictor")


@torch.no_grad()
def evaluate_one(entry, checkpoint_root, output, budget, benchmark, threads=2, protocol=None):
    validate_budget(budget)
    validate_benchmark(benchmark)
    checkpoint, files, before, metadata, saved = checkpoint_context(entry, checkpoint_root)
    if protocol is None:
        protocol, declared_benchmark = prepare_protocol(family_entries(), checkpoint_root, output, budget, threads)
        if declared_benchmark != benchmark:
            raise ValueError("C1 evaluation diagnostic differs from its immutable protocol")
    if protocol["benchmark_sha256"] != content_sha(benchmark):
        raise ValueError("C1 diagnostic changed after protocol declaration")
    source = next(item for item in protocol["sources"] if item["task_id"] == entry["task_id"])
    if source["source_fingerprints"] != before or protocol["budget"] != budget or protocol["threads"] != threads:
        raise ValueError("C1 source or budget changed after immutable protocol declaration")
    code_verification = verify_training_sources(metadata["git_sha"])
    sampler_verification = verify_sampler_source(metadata["git_sha"])
    config = normalize_config({**saved, "device": "cpu", "threads": threads, "download": False})
    check_device(config)
    torch.set_num_threads(threads)
    streams = fresh_streams(config)
    request = dict(checkpoint=str(checkpoint.resolve()), source_fingerprints=before,
                   budget=budget, streams=streams, threads=threads, tau=TAU,
                   protocol_sha256=content_sha(protocol), benchmark_sha256=content_sha(benchmark),
                   sigmas=list(SEVERITIES), erasure_probabilities=list(ERASURE_PROBABILITIES))
    row_path = Path(output) / "rows" / (entry["task_id"] + ".json")
    state = torch.load(files["weights.pt"], map_location="cpu", weights_only=True)
    from .run_experiment import digest
    state_bank_hash = digest({key: state["bank"][key].tolist() for key in ("gain", "noise")})
    if (state_bank_hash != metadata["training_channel_bank_hash"] or
            not bool((state["bank"]["gain"] == 1).all()) or not bool((state["bank"]["noise"] == 0).all())):
        raise ValueError("C1 source bank differs from the frozen perfect-training identity bank")
    if row_path.exists():
        cached = json.loads(row_path.read_text())
        validate_cached_result(cached, entry, request, metadata, saved, state, benchmark,
                               code_verification, sampler_verification)
        if before != {name: fingerprint(path) for name, path in files.items()}:
            raise ValueError("Frozen checkpoint changed during C1 cache verification")
        return cached
    train, _, test, input_dim, classes, split = load_data(config, include_test=True)
    if input_dim != 2 or classes != 2 or config["width"] != 8 or config["train_size"] != 512:
        raise ValueError("This C1 diagnostic uses the declared 2->8->2 IID model")
    if any(split[key] != metadata["split"][key] for key in ("train_indices_hash", "validation_indices_hash")):
        raise ValueError("Recreated C1 data splits differ from the checkpoint")
    model, bank = make_models(config, input_dim, classes)
    model.load_state_dict(state["model"])
    bank.load_state_dict(state["bank"])
    if not model.stochastic or bank_hash(bank) != metadata["training_channel_bank_hash"]:
        raise ValueError("Require the actual saved positive-Gaussian posterior and perfect bank")
    if not bool((bank.gain == 1).all()) or not bool((bank.noise == 0).all()):
        raise ValueError("C1 perfect training requires identity transmission")
    model.eval()
    x_train, y_train = train[:]
    x_test, y_test = test[:]
    empirical_replicates = []
    weight_generator = torch.Generator().manual_seed(streams["empirical_weights"])
    for _ in range(budget["empirical_draws"]):
        theta = model.sample(weight_generator)
        w1, b1, w2, b2 = model.split(theta)
        logits = F.relu(x_train @ w1 + b1) @ w2 + b2
        empirical_replicates.append(metrics(logits, y_train, config))
    empirical = average(empirical_replicates)
    kl = gaussian_kl(model.mu.detach().double(), model.rho.detach().double(), config["prior_std"])
    if not torch.isfinite(kl) or kl < 0:
        raise ValueError("Require finite nonnegative frozen Gaussian KL")
    data_term = theorem1_terms(kl, n=512, m=config["m"], delta=budget["data_failure_per_run"],
                               perfect_training=True)["sample_complexity"].item()
    empirical_upper = min(1., empirical["bounded_risk"] + budget["empirical_mc_radius"])
    baseline = empirical_upper + data_term
    erasure_draws = [[] for _ in ERASURE_PROBABILITIES]
    noise_draws = [[] for _ in SEVERITIES]
    weights = torch.Generator().manual_seed(streams["test_weights"])
    channels = torch.Generator().manual_seed(streams["test_channels"])
    tau = TAU
    for _ in range(budget["test_draws"]):
        theta = model.sample(weights)
        w1, b1, w2, b2 = model.split(theta)
        hidden = F.relu(x_test @ w1 + b1)
        logits = hidden @ w2 + b2
        uniforms = torch.rand(hidden.shape, generator=channels)
        z = torch.randn(logits.shape, generator=channels)
        standardized_noise = torch.expm1(tau * z - tau * tau / 2) / math.sqrt(math.expm1(tau * tau))
        for probability, replicates in zip(ERASURE_PROBABILITIES, erasure_draws):
            distorted = (hidden * (uniforms >= probability)) @ w2 + b2
            replicates.append(metrics(distorted, y_test, config))
        for sigma, replicates in zip(SEVERITIES, noise_draws):
            replicates.append(metrics(logits + sigma * standardized_noise, y_test, config))
    erasure, lognormal = derive_channels(config, baseline, benchmark, erasure_draws, noise_draws)
    result = dict(task_id=entry["task_id"], method_label=entry["method_label"], seed=config["seed"],
        checkpoint=str(checkpoint.resolve()), source_config=saved, source_hashes=before,
        source_revision=metadata["git_sha"], request=request, no_training=True,
        empirical_bounded_risk_mc=empirical["bounded_risk"], empirical_mc_radius=budget["empirical_mc_radius"],
        empirical_replicates=empirical_replicates, empirical_bounded_risk_upper=empirical_upper,
        data_complexity=data_term, posterior_kl_float64=kl.item(), baseline_risk_upper_uncapped=baseline,
        training_source_verification=code_verification, sampler_source_verification=sampler_verification,
        split_hashes=split, saved_bank_hash=bank_hash(bank), erasure=erasure, lognormal=lognormal)
    result["result_sha256"] = content_sha(result)
    validate_cached_result(result, entry, request, metadata, saved, state, benchmark,
                           code_verification, sampler_verification)
    if before != {name: fingerprint(path) for name, path in files.items()}:
        raise ValueError("Frozen checkpoint changed during C1 inference")
    save_json(row_path, result)
    return result


def aggregate(entries, output, budget, benchmark, checkpoint_root=None, threads=2, protocol=None):
    from .c1_reference import smooth_loss_lipschitz
    if entries != family_entries() or any(not (Path(output) / "rows" / (entry["task_id"] + ".json")).is_file() for entry in entries):
        raise ValueError("C1 reporting requires every declared posterior")
    if checkpoint_root is None:
        checkpoint_root = Path(json.loads((Path(output) / "protocol.json").read_text())["sources"][0]["checkpoint"]).parent
    reports = [evaluate_one(entry, checkpoint_root, output, budget, benchmark, threads, protocol) for entry in entries]
    if len(reports) != RUNS or {row["task_id"] for row in reports} != {entry["task_id"] for entry in entries}:
        raise ValueError("C1 reporting requires every declared posterior")
    source_hashes = {row["checkpoint"] + "/" + name: values for row in reports for name, values in row["source_hashes"].items()}
    data = dict(schema_version=1, confidence=budget, protocol=protocol,
        confidence_scope="Standalone C1 family; no joint confidence claim with the main all-Bayesian campaign or its separate certificates.",
        analytic=dict(epsilon=.01, tau=TAU,
        K=smooth_loss_lipschitz(.01), benchmark=benchmark, reference="N(0,sigma^2 I_2), fixed moment matched",
        direct_expectation="No simple known closed form; independent MC diagnostic only",
        lognormal_split="Full trained NN is transmitter, fixed softmax is receiver; additive noise acts on two output logits",
        erasure_split="Original hidden-feature split, discrete metric on complete masks, K=1",
        closed_form_scope="Reference channel-shift term is fully analytic. Empirical risk is estimated with explicit fresh bounded-loss MC correction.",
        uniqueness="Jensen is another valid analytic upper bound; C1 is a tractability device, not the only possible analytic guarantee."),
        reports=reports, source_hashes=source_hashes)
    save_json(Path(output) / "c1_evaluation.json", data)
    fields = ["method_label", "seed", "channel_family", "severity", "test_accuracy", "test_bounded_risk",
              "empirical_bounded_risk_mc", "empirical_mc_radius", "data_complexity", "shift_first_exact",
              "shift_first_mc_diagnostic", "shift_second_closed", "log_reference_moment", "reference_kl",
              "jensen_shift_upper", "risk_upper_uncapped", "population_risk_upper", "nonvacuous"]
    with (Path(output) / "c1_evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            for family in ("erasure", "lognormal"):
                for row in report[family]:
                    writer.writerow({**report, **row, "channel_family": family})
    return data


def run(checkpoint_root, output, empirical_draws=1000, test_draws=100, delta=.05, threads=2,
        index=None, aggregate_only=False, prepare_only=False):
    entries = family_entries()
    budget = confidence(delta, empirical_draws, test_draws)
    root = Path(output)
    if sum((index is not None, aggregate_only, prepare_only)) > 1:
        raise ValueError("Choose only one of index, aggregate-only or prepare-only")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < RUNS):
        raise ValueError("Index must be in the twelve-posterior C1 family")
    protocol, benchmark = prepare_protocol(entries, checkpoint_root, root, budget, threads)
    if prepare_only:
        return dict(protocol=protocol, benchmark=benchmark, prepared=True)
    if aggregate_only:
        return aggregate(entries, root, budget, benchmark, checkpoint_root, threads, protocol)
    if index is not None:
        return evaluate_one(entries[index], checkpoint_root, root, budget, benchmark, threads, protocol)
    for entry in entries:
        evaluate_one(entry, checkpoint_root, root, budget, benchmark, threads, protocol)
    return aggregate(entries, root, budget, benchmark, checkpoint_root, threads, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/journal_bayesian_certificate/20261117/campaign"))
    parser.add_argument("--output", type=Path, default=Path("results/journal_c1_bayesian/20261117"))
    parser.add_argument("--empirical-draws", type=int, default=1000)
    parser.add_argument("--test-draws", type=int, default=100)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--index", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data = run(args.checkpoint_root, args.output, args.empirical_draws, args.test_draws, args.delta,
               args.threads, args.index, args.aggregate_only, args.prepare_only)
    print(json.dumps(dict(output=str(args.output.resolve()), reports=len(data.get("reports", [])), task_id=data.get("task_id"))))


if __name__ == "__main__":
    main()
