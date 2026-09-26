"""Compare RTX 4080 replay runs without hiding per-request noise or drift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import fmean, median
import json
from pathlib import Path
import random
from typing import Any, Iterable

from racebench.metrics import percentile
from racebench.score import effective_request_score


@dataclass(frozen=True)
class RunData:
    path: Path
    summary: dict[str, Any]
    requests: dict[str, dict[str, Any]]


def load_run(path: Path) -> RunData:
    summary: dict[str, Any] | None = None
    requests: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "summary":
                summary = record
            elif record.get("record_type") == "request":
                request_id = record.get("request_id")
                if not isinstance(request_id, str):
                    raise ValueError(f"{path}:{line_number}: request_id is required")
                if request_id in requests:
                    raise ValueError(f"{path}:{line_number}: duplicate request_id {request_id}")
                requests[request_id] = record
    if summary is None:
        raise ValueError(f"{path}: summary record is required")
    return RunData(path, summary, requests)


def metric_values(run: RunData, metric: str) -> list[float]:
    return [
        float(record[metric])
        for record in run.requests.values()
        if record.get("error") is None and record.get(metric) is not None
    ]


def aggregate_metric(values: Iterable[float]) -> dict[str, float | int]:
    collected = list(values)
    if not collected:
        raise ValueError("metric has no valid observations")
    return {
        "n": len(collected),
        "mean": fmean(collected),
        "median": median(collected),
        "p95": percentile(collected, 0.95),
        "p99": percentile(collected, 0.99),
    }


def failed_requests(run: RunData) -> int:
    return sum(1 for record in run.requests.values() if record.get("error") is not None)


# Summary fields that define the workload. Runs that differ in any of them share
# request ids but not prompts, arrival times or output limits, so pairing by id
# would compare different requests.
# A percentile bootstrap over a handful of clusters is badly under-covered: two
# identical 4-conversation runs on a busy GPU produced a "significant" loss.
MIN_CLUSTERS = 20

WORKLOAD_KEYS = (
    "model",
    "seed",
    "request_rate",
    "conversations",
    "turns",
    "max_tokens",
    "history",
    "output_length",
)


def check_comparable(baseline: RunData, candidate: RunData) -> None:
    mismatched = [
        f"{key}: {baseline.summary[key]!r} vs {candidate.summary[key]!r}"
        for key in WORKLOAD_KEYS
        if key in baseline.summary
        and key in candidate.summary
        and baseline.summary[key] != candidate.summary[key]
    ]
    if baseline.requests.keys() != candidate.requests.keys():
        mismatched.append(
            f"request ids differ ({len(baseline.requests.keys() ^ candidate.requests.keys())} unpaired)"
        )
    if mismatched:
        raise ValueError(
            f"{baseline.path} and {candidate.path} are not the same workload: " + "; ".join(mismatched)
        )


def _valid(record: dict[str, Any], *metrics: str) -> bool:
    return record.get("error") is None and all(record.get(metric) is not None for metric in metrics)


def same_prompt(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """False only when both runs recorded prompt hashes and they differ.

    With live history a later turn embeds that run's own earlier answers, so
    the "same" request id can carry a different prompt in each run.
    """

    left, right = before.get("prompt_sha256"), after.get("prompt_sha256")
    return left is None or right is None or left == right


def paired_rows(baseline: RunData, candidate: RunData, metric: str) -> list[tuple[Any, float]]:
    """(conversation, candidate - baseline) for identical requests valid in both runs."""

    rows: list[tuple[Any, float]] = []
    for request_id in sorted(baseline.requests.keys() & candidate.requests.keys()):
        before = baseline.requests[request_id]
        after = candidate.requests[request_id]
        if not (_valid(before, metric) and _valid(after, metric) and same_prompt(before, after)):
            continue
        cluster = before.get("conversation_id", request_id)
        rows.append((cluster, float(after[metric]) - float(before[metric])))
    if not rows:
        raise ValueError(f"no paired observations for {metric}")
    return rows


def paired_deltas(baseline: RunData, candidate: RunData, metric: str) -> list[float]:
    return [delta for _, delta in paired_rows(baseline, candidate, metric)]


def pairing_report(baseline: RunData, candidate: RunData) -> dict[str, Any]:
    """How many pairs are provably the same prompt, and how often outputs agree."""

    shared = sorted(baseline.requests.keys() & candidate.requests.keys())
    hashed = [
        request_id
        for request_id in shared
        if baseline.requests[request_id].get("prompt_sha256")
        and candidate.requests[request_id].get("prompt_sha256")
    ]
    mismatches = sum(
        1
        for request_id in hashed
        if not same_prompt(baseline.requests[request_id], candidate.requests[request_id])
    )
    outputs = [
        request_id
        for request_id in shared
        if _valid(baseline.requests[request_id], "output_text")
        and _valid(candidate.requests[request_id], "output_text")
        and same_prompt(baseline.requests[request_id], candidate.requests[request_id])
    ]
    matches = sum(
        1
        for request_id in outputs
        if baseline.requests[request_id]["output_text"] == candidate.requests[request_id]["output_text"]
    )
    return {
        "prompt_identity": "verified" if hashed else "not recorded",
        "prompt_mismatches": mismatches,
        "output_pairs": len(outputs),
        "output_match_rate": matches / len(outputs) if outputs else None,
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    clusters: list[Any] | None = None,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 2025,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean.

    With ``clusters`` whole clusters are resampled. Turns of one conversation
    share history and server state, so resampling them as independent requests
    would understate the uncertainty.
    """

    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if clusters is not None and len(clusters) != len(values):
        raise ValueError("clusters must label every value")
    groups: dict[Any, list[float]] = {}
    for index, value in enumerate(values):
        groups.setdefault(index if clusters is None else clusters[index], []).append(value)
    totals = [(sum(group), len(group)) for group in groups.values()]
    rng = random.Random(seed)
    count = len(totals)
    means: list[float] = []
    for _ in range(samples):
        picked = [totals[rng.randrange(count)] for _ in range(count)]
        means.append(sum(total for total, _ in picked) / sum(n for _, n in picked))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return percentile(means, tail), percentile(means, 1.0 - tail)


def comparison(baseline: RunData, candidate: RunData, seed: int, samples: int) -> dict[str, Any]:
    check_comparable(baseline, candidate)
    report: dict[str, Any] = {
        "baseline": str(baseline.path),
        "candidate": str(candidate.path),
        "interpretation": "delta = candidate - baseline; negative latency delta is faster",
        "bootstrap": "percentile, resampling whole conversations",
        "failures": {
            "baseline": failed_requests(baseline),
            "candidate": failed_requests(candidate),
        },
        "pairing": pairing_report(baseline, candidate),
        "metrics": {},
    }
    means: dict[str, dict[str, float]] = {}
    for index, metric in enumerate(("ttft_ms", "tpot_ms")):
        rows = paired_rows(baseline, candidate, metric)
        deltas = [delta for _, delta in rows]
        ci_low, ci_high = bootstrap_mean_ci(
            deltas,
            clusters=[cluster for cluster, _ in rows],
            samples=samples,
            seed=seed + index,
        )
        before = aggregate_metric(metric_values(baseline, metric))
        after = aggregate_metric(metric_values(candidate, metric))
        means[metric] = {"baseline": float(before["mean"]), "candidate": float(after["mean"])}
        report["metrics"][metric] = {
            "baseline": before,
            "candidate": after,
            "paired_n": len(deltas),
            "paired_clusters": len({cluster for cluster, _ in rows}),
            "paired_delta_mean": fmean(deltas),
            "paired_delta_median": median(deltas),
            "paired_delta_p95": percentile(deltas, 0.95),
            "bootstrap_mean_delta_95ci": [ci_low, ci_high],
            "direction": "faster" if ci_high < 0 else "slower" if ci_low > 0 else "uncertain",
        }
    # Score both runs from the same request-level means rather than trusting
    # each summary, whose aggregation (--aggregate) may differ between runs.
    report["quoted_ers"] = {
        "aggregation": "mean over successful requests",
        "baseline": effective_request_score(means["ttft_ms"]["baseline"], means["tpot_ms"]["baseline"]),
        "candidate": effective_request_score(means["ttft_ms"]["candidate"], means["tpot_ms"]["candidate"]),
    }
    report["quoted_ers"]["delta"] = report["quoted_ers"]["candidate"] - report["quoted_ers"]["baseline"]
    return report


def score_bootstrap(
    baseline: RunData,
    candidate: RunData,
    baseline_return: RunData | None = None,
    *,
    samples: int = 2_000,
    seed: int = 2025,
) -> dict[str, Any]:
    """Conversation bootstrap of the candidate's quoted-formula ERS change.

    ERS, not either latency alone, is the objective, so this interval is the
    gate. The reference is R0, or the R0/R0-prime midpoint when R0-prime
    exists, which cancels drift that is linear in time. Only requests that are
    valid in every run, and identical where hashes exist, are used.
    """

    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    runs = [baseline, candidate, *([baseline_return] if baseline_return else [])]
    groups: dict[Any, list[list[float]]] = {}
    for request_id in sorted(set.intersection(*(set(run.requests) for run in runs))):
        records = [run.requests[request_id] for run in runs]
        if not all(_valid(record, "ttft_ms", "tpot_ms") for record in records):
            continue
        if not all(same_prompt(records[0], record) for record in records[1:]):
            continue
        sums = groups.setdefault(
            records[0].get("conversation_id", request_id), [[0.0, 0.0, 0.0] for _ in runs]
        )
        for run_sums, record in zip(sums, records):
            run_sums[0] += float(record["ttft_ms"])
            run_sums[1] += float(record["tpot_ms"])
            run_sums[2] += 1
    if not groups:
        raise ValueError("no request is valid in every run")
    clusters = list(groups.values())

    def delta(picked: list[list[list[float]]]) -> float:
        scores = []
        for index in range(len(runs)):
            count = sum(group[index][2] for group in picked)
            scores.append(
                effective_request_score(
                    sum(group[index][0] for group in picked) / count,
                    sum(group[index][1] for group in picked) / count,
                )
            )
        reference = scores[0] if len(runs) == 2 else (scores[0] + scores[2]) / 2
        return scores[1] - reference

    rng = random.Random(seed)
    draws = sorted(
        delta([clusters[rng.randrange(len(clusters))] for _ in clusters]) for _ in range(samples)
    )
    low, high = percentile(draws, 0.025), percentile(draws, 0.975)
    return {
        "reference": "R0/R0-prime midpoint" if baseline_return else "R0",
        "method": "percentile bootstrap over conversations of quoted-formula ERS from request means",
        "n": int(sum(group[0][2] for group in clusters)),
        "clusters": len(clusters),
        "delta": delta(clusters),
        "bootstrap_95ci": [low, high],
        "direction": "gain" if low > 0 else "loss" if high < 0 else "uncertain",
    }


def overall_decision(
    candidate_report: dict[str, Any],
    drift_report: dict[str, Any] | None,
    score: dict[str, Any],
) -> dict[str, Any]:
    """Advisory promotion gate; ``promote`` is never set automatically.

    The ERS interval decides direction because the score is the objective: a
    slower TTFT can be worth a faster TPOT. Failures, outliers, drift and
    output agreement can only hold a candidate back.
    """

    def result(classification: str, warnings: list[str]) -> dict[str, Any]:
        clusters = score.get("clusters")
        if isinstance(clusters, int) and clusters < MIN_CLUSTERS:
            warnings = [
                *warnings,
                f"Only {clusters} conversations: a bootstrap over so few clusters understates "
                f"uncertainty; use at least {MIN_CLUSTERS} before reading the interval.",
            ]
        return {"classification": classification, "promote": False, "score": score, "warnings": warnings}

    if candidate_report["failures"]["candidate"] > candidate_report["failures"]["baseline"]:
        return result("reject_failures", ["Candidate has more failed requests than baseline."])

    # A mean-based "faster" that the typical paired request does not share is
    # carried by a few requests, e.g. cold-start JIT on a freshly started server.
    outlier_led = [
        metric
        for metric, evidence in candidate_report["metrics"].items()
        if evidence["direction"] == "faster" and float(evidence["paired_delta_median"]) >= 0
    ]
    if outlier_led:
        return result(
            "inconclusive_outlier_dominated",
            [
                "Mean paired delta is faster but the median paired delta is not for: "
                + ", ".join(outlier_led)
                + ". Inspect the slowest requests: a cold start is an artifact, while a real "
                "tail improvement should persist when the first arrivals are excluded."
            ],
        )
    if drift_report is None:
        return result(
            "incomplete_without_baseline_return",
            ["No R0-prime baseline-return run was supplied; drift cannot be excluded."],
        )

    warnings: list[str] = []
    candidate_ers_delta = float(candidate_report["quoted_ers"]["delta"])
    drift_ers_delta = float(drift_report["quoted_ers"]["delta"])
    confounded_metrics: list[str] = []
    for metric, evidence in candidate_report["metrics"].items():
        candidate_delta = float(evidence["paired_delta_mean"])
        drift_delta = float(drift_report["metrics"][metric]["paired_delta_mean"])
        # A baseline that improved by as much as the candidate explains the
        # gain even when that drift is too noisy for its own CI to exclude 0.
        if (
            evidence["direction"] == "faster"
            and drift_delta < 0
            and abs(drift_delta) >= abs(candidate_delta)
        ):
            confounded_metrics.append(metric)
    if drift_ers_delta > 0 and drift_ers_delta >= candidate_ers_delta:
        warnings.append("R0-prime ERS improved at least as much as the candidate versus initial R0.")
    if drift_report["failures"]["candidate"] > drift_report["failures"]["baseline"]:
        warnings.append("R0-prime failed more requests than R0; the environment was not stable.")
    if confounded_metrics:
        warnings.append(
            "Baseline-return drift matches or exceeds candidate mean improvement for: "
            + ", ".join(confounded_metrics)
            + "."
        )
    if warnings:
        return result("inconclusive_due_to_drift", warnings)

    if score["direction"] == "loss":
        return result("reject_score_loss", ["Quoted-formula ERS is lower than the R0/R0-prime midpoint."])
    if score["direction"] == "uncertain":
        return result("uncertain", ["The ERS confidence interval crosses zero."])

    notes: list[str] = []
    slower = [metric for metric, evidence in candidate_report["metrics"].items() if evidence["direction"] == "slower"]
    if slower:
        notes.append(f"Trade-off: slower on {', '.join(slower)}; the net score still improves.")
    pairing = candidate_report.get("pairing") or {}
    noise = (drift_report.get("pairing") or {}).get("output_match_rate")
    agreement = pairing.get("output_match_rate")
    if agreement is not None and noise is not None and agreement < noise:
        notes.append(
            f"B's outputs match R0 on {agreement:.1%} of requests versus {noise:.1%} for R0-prime; "
            "check correctness before trusting the speed-up."
        )
    if pairing.get("prompt_mismatches"):
        notes.append(f"{pairing['prompt_mismatches']} request pairs had different prompts and were excluded.")
    notes.append("Score gain passed drift checks; correctness and repeated-block gates remain.")
    return result("candidate_score_gain_pending_correctness", notes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-return", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    baseline_return = load_run(args.baseline_return) if args.baseline_return else None
    candidate_report = comparison(baseline, candidate, args.seed, args.bootstrap_samples)
    report: dict[str, Any] = {"candidate_vs_baseline": candidate_report}
    drift_report = None
    if baseline_return is not None:
        drift_report = comparison(baseline, baseline_return, args.seed + 100, args.bootstrap_samples)
        report["baseline_return_drift"] = drift_report
    score = score_bootstrap(
        baseline, candidate, baseline_return, samples=args.bootstrap_samples, seed=args.seed + 200
    )
    report["decision"] = overall_decision(candidate_report, drift_report, score)
    report["decision_note"] = (
        "The ERS interval decides direction; failures, outliers, drift and output agreement can only "
        "hold a candidate back. Correctness and repeated blocks remain human gates."
    )

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
