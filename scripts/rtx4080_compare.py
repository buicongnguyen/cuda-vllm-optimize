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
WORKLOAD_KEYS = ("model", "seed", "request_rate", "conversations", "turns", "max_tokens")


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


def paired_rows(baseline: RunData, candidate: RunData, metric: str) -> list[tuple[Any, float]]:
    """(conversation, candidate - baseline) for requests valid in both runs."""

    rows: list[tuple[Any, float]] = []
    for request_id in sorted(baseline.requests.keys() & candidate.requests.keys()):
        before = baseline.requests[request_id]
        after = candidate.requests[request_id]
        if before.get("error") is not None or after.get("error") is not None:
            continue
        if before.get(metric) is None or after.get(metric) is None:
            continue
        cluster = before.get("conversation_id", request_id)
        rows.append((cluster, float(after[metric]) - float(before[metric])))
    if not rows:
        raise ValueError(f"no paired observations for {metric}")
    return rows


def paired_deltas(baseline: RunData, candidate: RunData, metric: str) -> list[float]:
    return [delta for _, delta in paired_rows(baseline, candidate, metric)]


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


def overall_decision(
    candidate_report: dict[str, Any],
    drift_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply a conservative promotion gate to the statistical evidence."""

    warnings: list[str] = []
    if candidate_report["failures"]["candidate"] > candidate_report["failures"]["baseline"]:
        return {
            "classification": "reject_failures",
            "promote": False,
            "warnings": ["Candidate has more failed requests than baseline."],
        }

    directions = {
        metric: evidence["direction"]
        for metric, evidence in candidate_report["metrics"].items()
    }
    if "slower" in directions.values():
        return {
            "classification": "reject_slower",
            "promote": False,
            "warnings": ["At least one latency metric has a 95% CI entirely above zero."],
        }
    # A mean-based "faster" that the typical paired request does not share is
    # carried by a few requests, e.g. cold-start JIT on a freshly started server.
    outlier_led = [
        metric
        for metric, evidence in candidate_report["metrics"].items()
        if evidence["direction"] == "faster" and float(evidence["paired_delta_median"]) >= 0
    ]
    if outlier_led:
        return {
            "classification": "inconclusive_outlier_dominated",
            "promote": False,
            "warnings": [
                "Mean paired delta is faster but the median paired delta is not for: "
                + ", ".join(outlier_led)
                + ". Inspect the slowest requests (cold start, tail) before attributing a gain."
            ],
        }
    if drift_report is None:
        return {
            "classification": "incomplete_without_baseline_return",
            "promote": False,
            "warnings": [
                "No R0-prime baseline-return run was supplied; drift cannot be excluded."
            ],
        }

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
        warnings.append(
            "R0-prime ERS improved at least as much as the candidate versus initial R0."
        )
    if drift_report["failures"]["candidate"] > drift_report["failures"]["baseline"]:
        warnings.append("R0-prime failed more requests than R0; the environment was not stable.")
    if confounded_metrics:
        warnings.append(
            "Baseline-return drift matches or exceeds candidate mean improvement for: "
            + ", ".join(confounded_metrics)
            + "."
        )
    if warnings:
        return {
            "classification": "inconclusive_due_to_drift",
            "promote": False,
            "warnings": warnings,
        }

    if all(direction == "uncertain" for direction in directions.values()):
        return {
            "classification": "uncertain",
            "promote": False,
            "warnings": ["Both latency confidence intervals cross zero."],
        }

    # One faster metric does not make a better score: TPOT has ~32x the
    # per-millisecond weight of TTFT near 47/4 ms. Compare B with the R0/R0'
    # midpoint, which cancels drift that is linear in time.
    bracketed_ers_delta = float(candidate_report["quoted_ers"]["candidate"]) - (
        float(candidate_report["quoted_ers"]["baseline"]) + float(drift_report["quoted_ers"]["candidate"])
    ) / 2
    if bracketed_ers_delta <= 0:
        return {
            "classification": "no_score_gain",
            "promote": False,
            "bracketed_ers_delta": bracketed_ers_delta,
            "warnings": [
                "A latency metric improved, but quoted-formula ERS does not exceed the R0/R0-prime midpoint."
            ],
        }

    return {
        "classification": "candidate_faster_pending_correctness",
        "promote": False,
        "bracketed_ers_delta": bracketed_ers_delta,
        "warnings": [
            "Performance signal passed drift checks; correctness and repeated-block gates remain."
        ],
    }


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
    candidate_report = comparison(baseline, candidate, args.seed, args.bootstrap_samples)
    report = {"candidate_vs_baseline": candidate_report}
    if args.baseline_return:
        baseline_return = load_run(args.baseline_return)
        drift_report = comparison(
            baseline, baseline_return, args.seed + 100, args.bootstrap_samples
        )
        report["baseline_return_drift"] = drift_report
        report["decision"] = overall_decision(candidate_report, drift_report)
        report["decision_note"] = (
            "Promote performance only when candidate improvement is larger than baseline-return "
            "drift, confidence intervals support it, and separate correctness/stability gates pass."
        )
    else:
        report["decision"] = overall_decision(candidate_report, None)
        report["decision_note"] = (
            "No baseline-return run supplied; performance signal is incomplete and cannot rule out drift."
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
