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


def paired_deltas(baseline: RunData, candidate: RunData, metric: str) -> list[float]:
    shared = sorted(baseline.requests.keys() & candidate.requests.keys())
    deltas: list[float] = []
    for request_id in shared:
        before = baseline.requests[request_id]
        after = candidate.requests[request_id]
        if before.get("error") is not None or after.get("error") is not None:
            continue
        if before.get(metric) is None or after.get(metric) is None:
            continue
        deltas.append(float(after[metric]) - float(before[metric]))
    if not deltas:
        raise ValueError(f"no paired observations for {metric}")
    return deltas


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 2025,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples))
    tail = (1.0 - confidence) / 2.0
    return percentile(means, tail), percentile(means, 1.0 - tail)


def comparison(baseline: RunData, candidate: RunData, seed: int, samples: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "baseline": str(baseline.path),
        "candidate": str(candidate.path),
        "interpretation": "delta = candidate - baseline; negative latency delta is faster",
        "failures": {
            "baseline": int(baseline.summary.get("failed", 0)),
            "candidate": int(candidate.summary.get("failed", 0)),
        },
        "metrics": {},
    }
    for index, metric in enumerate(("ttft_ms", "tpot_ms")):
        deltas = paired_deltas(baseline, candidate, metric)
        ci_low, ci_high = bootstrap_mean_ci(
            deltas,
            samples=samples,
            seed=seed + index,
        )
        report["metrics"][metric] = {
            "baseline": aggregate_metric(metric_values(baseline, metric)),
            "candidate": aggregate_metric(metric_values(candidate, metric)),
            "paired_n": len(deltas),
            "paired_delta_mean": fmean(deltas),
            "paired_delta_median": median(deltas),
            "paired_delta_p95": percentile(deltas, 0.95),
            "bootstrap_mean_delta_95ci": [ci_low, ci_high],
            "direction": "faster" if ci_high < 0 else "slower" if ci_low > 0 else "uncertain",
        }
    candidate_ttft = float(candidate.summary["ttft_ms"])
    candidate_tpot = float(candidate.summary["tpot_ms"])
    report["quoted_ers"] = {
        "baseline": effective_request_score(
            float(baseline.summary["ttft_ms"]), float(baseline.summary["tpot_ms"])
        ),
        "candidate": effective_request_score(candidate_ttft, candidate_tpot),
    }
    report["quoted_ers"]["delta"] = report["quoted_ers"]["candidate"] - report["quoted_ers"]["baseline"]
    return report


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
    report = {
        "candidate_vs_baseline": comparison(
            baseline, candidate, args.seed, args.bootstrap_samples
        )
    }
    if args.baseline_return:
        baseline_return = load_run(args.baseline_return)
        report["baseline_return_drift"] = comparison(
            baseline, baseline_return, args.seed + 100, args.bootstrap_samples
        )
        report["decision_note"] = (
            "Promote performance only when candidate improvement is larger than baseline-return "
            "drift, confidence intervals support it, and separate correctness/stability gates pass."
        )
    else:
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
