"""Metric definitions that make TTFT/TPOT aggregation explicit."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median
from typing import Iterable, Literal


Aggregate = Literal["mean", "median", "p90", "p95", "p99"]


@dataclass(frozen=True)
class RequestTiming:
    request_id: str
    submitted_s: float
    first_token_s: float
    last_token_s: float
    output_tokens: int

    def __post_init__(self) -> None:
        if not (self.submitted_s <= self.first_token_s <= self.last_token_s):
            raise ValueError("timestamps must be ordered")
        if self.output_tokens < 1:
            raise ValueError("output_tokens must be at least one")

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_s - self.submitted_s) * 1_000

    @property
    def tpot_ms(self) -> float | None:
        if self.output_tokens == 1:
            return None
        return (self.last_token_s - self.first_token_s) * 1_000 / (self.output_tokens - 1)


def percentile(values: Iterable[float], quantile: float) -> float:
    """Linear-interpolated percentile, matching the common type-7 method."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot aggregate an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(values: Iterable[float], method: Aggregate = "mean") -> float:
    collected = list(values)
    if not collected:
        raise ValueError("cannot aggregate an empty sequence")
    if method == "mean":
        return fmean(collected)
    if method == "median":
        return median(collected)
    if method.startswith("p"):
        return percentile(collected, int(method[1:]) / 100)
    raise ValueError(f"unsupported aggregate: {method}")


def summarize_requests(
    timings: Iterable[RequestTiming],
    method: Aggregate = "mean",
) -> dict[str, float | int | str]:
    requests = list(timings)
    if not requests:
        raise ValueError("at least one timing is required")
    tpots = [value for timing in requests if (value := timing.tpot_ms) is not None]
    return {
        "requests": len(requests),
        "aggregation": method,
        "ttft_ms": aggregate((timing.ttft_ms for timing in requests), method),
        "tpot_ms": aggregate(tpots, method),
    }

