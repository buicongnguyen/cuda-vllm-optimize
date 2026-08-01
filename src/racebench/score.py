"""Contest score calculation and sensitivity analysis.

The defaults reproduce the formula quoted in the supplied Vietnamese article.
They are intentionally configurable because the official evaluator definition
has not been supplied with this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ScorePolicy:
    ttft_bad_ms: float = 400.0
    ttft_span_ms: float = 390.0
    tpot_bad_ms: float = 10.0
    tpot_span_ms: float = 9.0
    ttft_weight: float = 0.5
    tpot_weight: float = 0.5
    scale: float = 100.0
    clamp_components: bool = False

    def __post_init__(self) -> None:
        if self.ttft_span_ms <= 0 or self.tpot_span_ms <= 0:
            raise ValueError("normalization spans must be positive")
        if self.ttft_weight < 0 or self.tpot_weight < 0:
            raise ValueError("weights must be non-negative")
        if abs(self.ttft_weight + self.tpot_weight - 1.0) > 1e-12:
            raise ValueError("weights must sum to one")
        if self.scale <= 0:
            raise ValueError("score scale must be positive")


def _component(value_ms: float, bad_ms: float, span_ms: float, clamp: bool) -> float:
    if value_ms < 0:
        raise ValueError("latencies cannot be negative")
    normalized = (bad_ms - value_ms) / span_ms
    if clamp:
        normalized = min(1.0, max(0.0, normalized))
    return normalized


def effective_request_score(
    ttft_ms: float,
    tpot_ms: float,
    policy: ScorePolicy | None = None,
) -> float:
    """Calculate ERS from TTFT and TPOT in milliseconds."""

    p = policy or ScorePolicy()
    ttft = _component(ttft_ms, p.ttft_bad_ms, p.ttft_span_ms, p.clamp_components)
    tpot = _component(tpot_ms, p.tpot_bad_ms, p.tpot_span_ms, p.clamp_components)
    return p.scale * (p.ttft_weight * ttft**2 + p.tpot_weight * tpot**2)


def sensitivity(
    ttft_ms: float,
    tpot_ms: float,
    policy: ScorePolicy | None = None,
) -> dict[str, float]:
    """Return local score derivatives in ERS points per millisecond.

    Negative values mean that increasing latency lowers the score. Derivatives
    are for the quoted, unclamped quadratic formula.
    """

    p = policy or ScorePolicy()
    ttft_n = _component(ttft_ms, p.ttft_bad_ms, p.ttft_span_ms, False)
    tpot_n = _component(tpot_ms, p.tpot_bad_ms, p.tpot_span_ms, False)
    return {
        "ers_per_ttft_ms": -2 * p.scale * p.ttft_weight * ttft_n / p.ttft_span_ms,
        "ers_per_tpot_ms": -2 * p.scale * p.tpot_weight * tpot_n / p.tpot_span_ms,
    }


def required_tpot_ms(
    target_score: float,
    ttft_ms: float,
    policy: ScorePolicy | None = None,
) -> float | None:
    """Solve for the largest TPOT that reaches ``target_score``.

    Returns ``None`` when the target cannot be reached for the supplied TTFT
    under the non-negative, lower-is-better branch of the formula.
    """

    p = policy or ScorePolicy()
    if not 0 <= target_score <= p.scale:
        raise ValueError("target_score must be between zero and the score scale")
    ttft_n = _component(ttft_ms, p.ttft_bad_ms, p.ttft_span_ms, False)
    remaining = target_score / p.scale - p.ttft_weight * ttft_n**2
    if remaining < 0:
        return p.tpot_bad_ms
    if p.tpot_weight == 0:
        return None
    tpot_n_sq = remaining / p.tpot_weight
    if tpot_n_sq > 1:
        return None
    tpot_n = sqrt(tpot_n_sq)
    return p.tpot_bad_ms - p.tpot_span_ms * tpot_n


def required_ttft_ms(
    target_score: float,
    tpot_ms: float,
    policy: ScorePolicy | None = None,
) -> float | None:
    """Solve for the largest TTFT that reaches ``target_score``."""

    p = policy or ScorePolicy()
    if not 0 <= target_score <= p.scale:
        raise ValueError("target_score must be between zero and the score scale")
    tpot_n = _component(tpot_ms, p.tpot_bad_ms, p.tpot_span_ms, False)
    remaining = target_score / p.scale - p.tpot_weight * tpot_n**2
    if remaining < 0:
        return p.ttft_bad_ms
    if p.ttft_weight == 0:
        return None
    ttft_n_sq = remaining / p.ttft_weight
    if ttft_n_sq > 1:
        return None
    ttft_n = sqrt(ttft_n_sq)
    return p.ttft_bad_ms - p.ttft_span_ms * ttft_n


def score_report(ttft_ms: float, tpot_ms: float, target_score: float = 72.0) -> dict[str, float | None]:
    """Create a compact report for CLI and notebooks."""

    derivatives = sensitivity(ttft_ms, tpot_ms)
    tpot_value = required_tpot_ms(target_score, ttft_ms)
    ttft_value = required_ttft_ms(target_score, tpot_ms)
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "ers": effective_request_score(ttft_ms, tpot_ms),
        **derivatives,
        "target_ers": target_score,
        "required_tpot_at_current_ttft_ms": tpot_value,
        "required_ttft_at_current_tpot_ms": ttft_value,
    }
