"""Experiment-ledger validation for submission-limited optimization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
from pathlib import Path


ALLOWED_STATUS = {"planned", "local", "submitted", "rejected", "accepted"}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    parent_id: str
    status: str
    hypothesis: str
    one_change: str
    hardware: str
    seed: str
    ttft_ms: float | None
    tpot_ms: float | None
    ers: float | None
    evidence: str


def _optional_float(value: str) -> float | None:
    return None if not value.strip() else float(value)


def load_ledger(path: Path) -> list[Experiment]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            Experiment(
                experiment_id=row["experiment_id"].strip(),
                parent_id=row["parent_id"].strip(),
                status=row["status"].strip(),
                hypothesis=row["hypothesis"].strip(),
                one_change=row["one_change"].strip(),
                hardware=row["hardware"].strip(),
                seed=row["seed"].strip(),
                ttft_ms=_optional_float(row["ttft_ms"]),
                tpot_ms=_optional_float(row["tpot_ms"]),
                ers=_optional_float(row["ers"]),
                evidence=row["evidence"].strip(),
            )
            for row in reader
        ]


def validate_ledger(experiments: list[Experiment]) -> list[str]:
    errors: list[str] = []
    ids = [experiment.experiment_id for experiment in experiments]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate experiment ids: {', '.join(sorted(duplicates))}")
    known_ids = set(ids)
    for experiment in experiments:
        prefix = experiment.experiment_id or "<missing-id>"
        if not experiment.experiment_id:
            errors.append("experiment_id is required")
        if experiment.status not in ALLOWED_STATUS:
            errors.append(f"{prefix}: unsupported status {experiment.status!r}")
        if experiment.parent_id and experiment.parent_id not in known_ids:
            errors.append(f"{prefix}: unknown parent_id {experiment.parent_id!r}")
        if not experiment.hypothesis:
            errors.append(f"{prefix}: hypothesis is required")
        if not experiment.one_change:
            errors.append(f"{prefix}: one_change is required")
        if experiment.status in {"submitted", "accepted", "rejected"}:
            if experiment.ttft_ms is None or experiment.tpot_ms is None:
                errors.append(f"{prefix}: submitted results require TTFT and TPOT")
        for label, value in (
            ("ttft_ms", experiment.ttft_ms),
            ("tpot_ms", experiment.tpot_ms),
            ("ers", experiment.ers),
        ):
            if value is not None and value < 0:
                errors.append(f"{prefix}: {label} cannot be negative")
    return errors
