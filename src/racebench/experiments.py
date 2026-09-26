"""Experiment-ledger validation for submission-limited optimization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import math
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


def _optional_float(value: str | None, where: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{where}: not a number: {value!r}") from None


def load_ledger(path: Path) -> list[Experiment]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        experiments: list[Experiment] = []
        for row in reader:
            where = f"{path}:{reader.line_num}"
            # A row with too few commas leaves trailing columns as None.
            missing = [column for column, value in row.items() if value is None]
            if missing or None in row:
                raise ValueError(f"{where}: expected {len(reader.fieldnames or [])} columns")
            experiments.append(
                Experiment(
                    experiment_id=row["experiment_id"].strip(),
                    parent_id=row["parent_id"].strip(),
                    status=row["status"].strip(),
                    hypothesis=row["hypothesis"].strip(),
                    one_change=row["one_change"].strip(),
                    hardware=row["hardware"].strip(),
                    seed=row["seed"].strip(),
                    ttft_ms=_optional_float(row["ttft_ms"], f"{where} ttft_ms"),
                    tpot_ms=_optional_float(row["tpot_ms"], f"{where} tpot_ms"),
                    ers=_optional_float(row["ers"], f"{where} ers"),
                    evidence=row["evidence"].strip(),
                )
            )
        return experiments


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
        if experiment.parent_id and experiment.parent_id == experiment.experiment_id:
            errors.append(f"{prefix}: an experiment cannot be its own parent")
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
            if value is not None and not math.isfinite(value):
                errors.append(f"{prefix}: {label} must be finite")
            elif value is not None and value < 0:
                errors.append(f"{prefix}: {label} cannot be negative")
    errors.extend(_parent_cycles(experiments))
    return errors


def _parent_cycles(experiments: list[Experiment]) -> list[str]:
    """A parent chain that loops has no root baseline to attribute changes to."""

    parent = {item.experiment_id: item.parent_id for item in experiments if item.experiment_id}
    reported: set[str] = set()
    errors: list[str] = []
    for start in parent:
        seen: list[str] = []
        node = start
        while node and node in parent and node not in seen:
            seen.append(node)
            node = parent[node]
        if node not in seen:
            continue
        cycle = seen[seen.index(node):]
        # One-node loops are reported as self-parents by validate_ledger.
        if len(cycle) > 1 and min(cycle) not in reported:
            reported.add(min(cycle))
            errors.append("parent cycle: " + " -> ".join([*cycle, node]))
    return errors
