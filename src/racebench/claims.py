"""Audit factual claims without silently mixing evidence and inference."""

from __future__ import annotations

import json
from pathlib import Path


ALLOWED_STATUS = {"verified", "contradicted", "inferred", "unverified"}


def load_claims(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("claims file must contain a JSON array")
    return data


def validate_claims(claims: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        claim_id = str(claim.get("id", "")).strip()
        prefix = claim_id or f"claim[{index}]"
        if not claim_id:
            errors.append(f"{prefix}: id is required")
        elif claim_id in seen:
            errors.append(f"{prefix}: duplicate id")
        seen.add(claim_id)
        status = claim.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{prefix}: unsupported status {status!r}")
        if not str(claim.get("claim", "")).strip():
            errors.append(f"{prefix}: claim text is required")
        sources = claim.get("sources", [])
        if status in {"verified", "contradicted"} and not sources:
            errors.append(f"{prefix}: {status} claims require a source")
        if not isinstance(sources, list):
            errors.append(f"{prefix}: sources must be a list")
    return errors

