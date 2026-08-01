"""Deterministic generation of the contest-style Poisson arrival plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path


@dataclass(frozen=True)
class Arrival:
    request_id: str
    conversation_id: int
    turn: int
    scheduled_s: float


def poisson_arrivals(
    conversations: int = 70,
    turns: int = 6,
    request_rate: float = 7.0,
    seed: int = 2025,
) -> list[Arrival]:
    """Generate globally Poisson arrivals and assign turns round-robin.

    This is a scheduling plan, not a full multi-turn client. A real replay must
    preserve per-conversation causality: turn N cannot be submitted until turn
    N-1 has completed and its answer has been added to the prompt.
    """

    if conversations < 1 or turns < 1 or request_rate <= 0:
        raise ValueError("conversations, turns and request_rate must be positive")
    rng = random.Random(seed)
    elapsed = 0.0
    arrivals: list[Arrival] = []
    for turn in range(1, turns + 1):
        for conversation_id in range(conversations):
            elapsed += rng.expovariate(request_rate)
            arrivals.append(
                Arrival(
                    request_id=f"c{conversation_id:03d}-t{turn:02d}",
                    conversation_id=conversation_id,
                    turn=turn,
                    scheduled_s=elapsed,
                )
            )
    return arrivals


def write_jsonl(arrivals: list[Arrival], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for arrival in arrivals:
            handle.write(json.dumps(asdict(arrival), ensure_ascii=False) + "\n")

