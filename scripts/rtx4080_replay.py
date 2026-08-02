"""Replay a deterministic 70-conversation, six-turn workload against vLLM.

This is a local workload emulator, not the unpublished contest evaluator. It
preserves conversation causality, streams responses, records raw client-side
timestamps and writes enough metadata to compare A/B runs.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from racebench.metrics import RequestTiming, summarize_requests
from racebench.score import effective_request_score
from racebench.workload import Arrival, poisson_arrivals


DEFAULT_TURNS = [
    "Define KV cache in two sentences for conversation {conversation_id}.",
    "Give one benefit and one cost of the mechanism you just described.",
    "Now compare prefill and decode in a compact table.",
    "Explain how continuous batching changes that comparison.",
    "Name one measurement that would falsify your bottleneck hypothesis.",
    "Summarize this conversation in no more than 80 words.",
]


@dataclass
class ReplayResult:
    request_id: str
    conversation_id: int
    turn: int
    scheduled_s: float
    submitted_s: float | None
    first_token_s: float | None
    last_token_s: float | None
    output_tokens: int | None
    token_count_source: str | None
    ttft_ms: float | None
    tpot_ms: float | None
    status_code: int | None
    error: str | None


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one OpenAI-compatible SSE data line."""

    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise ValueError("SSE payload must be a JSON object")
    return decoded


def load_turns(path: Path | None, turns: int) -> list[str]:
    if path is None:
        prompts = DEFAULT_TURNS
    else:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("turn prompt file must be a JSON array of strings")
        prompts = decoded
    if len(prompts) < turns:
        raise ValueError(f"need at least {turns} turn prompts, found {len(prompts)}")
    return prompts[:turns]


async def stream_chat(
    client: Any,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, float, float, float, int, str, int]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    submitted = perf_counter()
    first_token: float | None = None
    last_token: float | None = None
    text_parts: list[str] = []
    content_events = 0
    completion_tokens: int | None = None

    async with client.stream("POST", endpoint, json=payload) as response:
        status_code = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            event = parse_sse_line(line)
            if event is None:
                continue
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if not content:
                continue
            now = perf_counter()
            first_token = first_token or now
            last_token = now
            content_events += 1
            text_parts.append(content)

    if first_token is None or last_token is None:
        raise RuntimeError("stream completed without a content-bearing chunk")
    if completion_tokens is None:
        completion_tokens = content_events
        token_source = "content_event_fallback"
    else:
        token_source = "server_usage"
    return (
        "".join(text_parts),
        submitted,
        first_token,
        last_token,
        completion_tokens,
        token_source,
        status_code,
    )


async def replay(args: argparse.Namespace) -> tuple[list[ReplayResult], dict[str, Any]]:
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError('install the lab extra with: pip install -e ".[rtx4080]"') from error

    prompts = load_turns(args.turn_prompts, args.turns)
    arrivals = poisson_arrivals(args.conversations, args.turns, args.rate, args.seed)
    histories: dict[int, list[dict[str, str]]] = {
        conversation_id: [] for conversation_id in range(args.conversations)
    }
    conversation_error: dict[int, str | None] = {
        conversation_id: None for conversation_id in range(args.conversations)
    }
    completed = {
        (conversation_id, turn): asyncio.Event()
        for conversation_id in range(args.conversations)
        for turn in range(1, args.turns + 1)
    }
    results: list[ReplayResult] = []
    run_started = perf_counter()
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    limits = httpx.Limits(max_connections=args.conversations, max_keepalive_connections=args.conversations)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def execute(arrival: Arrival) -> None:
            if arrival.turn > 1:
                await completed[(arrival.conversation_id, arrival.turn - 1)].wait()
                if conversation_error[arrival.conversation_id] is not None:
                    results.append(
                        ReplayResult(
                            request_id=arrival.request_id,
                            conversation_id=arrival.conversation_id,
                            turn=arrival.turn,
                            scheduled_s=arrival.scheduled_s,
                            submitted_s=None,
                            first_token_s=None,
                            last_token_s=None,
                            output_tokens=None,
                            token_count_source=None,
                            ttft_ms=None,
                            tpot_ms=None,
                            status_code=None,
                            error="Skipped: an earlier turn in this conversation failed",
                        )
                    )
                    completed[(arrival.conversation_id, arrival.turn)].set()
                    return
            remaining = run_started + arrival.scheduled_s - perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)

            history = histories[arrival.conversation_id]
            user_text = prompts[arrival.turn - 1].format(
                conversation_id=arrival.conversation_id,
                turn=arrival.turn,
            )
            messages = [*history, {"role": "user", "content": user_text}]
            result = ReplayResult(
                request_id=arrival.request_id,
                conversation_id=arrival.conversation_id,
                turn=arrival.turn,
                scheduled_s=arrival.scheduled_s,
                submitted_s=None,
                first_token_s=None,
                last_token_s=None,
                output_tokens=None,
                token_count_source=None,
                ttft_ms=None,
                tpot_ms=None,
                status_code=None,
                error=None,
            )
            try:
                (
                    assistant_text,
                    submitted,
                    first_token,
                    last_token,
                    output_tokens,
                    token_source,
                    status_code,
                ) = await stream_chat(client, endpoint, args.model, messages, args.max_tokens)
                timing = RequestTiming(
                    arrival.request_id,
                    submitted,
                    first_token,
                    last_token,
                    output_tokens,
                )
                result.submitted_s = submitted - run_started
                result.first_token_s = first_token - run_started
                result.last_token_s = last_token - run_started
                result.output_tokens = output_tokens
                result.token_count_source = token_source
                result.ttft_ms = timing.ttft_ms
                result.tpot_ms = timing.tpot_ms
                result.status_code = status_code
                history.extend(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            except Exception as error:  # preserve failures as benchmark evidence
                result.error = f"{type(error).__name__}: {error}"
                conversation_error[arrival.conversation_id] = result.error
            finally:
                results.append(result)
                completed[(arrival.conversation_id, arrival.turn)].set()

        await asyncio.gather(*(execute(arrival) for arrival in arrivals))

    results.sort(key=lambda item: (item.turn, item.conversation_id))
    successful = [item for item in results if item.error is None]
    metric_ready = [item for item in successful if item.tpot_ms is not None]
    summary: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "request_rate": args.rate,
        "requested": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "metric_eligible": len(metric_ready),
        "note": "Local emulator; not the unpublished contest evaluator.",
    }
    if metric_ready:
        timings = [
            RequestTiming(
                item.request_id,
                item.submitted_s or 0.0,
                item.first_token_s or 0.0,
                item.last_token_s or 0.0,
                item.output_tokens or 1,
            )
            for item in metric_ready
        ]
        metrics = summarize_requests(timings, args.aggregate)
        summary.update(metrics)
        summary["ers_from_quoted_formula"] = effective_request_score(
            float(metrics["ttft_ms"]), float(metrics["tpot_ms"])
        )
    return results, summary


def write_results(path: Path, results: list[ReplayResult], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"record_type": "summary", **summary}, ensure_ascii=False) + "\n")
        for result in results:
            handle.write(
                json.dumps({"record_type": "request", **asdict(result)}, ensure_ascii=False) + "\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--conversations", type=int, default=70)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--rate", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--aggregate", choices=("mean", "median", "p90", "p95", "p99"), default="mean")
    parser.add_argument("--turn-prompts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/rtx4080/baseline.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.conversations, args.turns, args.rate, args.max_tokens, args.timeout) <= 0:
        raise SystemExit("conversations, turns, rate, max-tokens and timeout must be positive")
    results, summary = asyncio.run(replay(args))
    write_results(args.output, results, summary)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
