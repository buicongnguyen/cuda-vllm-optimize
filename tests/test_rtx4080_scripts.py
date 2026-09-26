import asyncio
from contextlib import asynccontextmanager
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rtx4080_replay import (
    DEFAULT_TURNS,
    WARMUP_TURNS,
    load_turns,
    parse_sse_line,
    stream_chat,
    warm_up,
)


def sse(**event: object) -> str:
    return "data: " + json.dumps(event)


def chunk(content: str | None, finish_reason: str | None = None) -> str:
    return sse(choices=[{"delta": {"content": content}, "finish_reason": finish_reason}])


class FakeResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeClient:
    def __init__(self, streams: list[list[str]]) -> None:
        self.streams = list(streams)
        self.payloads: list[dict[str, object]] = []

    @asynccontextmanager
    async def stream(self, method: str, url: str, json: dict[str, object]):
        self.payloads.append(json)
        yield FakeResponse(self.streams.pop(0))


def run_stream(lines: list[str], clock: list[float]):
    with patch("scripts.rtx4080_replay.perf_counter", side_effect=clock):
        return asyncio.run(stream_chat(FakeClient([lines]), "/v1/chat/completions", "m", [], 8))


class Rtx4080ReplayTests(unittest.TestCase):
    def test_eos_finish_chunk_ends_the_tpot_interval(self) -> None:
        # Server counts 4 tokens: three content tokens plus EOS, which arrives
        # as a content-free finish chunk at t=62 ms.
        outcome = run_stream(
            [
                sse(choices=[{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]),
                chunk("a"),
                chunk("b"),
                chunk("c"),
                chunk("", "stop"),
                sse(choices=[], usage={"completion_tokens": 4}),
                "data: [DONE]",
            ],
            [0.0, 0.010, 0.050, 0.054, 0.058, 0.062],
        )
        self.assertEqual((outcome.output_tokens, outcome.content_events), (4, 3))
        self.assertEqual(outcome.finish_reason, "stop")
        self.assertEqual((outcome.last_content, outcome.last_token), (0.058, 0.062))
        tpot_ms = (outcome.last_token - outcome.first_token) * 1000 / (outcome.output_tokens - 1)
        self.assertAlmostEqual(tpot_ms, 4.0)

    def test_length_finish_on_content_chunk_keeps_last_content_time(self) -> None:
        outcome = run_stream(
            [
                chunk("a"),
                chunk("b"),
                chunk("c", "length"),
                sse(choices=[], usage={"completion_tokens": 3}),
            ],
            [0.0, 0.050, 0.054, 0.058],
        )
        self.assertEqual(outcome.last_token, outcome.last_content)
        self.assertEqual(outcome.finish_reason, "length")

    def test_event_count_fallback_uses_last_content_time(self) -> None:
        outcome = run_stream(
            [chunk("a"), chunk("b"), chunk("", "stop")],
            [0.0, 0.050, 0.054, 0.070],
        )
        self.assertEqual(outcome.token_count_source, "content_event_fallback")
        self.assertEqual((outcome.output_tokens, outcome.last_token), (2, 0.054))

    def test_warm_up_runs_unrecorded_multi_turn_conversations(self) -> None:
        answer = [chunk("ok", "stop"), sse(choices=[], usage={"completion_tokens": 2})]
        client = FakeClient([answer] * 4)
        report = asyncio.run(warm_up(client, "/v1/chat/completions", "m", 2, 8))
        self.assertEqual(report["warmup_requests"], 4)
        second_turns = [p["messages"] for p in client.payloads if len(p["messages"]) == 3]
        self.assertEqual(len(second_turns), 2)
        self.assertEqual(second_turns[0][1], {"role": "assistant", "content": "ok"})

    def test_warm_up_prompts_do_not_share_measured_prompts(self) -> None:
        self.assertFalse(set(WARMUP_TURNS) & set(DEFAULT_TURNS))

    def test_sse_parser_accepts_openai_data_event(self) -> None:
        event = parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
        self.assertEqual(event["choices"][0]["delta"]["content"], "hi")

    def test_sse_parser_ignores_non_data_and_done(self) -> None:
        self.assertIsNone(parse_sse_line("event: message"))
        self.assertIsNone(parse_sse_line("data: [DONE]"))

    def test_default_turns_cover_contest_shape(self) -> None:
        self.assertEqual(load_turns(None, 6), DEFAULT_TURNS)

    def test_prompt_file_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "turns.json"
            path.write_text(json.dumps(["only one"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "need at least 2"):
                load_turns(path, 2)

            path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_turns(path, 1)


if __name__ == "__main__":
    unittest.main()
