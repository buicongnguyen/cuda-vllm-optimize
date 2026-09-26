import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rtx4080_replay import (
    CANNED_ANSWERS,
    DEFAULT_TURNS,
    WARMUP_TURNS,
    load_turns,
    parse_sse_line,
    prompt_digest,
    render_prompt,
    replay,
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
                sse(choices=[], usage={"completion_tokens": 4, "prompt_tokens": 12}),
                "data: [DONE]",
            ],
            [0.0, 0.010, 0.050, 0.054, 0.058, 0.062],
        )
        self.assertEqual((outcome.output_tokens, outcome.content_events), (4, 3))
        self.assertEqual(outcome.prompt_tokens, 12)
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

    def test_warm_up_cannot_seed_a_measured_prefix(self) -> None:
        # Prefix caching keys on the conversation's leading tokens, which start
        # with its first user message; warm-up and measured ones share none.
        warm = [render_prompt(WARMUP_TURNS[0], index, 1) for index in range(4)]
        measured = [render_prompt(DEFAULT_TURNS[0], index, 1) for index in range(70)]
        self.assertTrue(all(os.path.commonprefix([w, m]) == "" for w in warm for m in measured))

    def test_error_event_inside_a_200_stream_is_a_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "server error in stream: engine died"):
            run_stream(
                [chunk("a"), sse(object="error", message="engine died"), "data: [DONE]"],
                [0.0, 0.05, 0.06],
            )

    def test_stream_without_finish_reason_is_truncated(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            run_stream([chunk("a"), chunk("b")], [0.0, 0.05, 0.06])

    def test_custom_prompts_may_contain_braces(self) -> None:
        self.assertEqual(
            render_prompt('Reply as JSON like {"a": 1} for {conversation_id}/{turn}', 3, 2),
            'Reply as JSON like {"a": 1} for 3/2',
        )

    def test_replay_warms_up_then_skips_turns_after_a_failure(self) -> None:
        class Client:
            payloads: list[dict[str, object]] = []

            def __init__(self, **_: object) -> None:
                Client.payloads = []

            async def __aenter__(self) -> "Client":
                return self

            async def __aexit__(self, *_: object) -> bool:
                return False

            @asynccontextmanager
            async def stream(self, method: str, url: str, json: dict[str, object]):
                Client.payloads.append(json)
                if json["messages"][-1]["content"].endswith("conversation 0."):
                    raise ConnectionError("boom")
                yield FakeResponse(
                    [chunk("ok", "stop"), sse(choices=[], usage={"completion_tokens": 2, "prompt_tokens": 9})]
                )

        fake_httpx = types.SimpleNamespace(
            Limits=lambda **_: None, Timeout=lambda *_: None, AsyncClient=Client
        )
        args = argparse.Namespace(
            turn_prompts=None, turns=2, conversations=2, rate=1000.0, seed=1, base_url="http://x",
            warmup_conversations=1, timeout=5.0, model="m", max_tokens=8, aggregate="mean",
            history="canned", output_length="fixed",
        )
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            results, summary = asyncio.run(replay(args))
        by_id = {item.request_id: item for item in results}
        self.assertTrue(all(p["messages"][0]["content"].startswith("Warm-up") for p in Client.payloads[:2]))
        self.assertEqual(summary["warmup_requests"], 2)
        self.assertIn("boom", by_id["c000-t01"].error)
        self.assertTrue(by_id["c000-t02"].error.startswith("Skipped"))
        self.assertIsNone(by_id["c001-t02"].error)
        self.assertEqual(by_id["c001-t02"].prompt_tokens, 9)
        self.assertEqual((summary["successful"], summary["failed"]), (2, 2))
        self.assertEqual((summary["conversations"], summary["turns"], summary["max_tokens"]), (2, 2, 8))
        second_turn = next(p for p in Client.payloads if p["messages"][-1]["content"] == DEFAULT_TURNS[1])
        self.assertEqual([m["role"] for m in second_turn["messages"]], ["user", "assistant", "user"])
        # Canned history: the context is the fixed answer, not this run's "ok",
        # so the same request id is the same prompt in every run.
        self.assertEqual(second_turn["messages"][1]["content"], CANNED_ANSWERS[0])
        self.assertEqual(by_id["c001-t02"].prompt_sha256, prompt_digest(second_turn["messages"]))
        self.assertEqual(by_id["c001-t02"].output_text, "ok")
        # Fixed length: every request decodes exactly max_tokens.
        self.assertTrue(all(p["ignore_eos"] and p["min_tokens"] == 8 for p in Client.payloads))
        self.assertEqual((summary["history"], summary["output_length"]), ("canned", "fixed"))

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
