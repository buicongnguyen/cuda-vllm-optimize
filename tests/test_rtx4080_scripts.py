import json
import tempfile
import unittest
from pathlib import Path

from scripts.rtx4080_replay import DEFAULT_TURNS, load_turns, parse_sse_line


class Rtx4080ReplayTests(unittest.TestCase):
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
