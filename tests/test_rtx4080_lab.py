import contextlib
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import rtx4080_lab, rtx4080_manifest
from scripts.rtx4080_lab import (
    ServerConfig,
    build_parser,
    config_diff,
    execute_run,
    managed_server,
    option_map,
    parse_args_file,
    port_in_use,
    server_command,
    server_environment,
)


def dry_run_plan(*extra: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        args = build_parser().parse_args(["run", "--dry-run", "--output-dir", directory, *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            execute_run(args)
        return json.loads((Path(directory) / "experiment-plan.json").read_text(encoding="utf-8"))


class Rtx4080LabTests(unittest.TestCase):
    def test_args_file_ignores_comments_and_preserves_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.args"
            path.write_text(
                "# comment\norg/model\n--dtype=auto\n--max-num-seqs 80\n--flag\n",
                encoding="utf-8",
            )
            config = parse_args_file(path)
            self.assertEqual(config.model, "org/model")
            self.assertEqual(
                option_map(config.arguments),
                {"--dtype": "auto", "--max-num-seqs": "80", "--flag": True},
            )

    def test_short_flags_and_negative_values_are_parsed(self) -> None:
        self.assertEqual(
            option_map(("-O3", "--seed", "-1", "--flag")),
            {"-O3": True, "--seed": "-1", "--flag": True},
        )
        with self.assertRaisesRegex(ValueError, "positional"):
            option_map(("stray",))

    def test_occupied_port_is_detected_before_starting_a_server(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            self.assertTrue(port_in_use(url))
            with self.assertRaisesRegex(RuntimeError, "already has a listener"):
                with managed_server(["unused"], url, 1, Path("unused.log")):
                    pass
        self.assertFalse(port_in_use(url))

    def test_config_diff_reports_one_variable_candidate(self) -> None:
        baseline = ServerConfig(Path("r0.args"), "org/model", ("--dtype=auto",))
        candidate = ServerConfig(
            Path("b.args"),
            "org/model",
            ("--dtype=auto", "--enable-prefix-caching"),
        )
        self.assertEqual(
            config_diff(baseline, candidate),
            {"--enable-prefix-caching": {"baseline": None, "candidate": True}},
        )

    def test_server_command_contains_model_config_and_port(self) -> None:
        config = ServerConfig(Path("r0.args"), "org/model", ("--dtype=auto",))
        command = server_command(config, 8123)
        self.assertEqual(command[1:4], ["serve", "org/model", "--dtype=auto"])
        self.assertEqual(command[-2:], ["--port", "8123"])

    def test_empty_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.args"
            path.write_text("# only a comment\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                parse_args_file(path)

    def test_server_disables_flashinfer_sampler_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(server_environment()["VLLM_USE_FLASHINFER_SAMPLER"], "0")

    def test_stage_passes_warmup_and_survives_recorded_failures(self) -> None:
        calls: list[tuple[list[str], tuple[int, ...]]] = []

        def fake_run(command: list[str], allowed_codes: tuple[int, ...] = (0,)) -> int:
            calls.append((command, allowed_codes))
            return rtx4080_lab.REPLAY_RECORDED_FAILURES if "rtx4080_replay.py" in command[1] else 0

        config = parse_args_file(Path(__file__).resolve().parents[1] / "configs/vllm/rtx4080-r0.args")
        with tempfile.TemporaryDirectory() as directory, patch.object(
            rtx4080_lab, "run_checked", fake_run
        ), patch.object(rtx4080_lab, "managed_server", lambda *_: contextlib.nullcontext()), contextlib.redirect_stdout(
            io.StringIO()
        ):
            rtx4080_lab.run_stage(
                label="B-candidate", config=config, run_dir=Path(directory), port=8123,
                startup_timeout=1, conversations=2, turns=2, rate=1.0, max_tokens=8,
                warmup_conversations=4, seed=1,
            )
        replay_command, allowed = next(call for call in calls if "rtx4080_replay.py" in call[0][1])
        self.assertEqual(replay_command[replay_command.index("--warmup-conversations") + 1], "4")
        self.assertIn(rtx4080_lab.REPLAY_RECORDED_FAILURES, allowed)
        self.assertNotEqual(rtx4080_lab.REPLAY_RECORDED_FAILURES, 2)  # argparse usage errors exit 2

    def test_every_measured_stage_is_warmed_up_before_the_clock(self) -> None:
        self.assertEqual(dry_run_plan("--mode", "aba")["shape"]["warmup_conversations"], 4)
        self.assertEqual(dry_run_plan("--mode", "smoke")["shape"]["warmup_conversations"], 1)
        self.assertEqual(
            dry_run_plan("--mode", "baseline", "--warmup-conversations", "0")["shape"][
                "warmup_conversations"
            ],
            0,
        )

    def test_mirror_marker_keeps_dirty_tree_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / ".source-git-sha"
            with patch.object(rtx4080_manifest, "ROOT", root), patch.object(
                rtx4080_manifest, "command_output", return_value=None
            ):
                self.assertEqual(rtx4080_manifest.source_git_state(), (None, None))
                marker.write_text("abc123-dirty\n", encoding="utf-8")
                self.assertEqual(rtx4080_manifest.source_git_state(), ("abc123", True))
                marker.write_text("abc123\n", encoding="utf-8")
                self.assertEqual(rtx4080_manifest.source_git_state(), ("abc123", False))

    def test_checked_in_candidate_changes_only_prefix_cache(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = parse_args_file(root / "configs/vllm/rtx4080-r0.args")
        candidate = parse_args_file(root / "configs/vllm/rtx4080-prefix-cache.args")
        self.assertEqual(
            config_diff(baseline, candidate),
            {"--enable-prefix-caching": {"baseline": None, "candidate": True}},
        )


if __name__ == "__main__":
    unittest.main()
