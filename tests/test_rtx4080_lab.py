import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rtx4080_lab import (
    ServerConfig,
    config_diff,
    option_map,
    parse_args_file,
    server_command,
    server_environment,
)


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
