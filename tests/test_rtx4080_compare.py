import json
import tempfile
import unittest
from pathlib import Path

from scripts.rtx4080_compare import (
    bootstrap_mean_ci,
    comparison,
    load_run,
    overall_decision,
    paired_deltas,
)


def write_run(path: Path, ttfts: list[float], tpots: list[float]) -> None:
    summary = {
        "record_type": "summary",
        "failed": 0,
        "ttft_ms": sum(ttfts) / len(ttfts),
        "tpot_ms": sum(tpots) / len(tpots),
    }
    records = [summary]
    for index, (ttft, tpot) in enumerate(zip(ttfts, tpots, strict=True)):
        records.append(
            {
                "record_type": "request",
                "request_id": f"r{index}",
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "error": None,
            }
        )
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


class Rtx4080CompareTests(unittest.TestCase):
    def test_paired_delta_is_candidate_minus_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path, after_path = root / "a.jsonl", root / "b.jsonl"
            write_run(before_path, [10, 20], [4, 6])
            write_run(after_path, [8, 17], [3, 4])
            before, after = load_run(before_path), load_run(after_path)
            self.assertEqual(paired_deltas(before, after, "ttft_ms"), [-2.0, -3.0])

    def test_clear_improvement_has_negative_interval(self) -> None:
        low, high = bootstrap_mean_ci([-2.0] * 20, samples=200)
        self.assertEqual((low, high), (-2.0, -2.0))

    def test_comparison_reports_direction_and_ers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path, after_path = root / "a.jsonl", root / "b.jsonl"
            write_run(before_path, [47] * 10, [4] * 10)
            write_run(after_path, [45] * 10, [3.5] * 10)
            report = comparison(load_run(before_path), load_run(after_path), 7, 200)
            self.assertEqual(report["metrics"]["tpot_ms"]["direction"], "faster")
            self.assertGreater(report["quoted_ers"]["delta"], 0)

    def test_missing_baseline_return_never_promotes(self) -> None:
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster"},
                "tpot_ms": {"direction": "faster"},
            },
            "quoted_ers": {"delta": 5.0},
        }
        decision = overall_decision(candidate, None)
        self.assertEqual(decision["classification"], "incomplete_without_baseline_return")
        self.assertFalse(decision["promote"])

    def test_baseline_return_larger_than_candidate_is_confounded(self) -> None:
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -10.0},
                "tpot_ms": {"direction": "faster", "paired_delta_mean": -0.2},
            },
            "quoted_ers": {"delta": 4.0},
        }
        drift = {
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -12.0},
                "tpot_ms": {"direction": "faster", "paired_delta_mean": -0.3},
            },
            "quoted_ers": {"delta": 5.0},
        }
        decision = overall_decision(candidate, drift)
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertFalse(decision["promote"])


if __name__ == "__main__":
    unittest.main()
