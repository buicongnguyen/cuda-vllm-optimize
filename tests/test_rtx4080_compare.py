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


def write_run(
    path: Path,
    ttfts: list[float],
    tpots: list[float],
    *,
    seed: int = 2025,
    summary_ttft: float | None = None,
) -> None:
    summary = {
        "record_type": "summary",
        "seed": seed,
        "failed": 0,
        "ttft_ms": summary_ttft if summary_ttft is not None else sum(ttfts) / len(ttfts),
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
                "ttft_ms": {"direction": "faster", "paired_delta_median": -1.0},
                "tpot_ms": {"direction": "faster", "paired_delta_median": -0.1},
            },
            "quoted_ers": {"delta": 5.0},
        }
        decision = overall_decision(candidate, None)
        self.assertEqual(decision["classification"], "incomplete_without_baseline_return")
        self.assertFalse(decision["promote"])

    def test_mean_gain_the_median_request_lacks_is_not_a_gain(self) -> None:
        # 2026-08-02 block: five cold-start requests (~1 s TTFT) in R0 made the
        # mean delta -11.8 ms while the median paired request was 1.3 ms slower.
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_median": 1.35},
                "tpot_ms": {"direction": "faster", "paired_delta_median": 0.009},
            },
            "quoted_ers": {"delta": 4.35},
        }
        decision = overall_decision(candidate, None)
        self.assertEqual(decision["classification"], "inconclusive_outlier_dominated")
        self.assertIn("ttft_ms, tpot_ms", decision["warnings"][0])
        self.assertFalse(decision["promote"])

    def test_baseline_return_larger_than_candidate_is_confounded(self) -> None:
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {
                    "direction": "faster",
                    "paired_delta_mean": -10.0,
                    "paired_delta_median": -9.0,
                },
                "tpot_ms": {
                    "direction": "faster",
                    "paired_delta_mean": -0.2,
                    "paired_delta_median": -0.2,
                },
            },
            "quoted_ers": {"delta": 4.0},
        }
        drift = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -12.0},
                "tpot_ms": {"direction": "faster", "paired_delta_mean": -0.3},
            },
            "quoted_ers": {"delta": 5.0},
        }
        decision = overall_decision(candidate, drift)
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertFalse(decision["promote"])

    def test_noisy_drift_as_large_as_the_gain_still_confounds_it(self) -> None:
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -10.0, "paired_delta_median": -9.0},
                "tpot_ms": {"direction": "uncertain", "paired_delta_mean": 0.0, "paired_delta_median": 0.0},
            },
            "quoted_ers": {"baseline": 60.0, "candidate": 62.35, "delta": 2.35},
        }
        drift = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                # CI crosses zero, but the baseline moved 12 ms in the gain's direction.
                "ttft_ms": {"direction": "uncertain", "paired_delta_mean": -12.0},
                "tpot_ms": {"direction": "uncertain", "paired_delta_mean": 0.1},
            },
            "quoted_ers": {"candidate": 62.1, "delta": 2.1},
        }
        decision = overall_decision(candidate, drift)
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertIn("ttft_ms", decision["warnings"][-1])

    def test_faster_ttft_with_worse_score_is_not_a_gain(self) -> None:
        # TTFT -2 ms is worth ~+0.46 ERS; TPOT +0.1 ms (inside noise) costs ~-0.74.
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -2.0, "paired_delta_median": -2.0},
                "tpot_ms": {"direction": "uncertain", "paired_delta_mean": 0.1, "paired_delta_median": 0.1},
            },
            "quoted_ers": {"baseline": 63.2, "candidate": 62.9, "delta": -0.3},
        }
        drift = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "uncertain", "paired_delta_mean": 0.0},
                "tpot_ms": {"direction": "uncertain", "paired_delta_mean": 0.0},
            },
            "quoted_ers": {"candidate": 63.2, "delta": 0.0},
        }
        decision = overall_decision(candidate, drift)
        self.assertEqual(decision["classification"], "no_score_gain")
        self.assertAlmostEqual(decision["bracketed_ers_delta"], -0.3)

    def test_baseline_return_failures_invalidate_the_block(self) -> None:
        candidate = {
            "failures": {"baseline": 0, "candidate": 0},
            "metrics": {
                "ttft_ms": {"direction": "faster", "paired_delta_mean": -2.0, "paired_delta_median": -2.0},
                "tpot_ms": {"direction": "faster", "paired_delta_mean": -0.2, "paired_delta_median": -0.2},
            },
            "quoted_ers": {"baseline": 63.0, "candidate": 65.0, "delta": 2.0},
        }
        drift = {
            "failures": {"baseline": 0, "candidate": 12},
            "metrics": {
                "ttft_ms": {"direction": "uncertain", "paired_delta_mean": 0.0},
                "tpot_ms": {"direction": "uncertain", "paired_delta_mean": 0.0},
            },
            "quoted_ers": {"candidate": 63.0, "delta": 0.0},
        }
        decision = overall_decision(candidate, drift)
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertIn("R0-prime failed more requests", decision["warnings"][0])

    def test_conversation_bootstrap_is_wider_than_request_bootstrap(self) -> None:
        # Ten conversations of six turns whose deltas move together.
        clusters = [conversation for conversation in range(10) for _ in range(6)]
        values = [float(conversation % 5) - 2.0 for conversation in clusters]
        naive = bootstrap_mean_ci(values, samples=1000, seed=3)
        clustered = bootstrap_mean_ci(values, clusters=clusters, samples=1000, seed=3)
        self.assertGreater(clustered[1] - clustered[0], 1.8 * (naive[1] - naive[0]))

    def test_runs_from_different_workloads_are_not_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path, after_path = root / "a.jsonl", root / "b.jsonl"
            write_run(before_path, [47] * 4, [4] * 4, seed=2025)
            write_run(after_path, [45] * 4, [3.5] * 4, seed=7)
            with self.assertRaisesRegex(ValueError, "seed: 2025 vs 7"):
                comparison(load_run(before_path), load_run(after_path), 7, 200)
            write_run(after_path, [45] * 3, [3.5] * 3, seed=2025)
            with self.assertRaisesRegex(ValueError, "1 unpaired"):
                comparison(load_run(before_path), load_run(after_path), 7, 200)

    def test_score_uses_request_means_not_summary_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path, after_path = root / "a.jsonl", root / "b.jsonl"
            write_run(before_path, [47] * 4, [4] * 4)
            write_run(after_path, [47] * 4, [4] * 4, summary_ttft=999.0)
            report = comparison(load_run(before_path), load_run(after_path), 7, 200)
            self.assertAlmostEqual(report["quoted_ers"]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
