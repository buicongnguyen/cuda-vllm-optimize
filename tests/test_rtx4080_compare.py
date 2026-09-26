import json
import tempfile
import unittest
from pathlib import Path

from racebench.score import effective_request_score

from scripts.rtx4080_compare import (
    bootstrap_mean_ci,
    comparison,
    load_run,
    overall_decision,
    paired_deltas,
    score_bootstrap,
)


def write_run(
    path: Path,
    ttfts: list[float],
    tpots: list[float],
    *,
    seed: int = 2025,
    summary_ttft: float | None = None,
    hashes: list[str] | None = None,
    outputs: list[str] | None = None,
) -> None:
    summary = {
        "record_type": "summary",
        "seed": seed,
        "failed": 0,
        "ttft_ms": summary_ttft if summary_ttft is not None else sum(ttfts) / len(ttfts),
        "tpot_ms": sum(tpots) / len(tpots),
    }
    records: list[dict[str, object]] = [summary]
    for index, (ttft, tpot) in enumerate(zip(ttfts, tpots, strict=True)):
        record: dict[str, object] = {
            "record_type": "request",
            "request_id": f"r{index}",
            "conversation_id": index // 2,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "error": None,
        }
        if hashes is not None:
            record["prompt_sha256"] = hashes[index]
        if outputs is not None:
            record["output_text"] = outputs[index]
        records.append(record)
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def evidence(direction: str, mean: float = 0.0, median: float | None = None) -> dict[str, object]:
    return {
        "direction": direction,
        "paired_delta_mean": mean,
        "paired_delta_median": mean if median is None else median,
    }


def report(
    ttft: dict[str, object],
    tpot: dict[str, object],
    *,
    failures: tuple[int, int] = (0, 0),
    ers_delta: float = 0.0,
    match_rate: float | None = None,
) -> dict[str, object]:
    return {
        "failures": {"baseline": failures[0], "candidate": failures[1]},
        "metrics": {"ttft_ms": ttft, "tpot_ms": tpot},
        "quoted_ers": {"delta": ers_delta},
        "pairing": {"output_match_rate": match_rate, "prompt_mismatches": 0},
    }


def score(direction: str) -> dict[str, object]:
    return {"direction": direction, "delta": 0.0, "bootstrap_95ci": [0.0, 0.0]}


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
        decision = overall_decision(report(evidence("faster", -1.0), evidence("faster", -0.1)), None, score("gain"))
        self.assertEqual(decision["classification"], "incomplete_without_baseline_return")
        self.assertFalse(decision["promote"])

    def test_mean_gain_the_median_request_lacks_is_not_a_gain(self) -> None:
        # 2026-08-02 block: five cold-start requests (~1 s TTFT) in R0 made the
        # mean delta -11.8 ms while the median paired request was 1.3 ms slower.
        candidate = report(evidence("faster", -11.8, 1.35), evidence("faster", -0.195, 0.009))
        decision = overall_decision(candidate, None, score("gain"))
        self.assertEqual(decision["classification"], "inconclusive_outlier_dominated")
        self.assertIn("ttft_ms, tpot_ms", decision["warnings"][0])
        self.assertFalse(decision["promote"])

    def test_baseline_return_larger_than_candidate_is_confounded(self) -> None:
        candidate = report(evidence("faster", -10.0, -9.0), evidence("faster", -0.2), ers_delta=4.0)
        drift = report(evidence("faster", -12.0), evidence("faster", -0.3), ers_delta=5.0)
        decision = overall_decision(candidate, drift, score("gain"))
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")

    def test_noisy_drift_as_large_as_the_gain_still_confounds_it(self) -> None:
        candidate = report(evidence("faster", -10.0, -9.0), evidence("uncertain"), ers_delta=2.35)
        # CI crosses zero, but the baseline moved 12 ms in the gain's direction.
        drift = report(evidence("uncertain", -12.0), evidence("uncertain", 0.1), ers_delta=2.1)
        decision = overall_decision(candidate, drift, score("gain"))
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertIn("ttft_ms", decision["warnings"][-1])

    def test_baseline_return_failures_invalidate_the_block(self) -> None:
        candidate = report(evidence("faster", -2.0), evidence("faster", -0.2), ers_delta=2.0)
        drift = report(evidence("uncertain"), evidence("uncertain"), failures=(0, 12))
        decision = overall_decision(candidate, drift, score("gain"))
        self.assertEqual(decision["classification"], "inconclusive_due_to_drift")
        self.assertIn("R0-prime failed more requests", decision["warnings"][0])

    def test_faster_ttft_with_a_worse_score_is_rejected(self) -> None:
        # TTFT -2 ms is worth ~+0.46 ERS; TPOT +0.1 ms (inside noise) costs ~-0.74.
        candidate = report(evidence("faster", -2.0), evidence("uncertain", 0.1), ers_delta=-0.3)
        drift = report(evidence("uncertain"), evidence("uncertain"))
        decision = overall_decision(candidate, drift, score("loss"))
        self.assertEqual(decision["classification"], "reject_score_loss")

    def test_slower_ttft_is_a_trade_off_when_the_score_improves(self) -> None:
        candidate = report(evidence("slower", 0.67), evidence("faster", -0.045), ers_delta=0.2)
        drift = report(evidence("uncertain", -0.3), evidence("uncertain"))
        decision = overall_decision(candidate, drift, score("gain"))
        self.assertEqual(decision["classification"], "candidate_score_gain_pending_correctness")
        self.assertIn("Trade-off: slower on ttft_ms", decision["warnings"][0])
        self.assertFalse(decision["promote"])

    def test_output_agreement_below_baseline_noise_is_flagged(self) -> None:
        candidate = report(evidence("faster", -2.0), evidence("faster", -0.2), ers_delta=2.0, match_rate=0.80)
        drift = report(evidence("uncertain"), evidence("uncertain"), match_rate=0.97)
        decision = overall_decision(candidate, drift, score("gain"))
        self.assertTrue(any("80.0% of requests versus 97.0%" in note for note in decision["warnings"]))

    def test_score_interval_weighs_ttft_against_tpot(self) -> None:
        # B: TTFT +1 ms (-0.23 ERS) but TPOT -0.2 ms (+1.5 ERS) on every request.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("r0.jsonl", "b.jsonl", "r0p.jsonl")]
            write_run(paths[0], [20.0, 22.0] * 6, [4.0, 4.1] * 6)
            write_run(paths[1], [21.0, 23.0] * 6, [3.8, 3.9] * 6)
            write_run(paths[2], [20.0, 22.0] * 6, [4.0, 4.1] * 6)
            runs = [load_run(path) for path in paths]
            result = score_bootstrap(runs[0], runs[1], runs[2], samples=200)
        self.assertEqual(result["direction"], "gain")
        self.assertEqual((result["n"], result["clusters"]), (12, 6))
        # Means: R0 = R0' = 21 ms / 4.05 ms, B = 22 ms / 3.85 ms.
        expected = effective_request_score(22.0, 3.85) - effective_request_score(21.0, 4.05)
        self.assertAlmostEqual(result["delta"], expected)
        self.assertGreater(result["bootstrap_95ci"][0], 0)

    def test_too_few_conversations_are_flagged(self) -> None:
        few = {**score("gain"), "clusters": 4}
        decision = overall_decision(report(evidence("faster", -1.0), evidence("faster", -0.1)), None, few)
        self.assertIn("Only 4 conversations", decision["warnings"][-1])
        decision = overall_decision(
            report(evidence("faster", -1.0), evidence("faster", -0.1)), None, {**score("gain"), "clusters": 70}
        )
        self.assertFalse(any("conversations:" in warning for warning in decision["warnings"]))

    def test_pairs_with_different_prompts_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path, after_path = root / "a.jsonl", root / "b.jsonl"
            write_run(before_path, [10, 20, 30], [4, 4, 4], hashes=["x", "y", "z"], outputs=["a", "b", "c"])
            write_run(after_path, [11, 21, 99], [4, 4, 4], hashes=["x", "y", "DIFFERENT"], outputs=["a", "B", "c"])
            result = comparison(load_run(before_path), load_run(after_path), 7, 200)
        self.assertEqual(result["pairing"]["prompt_mismatches"], 1)
        self.assertEqual(result["metrics"]["ttft_ms"]["paired_n"], 2)
        self.assertEqual(result["pairing"]["output_match_rate"], 0.5)

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
