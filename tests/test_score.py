import unittest

from racebench.score import (
    ScorePolicy,
    effective_request_score,
    required_tpot_ms,
    required_ttft_ms,
    sensitivity,
)


class ScoreTests(unittest.TestCase):
    def test_quoted_baseline_is_not_42(self) -> None:
        self.assertAlmostEqual(effective_request_score(47, 4), 63.1851, places=4)

    def test_endpoints(self) -> None:
        self.assertAlmostEqual(effective_request_score(10, 1), 100.0)
        self.assertAlmostEqual(effective_request_score(400, 10), 0.0)

    def test_negative_latency_rejected(self) -> None:
        with self.assertRaises(ValueError):
            effective_request_score(-1, 4)

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ScorePolicy(ttft_weight=0.7, tpot_weight=0.7)

    def test_target_72_at_ttft_47(self) -> None:
        required = required_tpot_ms(72, 47)
        self.assertIsNotNone(required)
        self.assertAlmostEqual(required, 2.9091, places=3)
        self.assertAlmostEqual(effective_request_score(47, required), 72.0, places=7)

    def test_tpot_is_more_score_sensitive_near_baseline(self) -> None:
        report = sensitivity(47, 4)
        ratio = abs(report["ers_per_tpot_ms"] / report["ers_per_ttft_ms"])
        self.assertGreater(ratio, 30)

    def test_latency_beyond_anchors_never_raises_the_score(self) -> None:
        self.assertEqual(effective_request_score(1000, 20), 0.0)
        self.assertEqual(effective_request_score(0, 0), 100.0)
        scores = [effective_request_score(ttft, 4) for ttft in range(0, 1200, 25)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_literal_quoted_formula_is_still_available(self) -> None:
        literal = ScorePolicy(clamp_components=False)
        self.assertAlmostEqual(effective_request_score(47, 4, literal), 63.1851, places=4)
        self.assertGreater(effective_request_score(1000, 20, literal), 100)

    def test_clamped_region_has_no_sign_flipped_sensitivity(self) -> None:
        report = sensitivity(1000, 20)
        self.assertEqual(report, {"ers_per_ttft_ms": 0.0, "ers_per_tpot_ms": 0.0})

    def test_unreachable_targets_report_none(self) -> None:
        self.assertIsNone(required_tpot_ms(72, 1000))
        self.assertIsNone(required_ttft_ms(72, 20))


if __name__ == "__main__":
    unittest.main()
