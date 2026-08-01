import unittest

from racebench.metrics import RequestTiming, aggregate, percentile, summarize_requests


class MetricsTests(unittest.TestCase):
    def test_request_timing(self) -> None:
        timing = RequestTiming("r1", 1.0, 1.05, 1.09, 5)
        self.assertAlmostEqual(timing.ttft_ms, 50.0)
        self.assertAlmostEqual(timing.tpot_ms, 10.0)

    def test_one_token_has_no_tpot(self) -> None:
        timing = RequestTiming("r1", 0.0, 0.1, 0.1, 1)
        self.assertIsNone(timing.tpot_ms)

    def test_summary(self) -> None:
        timings = [
            RequestTiming("a", 0.0, 0.010, 0.014, 3),
            RequestTiming("b", 0.0, 0.020, 0.026, 3),
        ]
        summary = summarize_requests(timings)
        self.assertEqual(summary["requests"], 2)
        self.assertAlmostEqual(summary["ttft_ms"], 15.0)
        self.assertAlmostEqual(summary["tpot_ms"], 2.5)

    def test_percentile(self) -> None:
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(aggregate([1, 2, 100], "p90"), 80.4)


if __name__ == "__main__":
    unittest.main()

