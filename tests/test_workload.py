import unittest

from racebench.workload import poisson_arrivals


class WorkloadTests(unittest.TestCase):
    def test_shape_and_determinism(self) -> None:
        first = poisson_arrivals(conversations=3, turns=2, request_rate=2, seed=7)
        second = poisson_arrivals(conversations=3, turns=2, request_rate=2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({item.turn for item in first}, {1, 2})
        self.assertTrue(all(a.scheduled_s < b.scheduled_s for a, b in zip(first, first[1:])))

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            poisson_arrivals(request_rate=0)


if __name__ == "__main__":
    unittest.main()
