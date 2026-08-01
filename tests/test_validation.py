import unittest

from racebench.claims import validate_claims
from racebench.experiments import Experiment, validate_ledger


class ValidationTests(unittest.TestCase):
    def test_verified_claim_needs_source(self) -> None:
        errors = validate_claims(
            [{"id": "c1", "claim": "something", "status": "verified", "sources": []}]
        )
        self.assertTrue(any("require a source" in error for error in errors))

    def test_valid_experiment(self) -> None:
        experiment = Experiment(
            experiment_id="E001",
            parent_id="",
            status="planned",
            hypothesis="smaller batch reduces TTFT",
            one_change="max-num-batched-tokens: 8192 -> 4096",
            hardware="H200 MIG 1g.18gb",
            seed="2025",
            ttft_ms=None,
            tpot_ms=None,
            ers=None,
            evidence="",
        )
        self.assertEqual(validate_ledger([experiment]), [])


if __name__ == "__main__":
    unittest.main()

