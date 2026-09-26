import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from collections import Counter

from racebench.claims import load_claims, validate_claims
from racebench.cli import main
from racebench.experiments import Experiment, load_ledger, validate_ledger

HEADER = "experiment_id,parent_id,status,hypothesis,one_change,hardware,seed,ttft_ms,tpot_ms,ers,evidence\n"


def experiment(experiment_id: str = "E001", parent_id: str = "", **changes: object) -> Experiment:
    base = Experiment(
        experiment_id=experiment_id,
        parent_id=parent_id,
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
    return replace(base, **changes)


class ValidationTests(unittest.TestCase):
    def test_verified_claim_needs_source(self) -> None:
        errors = validate_claims(
            [{"id": "c1", "article_claim": "a", "finding": "f", "status": "verified", "sources": []}]
        )
        self.assertTrue(any("require a source" in error for error in errors))

    def test_malformed_claims_are_reported_not_crashed_on(self) -> None:
        errors = validate_claims(
            [
                "not an object",
                {"id": "c2", "finding": "x", "status": "contradicted", "sources": ["https://a"]},
                {"id": "c3", "article_claim": "a", "finding": "x", "status": "verified", "sources": [""]},
                {"id": "c4", "article_claim": "a", "status": "unverified", "sources": []},
            ]
        )
        self.assertTrue(any("must be a JSON object" in error for error in errors))
        self.assertTrue(any("c2: article_claim" in error for error in errors))
        self.assertTrue(any("c3: every source" in error for error in errors))
        self.assertTrue(any("c4: finding is required" in error for error in errors))

    def test_checked_in_audit_grades_every_article_claim(self) -> None:
        claims = load_claims(Path(__file__).resolve().parents[1] / "data" / "claims.json")
        self.assertEqual(validate_claims(claims), [])
        counts = Counter(claim["status"] for claim in claims)
        # docs/analysis.html charts these counts; keep the two in step.
        self.assertEqual(counts, {"verified": 1, "contradicted": 5, "inferred": 1, "unverified": 3})
        chart = (Path(__file__).resolve().parents[1] / "docs" / "analysis.html").read_text(encoding="utf-8")
        for status, count in counts.items():
            self.assertIn(f'style="--share: {count}"><b>{count}</b><small>{status}</small>', chart)

    def test_valid_experiment(self) -> None:
        self.assertEqual(validate_ledger([experiment()]), [])

    def test_parent_loops_have_no_root_baseline(self) -> None:
        errors = validate_ledger([experiment("E1", "E2"), experiment("E2", "E1"), experiment("E3", "E3")])
        self.assertIn("parent cycle: E1 -> E2 -> E1", errors)
        self.assertIn("E3: an experiment cannot be its own parent", errors)
        self.assertEqual(sum("parent cycle" in error for error in errors), 1)

    def test_non_finite_measurements_are_rejected(self) -> None:
        errors = validate_ledger([experiment(ttft_ms=float("nan"), tpot_ms=float("inf"))])
        self.assertIn("E001: ttft_ms must be finite", errors)
        self.assertIn("E001: tpot_ms must be finite", errors)

    def test_malformed_ledger_row_names_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            path.write_text(HEADER + "E1,,local,h,c,hw,1,fast,,,e\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"ledger.csv:2 ttft_ms: not a number: 'fast'"):
                load_ledger(path)
            path.write_text(HEADER + "E1,,local,h,c\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 11 columns"):
                load_ledger(path)

    def test_cli_reports_bad_input_without_a_traceback(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = main(["score", "--ttft", "47", "--tpot", "4", "--target", "120"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR: target_score must be between zero and the score scale", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
