import tempfile
import unittest
from pathlib import Path

from scripts.check_site import check_site


class SiteCheckTests(unittest.TestCase):
    def test_repository_site_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_site(root / "docs"), [])

    def test_missing_local_link_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text(
                '<!doctype html><html lang="en"><head><title>x</title></head>'
                '<body><h1>x</h1><a href="missing.html">missing</a></body></html>',
                encoding="utf-8",
            )
            errors = check_site(root)
            self.assertTrue(any("missing local target" in error for error in errors))

    def test_visual_flows_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        experiment = (root / "docs" / "index.html").read_text(encoding="utf-8")
        learning = (root / "docs" / "learn.html").read_text(encoding="utf-8")

        self.assertIn('id="dataflow"', experiment)
        self.assertIn('class="decision-map reveal"', experiment)
        self.assertIn("data-ttft-contribution", experiment)
        self.assertIn('class="knowledge-map reveal"', learning)
        for module in range(10):
            self.assertIn(f'href="#m{module}"', learning)

    def test_learning_modules_have_runnable_evidence_labs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        learning = (root / "docs" / "learn.html").read_text(encoding="utf-8")

        self.assertIn('class="practice-loop"', learning)
        for module in range(10):
            self.assertIn(f"CODE LAB {module:02d}", learning)
        for authority in (
            "docs.vllm.ai",
            "docs.nvidia.com",
            "triton-lang.org",
            "docs.pytorch.org",
            "numpy.org",
            "huggingface.co/LiquidAI",
        ):
            self.assertIn(authority, learning)

    def test_rtx4080_runbook_has_complete_reproduction_flow(self) -> None:
        root = Path(__file__).resolve().parents[1]
        page = (root / "docs" / "reproduce-rtx4080.html").read_text(encoding="utf-8")

        for section in (
            "contract",
            "setup",
            "install",
            "baseline",
            "replay",
            "matrix",
            "profile",
            "implement",
            "decide",
            "troubleshoot",
        ):
            self.assertIn(f'id="{section}"', page)
        self.assertIn("scripts/rtx4080_replay.py", page)
        self.assertIn("scripts/rtx4080_manifest.py", page)
        self.assertIn("scripts/rtx4080_compare.py", page)
        self.assertIn("Method reproduction—not H200 score equivalence", page)
        for step in range(9):
            self.assertIn(f"LOGIC REVIEW {step:02d}", page)

    def test_reader_documents_use_html_routes(self) -> None:
        root = Path(__file__).resolve().parents[1] / "docs"
        pages = {
            "analysis.html": "ANALYSIS.vi.md",
            "decision-flow.html": "DECISION_FLOW.vi.md",
            "score-strategy.html": "SCORE_STRATEGY.vi.md",
            "skills-roadmap.html": "SKILLS_ROADMAP.vi.md",
        }

        for page, source in pages.items():
            rendered = (root / page).read_text(encoding="utf-8")
            self.assertIn(f'data-doc-source="{source}"', rendered)
            self.assertTrue((root / source).is_file())

        for html_page in root.glob("*.html"):
            rendered = html_page.read_text(encoding="utf-8")
            self.assertNotIn('href="ANALYSIS.vi.md"', rendered)
            self.assertNotIn('href="DECISION_FLOW.vi.md"', rendered)
            self.assertNotIn('href="SCORE_STRATEGY.vi.md"', rendered)
            self.assertNotIn('href="SKILLS_ROADMAP.vi.md"', rendered)

    def test_problem_page_connects_requirements_to_experiments(self) -> None:
        root = Path(__file__).resolve().parents[1] / "docs"
        problem = (root / "problem.html").read_text(encoding="utf-8")

        self.assertIn("REQUIREMENT CONTRACT", problem)
        self.assertIn("REQUIREMENT ANALYSIS", problem)
        self.assertIn("IDEA PORTFOLIO", problem)
        self.assertIn("EXPERIMENT BACKLOG", problem)
        self.assertIn("Official spec needed", problem)
        for experiment in range(12):
            self.assertIn(f"E{experiment:02d}", problem)


if __name__ == "__main__":
    unittest.main()
