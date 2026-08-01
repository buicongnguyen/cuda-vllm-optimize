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


if __name__ == "__main__":
    unittest.main()
