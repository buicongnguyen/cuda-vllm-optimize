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


if __name__ == "__main__":
    unittest.main()
