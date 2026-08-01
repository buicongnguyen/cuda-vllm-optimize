"""Dependency-free structural checks for the static GitHub Pages site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.links: list[tuple[str, int]] = []
        self.sources: list[tuple[str, int]] = []
        self.lang = ""
        self.title_depth = 0
        self.title = ""
        self.h1_count = 0
        self.images_missing_alt: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang") or ""
        if tag == "title":
            self.title_depth += 1
        if tag == "h1":
            self.h1_count += 1
        element_id = values.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        if tag == "a" and values.get("href"):
            self.links.append((values["href"] or "", self.getpos()[0]))
        if tag in {"script", "img", "link"}:
            source = values.get("src") or values.get("href")
            if source:
                self.sources.append((source, self.getpos()[0]))
        if tag == "img" and "alt" not in values:
            self.images_missing_alt.append(self.getpos()[0])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title += data


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def local_target(site_root: Path, source: Path, url: str) -> tuple[Path, str] | None:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or url.startswith(("mailto:", "tel:")):
        return None
    path_text = unquote(parts.path)
    if not path_text:
        target = source
    elif path_text.startswith("/"):
        target = site_root / path_text.lstrip("/")
    else:
        target = source.parent / path_text
    target = target.resolve()
    if site_root.resolve() not in (target, *target.parents):
        raise ValueError(f"path escapes site root: {url}")
    if target.is_dir():
        target /= "index.html"
    return target, unquote(parts.fragment)


def check_site(site_root: Path) -> list[str]:
    errors: list[str] = []
    pages = sorted(site_root.rglob("*.html"))
    if not pages:
        return [f"no HTML pages found under {site_root}"]
    parsed = {path.resolve(): parse_page(path) for path in pages}

    for page, parser in parsed.items():
        relative = page.relative_to(site_root.resolve())
        if not parser.lang:
            errors.append(f"{relative}: html lang is required")
        if not parser.title.strip():
            errors.append(f"{relative}: non-empty title is required")
        if parser.h1_count != 1:
            errors.append(f"{relative}: expected one h1, found {parser.h1_count}")
        for duplicate in sorted(parser.duplicate_ids):
            errors.append(f"{relative}: duplicate id {duplicate!r}")
        for line in parser.images_missing_alt:
            errors.append(f"{relative}:{line}: image is missing alt")

        for url, line in parser.links + parser.sources:
            if url.startswith(("#", "data:")):
                if url.startswith("#") and len(url) > 1 and url[1:] not in parser.ids:
                    errors.append(f"{relative}:{line}: missing fragment {url}")
                continue
            try:
                target_info = local_target(site_root, page, url)
            except ValueError as error:
                errors.append(f"{relative}:{line}: {error}")
                continue
            if target_info is None:
                continue
            target, fragment = target_info
            if not target.exists():
                errors.append(f"{relative}:{line}: missing local target {url}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_parser = parsed.get(target.resolve()) or parse_page(target)
                if fragment not in target_parser.ids:
                    errors.append(f"{relative}:{line}: missing target fragment {url}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_root", type=Path, nargs="?", default=Path("docs"))
    args = parser.parse_args()
    errors = check_site(args.site_root.resolve())
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print(f"OK: {args.site_root} ({len(list(args.site_root.rglob('*.html')))} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
