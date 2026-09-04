#!/usr/bin/env python3
"""Vérifie les clôtures de code, liens locaux et ancres des supports Markdown."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def github_slug(title: str) -> str:
    """Approximation volontairement stricte des ancres générées par GitHub."""
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[`*_~]", "", title).strip().lower()
    title = re.sub(r"[^\w\- ]", "", title, flags=re.UNICODE)
    return title.replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: Counter[str] = Counter()
    result: set[str] = set()
    for title in HEADING_RE.findall(text):
        base = github_slug(title)
        index = seen[base]
        result.add(base if index == 0 else f"{base}-{index}")
        seen[base] += 1
    return result


def split_target(raw: str) -> tuple[str, str]:
    target = raw.strip().split(maxsplit=1)[0].strip("<>")
    path, separator, fragment = target.partition("#")
    return unquote(path), unquote(fragment) if separator else ""


def validate_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    if sum(line.lstrip().startswith("```") for line in text.splitlines()) % 2:
        errors.append(f"{path}: bloc de code non refermé")

    cache: dict[Path, set[str]] = {path.resolve(): anchors(text)}
    for raw_target in LINK_RE.findall(text):
        if raw_target.startswith(("http://", "https://", "mailto:")):
            continue

        relative, fragment = split_target(raw_target)
        target_path = (path.parent / relative).resolve() if relative else path.resolve()
        if not target_path.exists():
            errors.append(f"{path}: cible locale absente : {raw_target}")
            continue

        if fragment and target_path.suffix.lower() == ".md":
            if target_path not in cache:
                cache[target_path] = anchors(target_path.read_text(encoding="utf-8"))
            if fragment not in cache[target_path]:
                errors.append(f"{path}: ancre absente : {raw_target}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args()

    files = sorted(args.root.resolve().rglob("*.md"))
    errors = [error for path in files for error in validate_file(path)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(f"Markdown valides : {len(files)} fichier(s), liens locaux et blocs vérifiés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
