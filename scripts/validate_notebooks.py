"""Exécute les notebooks du cours de haut en bas et signale toute erreur."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def discover_notebooks() -> list[Path]:
    return sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb"))


def execute_notebook(path: Path, timeout: int, write: bool) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    if write:
        nbformat.write(notebook, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=600, help="secondes par cellule")
    parser.add_argument(
        "--write",
        action="store_true",
        help="enregistre les sorties et compteurs d'exécution dans les notebooks",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="chemins optionnels ; sans argument, valide tout le cursus",
    )
    args = parser.parse_args()

    notebooks = [
        path if path.is_absolute() else ROOT / path
        for path in args.notebooks
    ] or discover_notebooks()
    if not notebooks:
        print("Aucun notebook trouvé.", file=sys.stderr)
        return 1

    failures: list[tuple[Path, Exception]] = []
    for path in notebooks:
        relative_path = path.relative_to(ROOT)
        try:
            execute_notebook(path, timeout=args.timeout, write=args.write)
            print(f"OK   {relative_path}")
        except Exception as exc:  # Le détail nbclient est utile à l'enseignant.
            failures.append((relative_path, exc))
            print(f"FAIL {relative_path}: {exc}", file=sys.stderr)

    print(f"\n{len(notebooks) - len(failures)}/{len(notebooks)} notebooks validés.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
