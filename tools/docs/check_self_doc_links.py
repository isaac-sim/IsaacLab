# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check links that point at our own published documentation against the source tree.

The external link checker resolves URLs against the deployed site, which still serves the
previous build while a pull request is open. A change that deletes or moves a page therefore
passes its own run and breaks every later pull request instead. This check resolves the same
URLs against ``docs/`` in the working tree, so the page and the reference to it are compared
in one state.

Run directly, or via the ``check-self-doc-links`` pre-commit hook::

    python tools/docs/check_self_doc_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SKIP_DIRS = {"_build", ".venv", ".git", "node_modules", "_isaac_sim"}

# Published page URLs, e.g. https://isaac-sim.github.io/IsaacLab/develop/source/setup/index.html
SELF_DOC_URL = re.compile(
    r"https://isaac-sim\.github\.io/IsaacLab/(?:develop|main|release/[^/]+)/(source/[^)\s\"'>#]+?)\.html"
)


def _source_candidates(doc_path: str) -> list[Path]:
    """Return the source files that would build the given documentation path.

    Both suffixes are accepted because ``docs/conf.py`` registers ``.rst`` and ``.md``
    (myst-parser). The tree is almost entirely reStructuredText today, so the Markdown
    forms exist to avoid reporting a MyST page as missing.
    """
    return [
        DOCS_ROOT / f"{doc_path}.rst",
        DOCS_ROOT / f"{doc_path}.md",
        DOCS_ROOT / doc_path / "index.rst",
        DOCS_ROOT / doc_path / "index.md",
    ]


def _iter_text_files() -> list[Path]:
    """Return every Markdown and reStructuredText file that may carry a link."""
    files: list[Path] = []
    for pattern in ("*.md", "*.rst"):
        for path in REPO_ROOT.rglob(pattern):
            if not SKIP_DIRS.intersection(path.parts):
                files.append(path)
    return files


def find_broken_links() -> list[tuple[Path, int, str]]:
    """Return every self-referencing documentation link with no matching source file."""
    broken: list[tuple[Path, int, str]] = []
    for path in _iter_text_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, start=1):
            for match in SELF_DOC_URL.finditer(line):
                doc_path = match.group(1)
                if not any(candidate.exists() for candidate in _source_candidates(doc_path)):
                    broken.append((path, number, doc_path))
    return broken


def main() -> int:
    """Report self-referencing documentation links whose page no longer exists."""
    broken = find_broken_links()
    if not broken:
        return 0
    print("Links point at documentation pages that do not exist in docs/:", file=sys.stderr)
    for path, number, doc_path in broken:
        relative = path.relative_to(REPO_ROOT)
        print(f"  {relative}:{number} -> docs/{doc_path}.rst (not found)", file=sys.stderr)
    print(
        "\nThe page was renamed, moved, or removed. Update the link to the new location,"
        "\nor restore the page. Searching docs/ for the page title usually finds it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
