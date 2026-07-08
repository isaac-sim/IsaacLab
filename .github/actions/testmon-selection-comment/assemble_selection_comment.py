# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combine per-job Testmon selection-graph fragments into one sticky PR comment body.

Each package-test job renders its ``changed file -> test`` graph (see
``tools/testmon_selection_graph.py``) and uploads it as a Markdown fragment artifact.
This script concatenates those fragments under a stable marker so a single sticky PR
comment can be created or updated, and enforces GitHub's comment size limit by dropping
the heavy Mermaid diagrams (keeping the tables) and finally truncating if still too large.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Hidden HTML marker used to find and update the existing sticky comment.
MARKER = "<!-- testmon-selection-graph -->"

# GitHub rejects issue/PR comment bodies longer than this many characters.
_GITHUB_COMMENT_LIMIT = 65536
# Headroom left under the hard limit for the truncation notice and safety.
_SIZE_BUDGET = 63000

_HEADER = (
    f"{MARKER}\n"
    "## 🧪 Testmon affected-test selection\n\n"
    "These graphs show which tests CI re-ran and **why**: each changed source file points "
    "to the test files whose recorded coverage includes it. Jobs running the full suite "
    "(non-Python or untracked changes) do not appear here.\n"
)

_EMPTY_NOTE = "\n_No affected-test selection ran for this change (full suite, or no tracked files changed)._\n"


def _strip_mermaid(markdown: str) -> str:
    """Remove fenced ```mermaid blocks, leaving the accompanying tables intact."""
    lines = markdown.splitlines()
    out: list[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```mermaid"):
            in_block = True
            out.append("> _(diagram omitted to fit the comment size limit; see the table below)_")
            continue
        if in_block:
            if line.strip() == "```":
                in_block = False
            continue
        out.append(line)
    return "\n".join(out)


def assemble_comment(fragments: list[str]) -> str:
    """Assemble the sticky-comment body from fragment texts, honoring the size limit.

    Args:
        fragments: Per-job Markdown fragments, already ordered.

    Returns:
        The full comment body, guaranteed to be within GitHub's comment size limit.
    """
    if not fragments:
        return _HEADER + _EMPTY_NOTE

    body = _HEADER + "\n" + "\n\n".join(fragments) + "\n"
    if len(body) <= _SIZE_BUDGET:
        return body

    # First reduction: drop the Mermaid diagrams but keep the per-job tables.
    stripped = [_strip_mermaid(fragment) for fragment in fragments]
    body = _HEADER + "\n" + "\n\n".join(stripped) + "\n"
    if len(body) <= _SIZE_BUDGET:
        return body

    # Last resort: hard-truncate with a notice.
    notice = "\n\n> 🟠 _Selection graph truncated to fit GitHub's comment size limit._\n"
    return body[: _SIZE_BUDGET - len(notice)] + notice


def _read_fragments(directory: Path) -> list[str]:
    """Read all ``*.md`` fragments under ``directory``, sorted by path for stable order."""
    if not directory.is_dir():
        return []
    fragments = []
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            fragments.append(text)
    return fragments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fragments_dir", help="Directory containing downloaded *.md graph fragments")
    parser.add_argument("--output", required=True, help="Path to write the assembled comment body")
    args = parser.parse_args(argv)

    fragments = _read_fragments(Path(args.fragments_dir))
    body = assemble_comment(fragments)
    Path(args.output).write_text(body, encoding="utf-8")
    print(f"Assembled comment from {len(fragments)} fragment(s), {len(body)} chars", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
