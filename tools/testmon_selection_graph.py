# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a Mermaid graph explaining which tests Testmon selects for a change and why.

Testmon stores, in its ``.testmondata`` SQLite database, the set of source files each
test executed the last time it ran (its coverage fingerprints). When a pull request
changes a tracked source file, Testmon re-runs exactly the tests whose fingerprints
include that file. This script reconstructs that ``changed file -> test`` relationship
from the database and renders it as a Mermaid diagram (plus a fallback table) suitable
for a GitHub job summary or a sticky PR comment, so reviewers can see *which* tests ran
and *why*.

The coverage graph is read from these Testmon tables (schema of pytest-testmon 2.x)::

    test_execution(id, environment_id, test_name, ...)
    test_execution_file_fp(test_execution_id, fingerprint_id)
    file_fp(id, filename, ...)

``file_fp.filename`` is stored relative to the repository root with forward slashes,
matching the paths returned by ``gh api repos/<repo>/pulls/<n>/files``, so changed-file
paths match directly. ``test_execution.test_name`` is the full pytest node id
(``path/to/test_x.py::TestClass::test_y[param]``).

Usage::

    # Changed files are read from stdin (one repo-relative path per line);
    # one or more Testmon databases are passed as positional arguments.
    printf '%s\n' source/isaaclab/isaaclab/foo.py \
        | python tools/testmon_selection_graph.py .testmon/*/.testmondata

The rendered Markdown is written to stdout.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import PurePosixPath

# Above this many diagram nodes (changed files + test nodes), the per-test-file graph is
# collapsed to per-directory to stay readable; GitHub also truncates very large diagrams.
_DEFAULT_MAX_NODES = 40

# A source-tree prefix stripped from node *labels* only (never from matching) to keep the
# diagram narrow. The full path is always kept in the fallback table.
_LABEL_STRIP_PREFIX = "source/"


def _normalize(path: str) -> str:
    """Normalize a path for matching: forward slashes, no leading ``./``."""
    path = path.replace("\\", "/")
    return path[2:] if path.startswith("./") else path


def load_coverage_edges(db_paths: list[str]) -> dict[str, set[str]]:
    """Load the ``test node id -> covered source files`` map, unioned across databases.

    Args:
        db_paths: Paths to one or more Testmon ``.testmondata`` SQLite files. Missing or
            unreadable databases are skipped with a warning on stderr rather than raising,
            so a single corrupt shard database does not suppress the whole graph.

    Returns:
        A mapping from pytest node id to the set of repo-relative source files it covers.
    """
    edges: dict[str, set[str]] = defaultdict(set)
    for db_path in db_paths:
        try:
            # ``mode=ro`` avoids creating an empty database if the path is wrong and never
            # mutates the CI-restored coverage data.
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            print(f"::warning::Could not open Testmon database {db_path}: {exc}", file=sys.stderr)
            continue
        try:
            rows = con.execute(
                """
                SELECT exec.test_name, ffp.filename
                FROM test_execution exec
                JOIN test_execution_file_fp link ON link.test_execution_id = exec.id
                JOIN file_fp ffp ON ffp.id = link.fingerprint_id
                """
            ).fetchall()
        except sqlite3.Error as exc:
            print(f"::warning::Could not query Testmon database {db_path}: {exc}", file=sys.stderr)
            continue
        finally:
            con.close()
        for test_name, filename in rows:
            if test_name and filename:
                edges[test_name].add(_normalize(filename))
    return edges


def select_affected(edges: dict[str, set[str]], changed_files: list[str]) -> tuple[dict[str, set[str]], list[str]]:
    """Map each *tracked* changed file to the test node ids that cover it.

    Args:
        edges: The ``node id -> covered files`` map from :func:`load_coverage_edges`.
        changed_files: Repo-relative paths changed by the pull request.

    Returns:
        A tuple ``(file_to_tests, untracked)`` where ``file_to_tests`` maps each changed
        file that appears in the coverage data to the node ids that would re-run for it,
        and ``untracked`` lists changed files absent from the coverage data (either not
        Python, newly added, or not yet covered by any test).
    """
    covered_by: dict[str, set[str]] = defaultdict(set)
    for node_id, files in edges.items():
        for filename in files:
            covered_by[filename].add(node_id)

    file_to_tests: dict[str, set[str]] = {}
    untracked: list[str] = []
    for changed in changed_files:
        norm = _normalize(changed)
        if norm in covered_by:
            file_to_tests[norm] = covered_by[norm]
        else:
            untracked.append(norm)
    return file_to_tests, untracked


def _test_file(node_id: str) -> str:
    """Return the test file portion of a pytest node id (before ``::``)."""
    return node_id.split("::", 1)[0]


def _label(path: str) -> str:
    """Shorten a path for a diagram label without losing the meaningful tail."""
    stripped = path[len(_LABEL_STRIP_PREFIX) :] if path.startswith(_LABEL_STRIP_PREFIX) else path
    return stripped


def _mermaid_escape(label: str) -> str:
    """Escape a label for use inside a Mermaid quoted node."""
    return label.replace('"', "&quot;")


def _render_mermaid(file_to_tests: dict[str, list[str]], collapse_dir: bool) -> str:
    """Render a ``graph LR`` of changed files (left) to test files or dirs (right)."""

    def target_of(test_file: str) -> str:
        return str(PurePosixPath(test_file).parent) if collapse_dir else test_file

    # Assign stable synthetic ids; Mermaid node ids cannot contain ``/``, ``.`` or ``[``.
    changed_ids: dict[str, str] = {}
    target_ids: dict[str, str] = {}
    lines = ["```mermaid", "graph LR"]

    for changed in sorted(file_to_tests):
        cid = f"C{len(changed_ids)}"
        changed_ids[changed] = cid
        lines.append(f'    {cid}["{_mermaid_escape(_label(changed))}"]')

    edges: set[tuple[str, str]] = set()
    for changed, node_ids in file_to_tests.items():
        for target in sorted({target_of(_test_file(n)) for n in node_ids}):
            if target not in target_ids:
                tid = f"T{len(target_ids)}"
                target_ids[target] = tid
                lines.append(f'    {tid}["{_mermaid_escape(_label(target))}"]')
            edges.add((changed_ids[changed], target_ids[target]))

    for cid, tid in sorted(edges):
        lines.append(f"    {cid} --> {tid}")

    # Tint the changed-file (source) nodes so the direction reads at a glance.
    for cid in sorted(changed_ids.values()):
        lines.append(f"    style {cid} fill:#ffe8cc,stroke:#e8830c")
    lines.append("```")
    return "\n".join(lines)


def _render_table(file_to_tests: dict[str, list[str]]) -> str:
    """Render a collapsible table of changed file -> selected test files (with counts)."""
    rows = ["<details><summary>Full changed-file → test-file mapping</summary>", "", "<br>", ""]
    rows.append("| Changed file | Test files | Test cases |")
    rows.append("|---|---|---|")
    for changed in sorted(file_to_tests):
        node_ids = file_to_tests[changed]
        test_files = sorted({_test_file(n) for n in node_ids})
        files_cell = "<br>".join(_label(f) for f in test_files)
        rows.append(f"| `{changed}` | {files_cell} | {len(node_ids)} |")
    rows.append("")
    rows.append("</details>")
    return "\n".join(rows)


def render_markdown(
    file_to_tests: dict[str, set[str]],
    untracked: list[str],
    title: str = "Testmon selection graph",
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> str:
    """Render the full Markdown block (header, Mermaid diagram, and fallback table).

    Args:
        file_to_tests: Output of :func:`select_affected` — tracked changed file to node ids.
        untracked: Changed files with no coverage mapping.
        title: Heading for the block; also used as a marker for sticky PR comments.
        max_nodes: Node budget above which the per-test-file graph collapses to per-directory.

    Returns:
        A Markdown string. When nothing is tracked, a short explanatory note is returned so
        the caller can still post a consistent comment.
    """
    file_to_tests_lists = {f: sorted(nodes) for f, nodes in file_to_tests.items()}
    all_nodes = {n for nodes in file_to_tests.values() for n in nodes}
    num_cases = len(all_nodes)
    num_test_files = len({_test_file(n) for n in all_nodes})

    out = [f"### {title}", ""]

    if not file_to_tests_lists:
        out.append(
            "🔵 No changed file matched Testmon's coverage data, so no affected-test graph is"
            " available. This is expected when the change is not tracked Python (Testmon runs the"
            " full suite) or when the changed files are newly added and not yet covered."
        )
        if untracked:
            out.append("")
            out.append("<details><summary>Changed files without coverage mapping</summary>")
            out.append("")
            for path in sorted(untracked):
                out.append(f"- `{path}`")
            out.append("")
            out.append("</details>")
        return "\n".join(out)

    out.append(
        f"Testmon selected **{num_cases}** test case(s) across **{num_test_files}** test file(s),"
        f" driven by **{len(file_to_tests_lists)}** changed source file(s)."
    )
    out.append("")

    # Node budget: changed files + distinct test files. Collapse to directories if over budget.
    num_diagram_nodes = len(file_to_tests_lists) + num_test_files
    collapse_dir = num_diagram_nodes > max_nodes
    if collapse_dir:
        num_dirs = len({str(PurePosixPath(_test_file(n)).parent) for n in all_nodes})
        if len(file_to_tests_lists) + num_dirs > max_nodes:
            # Even per-directory is too large for a legible diagram: table only.
            out.append(
                f"🟠 The selection graph has too many nodes to draw legibly"
                f" ({num_diagram_nodes} test files); showing the mapping as a table instead."
            )
            out.append("")
            out.append(_render_table(file_to_tests_lists))
            return "\n".join(out)
        out.append(
            "> Test files collapsed to their directories to keep the diagram readable;"
            " expand the table below for the exact files."
        )
        out.append("")

    out.append(_render_mermaid(file_to_tests_lists, collapse_dir=collapse_dir))
    out.append("")
    out.append(_render_table(file_to_tests_lists))

    if untracked:
        out.append("")
        out.append(f"<details><summary>{len(untracked)} changed file(s) without coverage mapping</summary>")
        out.append("")
        for path in sorted(untracked):
            out.append(f"- `{path}`")
        out.append("")
        out.append("</details>")

    return "\n".join(out)


def build_report(
    db_paths: list[str],
    changed_files: list[str],
    title: str = "Testmon selection graph",
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> str:
    """Load the coverage graph and render the Markdown report end-to-end."""
    edges = load_coverage_edges(db_paths)
    file_to_tests, untracked = select_affected(edges, changed_files)
    return render_markdown(file_to_tests, untracked, title=title, max_nodes=max_nodes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("databases", nargs="+", help="Path(s) to Testmon .testmondata SQLite files")
    parser.add_argument(
        "--title",
        default="Testmon selection graph",
        help="Heading for the report (also used as the sticky-comment marker)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=_DEFAULT_MAX_NODES,
        help="Diagram node budget before collapsing test files to directories",
    )
    args = parser.parse_args(argv)

    changed_files = [line.strip() for line in sys.stdin if line.strip()]
    print(build_report(args.databases, changed_files, title=args.title, max_nodes=args.max_nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
