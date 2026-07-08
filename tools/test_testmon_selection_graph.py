# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Testmon selection-graph renderer."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("testmon_selection_graph.py")
_SPEC = importlib.util.spec_from_file_location("testmon_selection_graph", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _make_testmon_db(path: Path, coverage: dict[str, list[str]]) -> None:
    """Write a minimal Testmon-schema SQLite DB mapping node ids to covered files.

    Args:
        path: Destination ``.testmondata`` path.
        coverage: Mapping from pytest node id to the repo-relative files it covers.
    """
    con = sqlite3.connect(path)
    # Only the columns the renderer queries are created; this mirrors the pytest-testmon
    # 2.x schema (test_execution / test_execution_file_fp / file_fp).
    con.executescript(
        """
        CREATE TABLE test_execution (id INTEGER PRIMARY KEY, environment_id INTEGER, test_name TEXT);
        CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT);
        CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER);
        """
    )
    file_ids: dict[str, int] = {}
    for node_id, files in coverage.items():
        cur = con.execute("INSERT INTO test_execution (test_name) VALUES (?)", (node_id,))
        te_id = cur.lastrowid
        for filename in files:
            if filename not in file_ids:
                cur = con.execute("INSERT INTO file_fp (filename) VALUES (?)", (filename,))
                file_ids[filename] = cur.lastrowid
            con.execute(
                "INSERT INTO test_execution_file_fp (test_execution_id, fingerprint_id) VALUES (?, ?)",
                (te_id, file_ids[filename]),
            )
    con.commit()
    con.close()


def test_load_coverage_edges_unions_across_databases(tmp_path: Path) -> None:
    db_a = tmp_path / "a.testmondata"
    db_b = tmp_path / "b.testmondata"
    _make_testmon_db(db_a, {"pkg/test_x.py::test_1": ["pkg/foo.py"]})
    _make_testmon_db(db_b, {"pkg/test_x.py::test_1": ["pkg/bar.py"], "pkg/test_y.py::test_2": ["pkg/foo.py"]})

    edges = _MODULE.load_coverage_edges([str(db_a), str(db_b)])

    assert edges["pkg/test_x.py::test_1"] == {"pkg/foo.py", "pkg/bar.py"}
    assert edges["pkg/test_y.py::test_2"] == {"pkg/foo.py"}


def test_select_affected_maps_changed_files_to_tests(tmp_path: Path) -> None:
    db = tmp_path / "a.testmondata"
    _make_testmon_db(
        db,
        {
            "pkg/test_x.py::test_1": ["pkg/foo.py"],
            "pkg/test_x.py::test_2": ["pkg/foo.py", "pkg/bar.py"],
            "pkg/test_y.py::test_3": ["pkg/bar.py"],
        },
    )
    edges = _MODULE.load_coverage_edges([str(db)])

    file_to_tests, untracked = _MODULE.select_affected(edges, ["pkg/foo.py", "pkg/new.py"])

    assert file_to_tests["pkg/foo.py"] == {"pkg/test_x.py::test_1", "pkg/test_x.py::test_2"}
    assert "pkg/bar.py" not in file_to_tests  # not among the changed files
    assert untracked == ["pkg/new.py"]  # changed but no coverage mapping


def test_select_affected_normalizes_backslash_paths(tmp_path: Path) -> None:
    db = tmp_path / "a.testmondata"
    _make_testmon_db(db, {"pkg/test_x.py::test_1": ["pkg/foo.py"]})
    edges = _MODULE.load_coverage_edges([str(db)])

    file_to_tests, untracked = _MODULE.select_affected(edges, ["pkg\\foo.py"])

    assert file_to_tests == {"pkg/foo.py": {"pkg/test_x.py::test_1"}}
    assert untracked == []


def test_render_markdown_emits_mermaid_for_small_graph() -> None:
    file_to_tests = {"source/pkg/foo.py": {"source/pkg/test/test_x.py::test_1"}}
    md = _MODULE.render_markdown(file_to_tests, untracked=[])

    assert "```mermaid" in md
    assert "graph LR" in md
    # source/ prefix stripped from labels; edge drawn between synthetic ids.
    assert '"pkg/foo.py"' in md
    assert '"pkg/test/test_x.py"' in md
    assert "C0 --> T0" in md
    # Fallback table is always included.
    assert "Full changed-file → test-file mapping" in md
    assert "1 test case" not in md  # sanity: count text formatting


def test_render_markdown_counts_cases_and_files() -> None:
    file_to_tests = {
        "source/pkg/foo.py": {"source/pkg/test/test_x.py::test_1", "source/pkg/test/test_x.py::test_2"},
        "source/pkg/bar.py": {"source/pkg/test/test_y.py::test_3"},
    }
    md = _MODULE.render_markdown(file_to_tests, untracked=[])

    assert "selected **3** test case(s)" in md
    assert "across **2** test file(s)" in md
    assert "**2** changed source file(s)" in md


def test_render_markdown_collapses_to_directories_when_over_budget() -> None:
    # Few changed files but many distinct test files sharing one directory: over the
    # per-test-file budget, but collapsing the test files to their directory fits.
    file_to_tests = {
        "source/pkg/foo.py": {f"source/pkg/test/test_{i}.py::t" for i in range(20)},
        "source/pkg/bar.py": {"source/pkg/test/test_0.py::t"},
    }
    md = _MODULE.render_markdown(file_to_tests, untracked=[], max_nodes=10)

    assert "```mermaid" in md
    assert "collapsed to their directories" in md
    # Collapsed target is the shared directory, not individual files.
    assert '"pkg/test"' in md


def test_render_markdown_falls_back_to_table_when_directories_still_too_many() -> None:
    # Each test file in its own directory, so per-directory collapse cannot help.
    file_to_tests = {f"source/s{i}.py": {f"source/d{i}/test_{i}.py::t"} for i in range(30)}
    md = _MODULE.render_markdown(file_to_tests, untracked=[], max_nodes=10)

    assert "```mermaid" not in md
    assert "too many nodes to draw legibly" in md
    assert "Full changed-file → test-file mapping" in md


def test_render_markdown_handles_no_tracked_files() -> None:
    md = _MODULE.render_markdown({}, untracked=["docs/guide.md", "source/pkg/new.py"])

    assert "No changed file matched Testmon's coverage data" in md
    assert "docs/guide.md" in md
    assert "```mermaid" not in md


def test_build_report_end_to_end(tmp_path: Path) -> None:
    db = tmp_path / "a.testmondata"
    _make_testmon_db(
        db,
        {
            "source/pkg/test/test_x.py::test_1": ["source/pkg/foo.py"],
            "source/pkg/test/test_y.py::test_2": ["source/pkg/foo.py", "source/pkg/bar.py"],
        },
    )

    md = _MODULE.build_report([str(db)], ["source/pkg/foo.py"], title="My graph")

    assert "### My graph" in md
    assert "selected **2** test case(s)" in md
    assert "```mermaid" in md


def test_load_coverage_edges_skips_missing_database(tmp_path: Path, capsys) -> None:
    good = tmp_path / "good.testmondata"
    _make_testmon_db(good, {"pkg/test_x.py::test_1": ["pkg/foo.py"]})

    edges = _MODULE.load_coverage_edges([str(tmp_path / "missing.testmondata"), str(good)])

    assert edges["pkg/test_x.py::test_1"] == {"pkg/foo.py"}
    assert "Could not open Testmon database" in capsys.readouterr().err
