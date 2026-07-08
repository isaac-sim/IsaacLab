# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for assembling the Testmon selection-graph sticky comment."""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("assemble_selection_comment.py")
_SPEC = importlib.util.spec_from_file_location("assemble_selection_comment", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_assemble_includes_marker_and_fragments() -> None:
    body = _MODULE.assemble_comment(["### job A\ngraph a", "### job B\ngraph b"])

    assert body.startswith(_MODULE.MARKER)
    assert "### job A" in body
    assert "### job B" in body


def test_assemble_empty_emits_note() -> None:
    body = _MODULE.assemble_comment([])

    assert _MODULE.MARKER in body
    assert "No affected-test selection ran" in body


def test_assemble_strips_mermaid_when_over_budget(monkeypatch) -> None:
    # Budget sits between the stripped and unstripped sizes so only the diagram is dropped.
    monkeypatch.setattr(_MODULE, "_SIZE_BUDGET", 700)
    fragment = (
        "### job A\n"
        "```mermaid\n"
        "graph LR\n" + "\n".join(f"    C{i} --> T{i}" for i in range(50)) + "\n"
        "```\n"
        "| Changed file | Test files |\n|---|---|\n| a.py | test_a.py |\n"
    )
    body = _MODULE.assemble_comment([fragment])

    assert "```mermaid" not in body
    assert "diagram omitted" in body
    # The table survives the diagram strip.
    assert "Changed file" in body


def test_assemble_hard_truncates_when_still_too_large(monkeypatch) -> None:
    monkeypatch.setattr(_MODULE, "_SIZE_BUDGET", 300)
    # A fragment with no mermaid block that still blows the budget.
    fragment = "### job A\n" + ("x" * 5000)
    body = _MODULE.assemble_comment([fragment])

    assert len(body) <= 300
    assert "truncated to fit" in body


def test_read_fragments_sorted_and_skips_empty(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a.md").write_text("### A\nbody a", encoding="utf-8")
    (tmp_path / "b" / "c.md").write_text("### C\nbody c", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")

    fragments = _MODULE._read_fragments(tmp_path)

    assert fragments == ["### A\nbody a", "### C\nbody c"]


def test_read_fragments_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert _MODULE._read_fragments(tmp_path / "nope") == []


def test_main_writes_output(tmp_path: Path) -> None:
    frag_dir = tmp_path / "frags"
    frag_dir.mkdir()
    (frag_dir / "a.md").write_text("### job A\ngraph a", encoding="utf-8")
    out = tmp_path / "comment.md"

    rc = _MODULE.main([str(frag_dir), "--output", str(out)])

    assert rc == 0
    written = out.read_text(encoding="utf-8")
    assert _MODULE.MARKER in written
    assert "### job A" in written
