# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify lazy_export() call-site conventions and _parse_stub behaviour.

Every ``__init__.py`` that calls ``lazy_export()`` should pass no arguments.
Fallback packages and wildcard re-exports are inferred from the ``.pyi``
stub.  Passing ``packages=`` is deprecated and indicates a stub that has
not been updated with the corresponding ``from pkg import *`` line.

This test is purely static (AST-based) and requires no simulator.
"""

import ast
import os
import tempfile
from pathlib import Path

import pytest

from isaaclab.utils.module import _parse_stub

_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_lazy_export_calls() -> list[tuple[Path, int, str]]:
    """Return ``(file, lineno, source_line)`` for every ``lazy_export(...)`` with args."""
    results: list[tuple[Path, int, str]] = []
    for root, _dirs, files in os.walk(_SOURCE_ROOT):
        for fname in files:
            if fname != "__init__.py":
                continue
            path = Path(root) / fname
            try:
                source = path.read_text()
            except OSError:
                continue
            if "lazy_export" not in source:
                continue

            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_lazy_export = (isinstance(func, ast.Attribute) and func.attr == "lazy_export") or (
                    isinstance(func, ast.Name) and func.id == "lazy_export"
                )
                if not is_lazy_export:
                    continue
                if node.args or node.keywords:
                    line = source.splitlines()[node.lineno - 1].strip()
                    results.append((path, node.lineno, line))

    return sorted(results)


_VIOLATIONS = _find_lazy_export_calls()
_IDS = [f"{p.relative_to(_SOURCE_ROOT)}:{lineno}" for p, lineno, _ in _VIOLATIONS]


@pytest.mark.parametrize("violation", _VIOLATIONS or [None], ids=_IDS or ["no-violations"])
def test_lazy_export_has_no_args(violation: tuple[Path, int, str] | None):
    """lazy_export() must be called with no arguments."""
    if violation is None:
        return
    path, lineno, line = violation
    pytest.fail(
        f"{path.relative_to(_SOURCE_ROOT)}:{lineno}: {line}\n\n"
        "lazy_export() should take no arguments. Move fallback packages into\n"
        "the .pyi stub as 'from <pkg> import *' and remove the packages= arg."
    )


def test_no_lazy_export_violations_found():
    """Canary: confirm we actually scanned files (guard against broken discovery)."""
    init_count = sum(
        1
        for root, _dirs, files in os.walk(_SOURCE_ROOT)
        for f in files
        if f == "__init__.py" and "lazy_export" in (Path(root) / f).read_text(errors="ignore")
    )
    assert init_count > 0, "No __init__.py files with lazy_export() found — discovery may be broken"


# ---------------------------------------------------------------------------
# _parse_stub unit tests
# ---------------------------------------------------------------------------


def _write_stub(content: str) -> str:
    """Write *content* to a temporary ``.pyi`` file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".pyi")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_parse_stub_single_absolute_named_import():
    """Test single absolute named import extraction."""
    stub = _write_stub("from some.package import alpha, beta\n")
    try:
        _, _, _, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert "some.package" in absolute_named
    assert absolute_named["some.package"] == ["alpha", "beta"]


def test_parse_stub_multiple_absolute_named_imports():
    """Test multiple absolute named imports from different packages."""
    stub = _write_stub("from pkg_a import foo\nfrom pkg_b import bar, baz\n")
    try:
        _, _, _, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert absolute_named["pkg_a"] == ["foo"]
    assert absolute_named["pkg_b"] == ["bar", "baz"]


def test_parse_stub_same_package_multiple_lines_accumulates():
    """Test that imports from the same package on multiple lines accumulate."""
    stub = _write_stub("from pkg import a\nfrom pkg import b, c\n")
    try:
        _, _, _, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert absolute_named["pkg"] == ["a", "b", "c"]


def test_parse_stub_absolute_wildcard_not_in_absolute_named():
    """Test that absolute wildcard imports go to fallbacks, not absolute_named."""
    stub = _write_stub("from some.package import *\n")
    try:
        _, fallbacks, _, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert "some.package" in fallbacks
    assert absolute_named == {}


def test_parse_stub_relative_import_not_in_absolute_named():
    """Test that relative imports are not included in absolute_named."""
    stub = _write_stub("from .sub import foo, bar\n")
    try:
        _, _, _, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert absolute_named == {}


def test_parse_stub_mixed_import_kinds():
    """All four import kinds in one stub are routed correctly."""
    stub = _write_stub(
        "from .local import thing\nfrom .wildmod import *\nfrom abs.pkg import *\nfrom abs.other import x, y\n"
    )
    try:
        filtered_path, fallbacks, rel_wildcards, absolute_named = _parse_stub(stub)
    finally:
        if filtered_path is not None:
            os.unlink(filtered_path)
        os.unlink(stub)

    assert fallbacks == ["abs.pkg"]
    assert rel_wildcards == ["wildmod"]
    assert absolute_named == {"abs.other": ["x", "y"]}
    assert filtered_path is not None


def test_parse_stub_no_imports_returns_empty():
    """Test that a stub with no imports returns empty collections."""
    stub = _write_stub("X: int\n")
    try:
        filtered_path, fallbacks, rel_wildcards, absolute_named = _parse_stub(stub)
    finally:
        os.unlink(stub)

    assert filtered_path is None
    assert fallbacks == []
    assert rel_wildcards == []
    assert absolute_named == {}


def test_parse_stub_filtered_stub_excludes_absolute_named():
    """Absolute named imports must not leak into the filtered stub.

    The filtered stub is passed to lazy_loader which only handles relative named imports.
    """
    stub = _write_stub("from .local import thing\nfrom abs.pkg import alpha\n")
    try:
        filtered_path, _, _, _ = _parse_stub(stub)
        assert filtered_path is not None
        with open(filtered_path) as f:
            content = f.read()
        assert "alpha" not in content
        assert "abs" not in content
        assert "thing" in content
    finally:
        if filtered_path is not None:
            os.unlink(filtered_path)
        os.unlink(stub)
