# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that lazy_export() call sites use no arguments.

Every ``__init__.py`` that calls ``lazy_export()`` should pass no arguments.
Fallback packages and wildcard re-exports are inferred from the ``.pyi``
stub.  Passing ``packages=`` is deprecated and indicates a stub that has
not been updated with the corresponding ``from pkg import *`` line.

This test is purely static (AST-based) and requires no simulator.
"""

import ast
import os
from pathlib import Path

import pytest

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
