# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enforce the Newton test suite's Kit-less default runtime boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TEST_ROOT = Path(__file__).parent
_MARK_EXPRESSION = "pytest.mark." + "requires_kit"
_MODULE_MARK = "pytestmark = " + _MARK_EXPRESSION
_KIT_MODULES = {"carb", "isaacsim", "omni", "usdrt"}


def _test_files() -> list[Path]:
    return sorted(path for path in _TEST_ROOT.rglob("test_*.py") if path != Path(__file__))


def _kit_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules & _KIT_MODULES


def _launches_app(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "AppLauncher":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "AppLauncher":
            return True
    return False


@pytest.mark.parametrize("test_file", _test_files(), ids=lambda path: str(path.relative_to(_TEST_ROOT)))
def test_kit_imports_are_confined_to_requires_kit_modules(test_file: Path) -> None:
    """Reject direct Kit imports in modules collected by the Kit-less lane."""
    source = test_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(test_file))
    kit_imports = _kit_imports(tree)
    if kit_imports or _launches_app(tree):
        reason = f"direct Kit imports {sorted(kit_imports)}" if kit_imports else "AppLauncher startup"
        assert _MODULE_MARK in source, f"{reason} requires a module-level marker"


@pytest.mark.parametrize("test_file", _test_files(), ids=lambda path: str(path.relative_to(_TEST_ROOT)))
def test_requires_kit_marker_is_module_level(test_file: Path) -> None:
    """Keep Kit-only files excludable before their module is imported."""
    source = test_file.read_text(encoding="utf-8")
    if _MARK_EXPRESSION in source:
        assert _MODULE_MARK in source, "requires_kit must be assigned to module-level pytestmark"
        tree = ast.parse(source, filename=str(test_file))
        assert _kit_imports(tree) or _launches_app(tree), "requires_kit module has no direct Kit dependency"
