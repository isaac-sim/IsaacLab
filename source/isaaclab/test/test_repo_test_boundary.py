# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Enforce the repository test rules documented under ``## Testing Guidelines`` in AGENTS.md.

Every rule checked here is mechanical. Judgement calls (is this test tautological, does this
parametrize earn its runtime) belong to review and to the ``isaaclab-writing-tests`` skill.

``test_repo_test_boundary_allowlist.txt`` records the modules that predate these rules. It may
only shrink: adding an entry needs a reason a reviewer accepts, and each package overhaul
removes its own entries.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
KIT_MODULES = {"carb", "isaacsim", "omni", "usdrt"}
# Split so this module does not match the collection hooks that scan test sources.
KIT_MARK = "pytest.mark." + "requires_kit"

# Suites with their own documented rules, validated by their own runner.
EXCLUDED_DIRS = ("install_ci",)

def _load_allowlist() -> dict[str, set[str]]:
    """Read ``<rule> <path>`` entries, ignoring blank lines and ``#`` comments."""
    allowlist: dict[str, set[str]] = {}
    text = (Path(__file__).with_name("test_repo_test_boundary_allowlist.txt")).read_text(encoding="utf-8")
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        rule, path = entry.split()
        allowlist.setdefault(rule, set()).add(path)
    return allowlist


ALLOWLIST = _load_allowlist()


def _tree_files(stem: str) -> list[Path]:
    paths = [
        path
        for pattern in (f"source/*/test/**/{stem}.py", f"scripts/**/test/**/{stem}.py")
        for path in REPO_ROOT.glob(pattern)
        if not any(part in EXCLUDED_DIRS for part in path.parts)
    ]
    return sorted(set(paths) - {Path(__file__)})


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _allowed(path: Path, rule: str) -> bool:
    return _relative(path) in ALLOWLIST.get(rule, set())


def _module_marks(tree: ast.Module) -> list[ast.Assign]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
    ]


def _toplevel_kit_imports(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules & KIT_MODULES


def _calls(node: ast.AST, names: set[str]) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id in names:
            return True
        if isinstance(func, ast.Attribute) and func.attr in names:
            return True
    return False


def _is_sys_modules(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "modules" and isinstance(node.value, ast.Name) and node.value.id == "sys"


def _mutates_sys_modules(node: ast.AST) -> bool:
    """Whether the statement assigns into or mutates ``sys.modules``."""
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and _is_sys_modules(child.value):
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                return True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"setdefault", "update", "pop", "clear"}
            and _is_sys_modules(child.func.value)
        ):
            return True
    return False


def _module_scope_statements(tree: ast.Module):
    return [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _parsed(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


# Marker and naming rules concern collection, so they apply to collected modules. Import-time
# side effects leak from any module the tests import, helpers included. ``check_*.py`` are
# standalone scripts run by hand, never imported by pytest.
TEST_FILES = _tree_files("test_*")
TREE_FILES = [path for path in _tree_files("*") if not path.name.startswith("check_")]
IDS = [_relative(path) for path in TEST_FILES]
TREE_IDS = [_relative(path) for path in TREE_FILES]


@pytest.mark.parametrize("path", TEST_FILES, ids=IDS)
def test_kit_dependency_is_declared_at_module_scope(path: Path) -> None:
    """Keep Kit-dependent modules excludable before they are imported."""
    source, tree = _parsed(path)
    kit_imports = _toplevel_kit_imports(tree)
    launches = any(_calls(node, {"AppLauncher"}) for node in _module_scope_statements(tree))
    if not (kit_imports or launches):
        return
    reason = f"top-level Kit imports {sorted(kit_imports)}" if kit_imports else "module-scope AppLauncher"
    assert any(KIT_MARK in ast.get_source_segment(source, node.value) for node in _module_marks(tree)), (
        f"{reason} requires a module-level 'pytestmark' including {KIT_MARK}"
    )


@pytest.mark.parametrize("path", TEST_FILES, ids=IDS)
def test_requires_kit_modules_actually_need_kit(path: Path) -> None:
    """Reject a Kit marker on a module that would run fine in the default lane."""
    if _allowed(path, "transitive_kit"):
        pytest.skip("Kit dependency is transitive and not statically visible")
    source, tree = _parsed(path)
    marks = _module_marks(tree)
    if not any(KIT_MARK in ast.get_source_segment(source, node.value) for node in marks):
        return
    launches = _calls(tree, {"AppLauncher"})
    assert _toplevel_kit_imports(tree) or launches, (
        f"{KIT_MARK} declared but the module has no Kit dependency; remove the marker"
    )


@pytest.mark.parametrize("path", TEST_FILES, ids=IDS)
def test_pytestmark_is_assigned_once(path: Path) -> None:
    """A second module-level assignment silently discards the first."""
    _, tree = _parsed(path)
    marks = _module_marks(tree)
    assert len(marks) <= 1, (
        f"'pytestmark' assigned {len(marks)} times at lines {[node.lineno for node in marks]};"
        " combine the marks into one list"
    )


@pytest.mark.parametrize("path", TREE_FILES, ids=TREE_IDS)
def test_no_import_time_side_effects(path: Path) -> None:
    """Import-time work runs during collection, before any marker can exclude the module."""
    if _allowed(path, "import_time_side_effect"):
        pytest.skip("allowlisted pending the package overhaul")
    _, tree = _parsed(path)
    statements = _module_scope_statements(tree)
    assert not any(_calls(node, {"parse_args", "parse_known_args"}) for node in statements), (
        "argument parsing at module scope consumes pytest's own argv and aborts collection"
    )
    assert not any(_mutates_sys_modules(node) for node in statements), (
        "mutating sys.modules at import time leaks into every later test; use a fixture that restores it"
    )


@pytest.mark.parametrize("path", TEST_FILES, ids=IDS)
def test_every_test_has_a_docstring(path: Path) -> None:
    """A test states the contract it protects."""
    if _allowed(path, "missing_docstring"):
        pytest.skip("allowlisted pending the package overhaul")
    _, tree = _parsed(path)
    undocumented = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and ast.get_docstring(node) is None
    ]
    assert not undocumented, f"tests without a docstring: {sorted(undocumented)}"


def test_module_names_are_globally_unique() -> None:
    """Same-named modules in different packages collide under pytest's prepend import mode."""
    seen: dict[str, list[str]] = {}
    for path in TEST_FILES:
        parts = [path.stem]
        directory = path.parent
        while (directory / "__init__.py").exists():
            parts.append(directory.name)
            directory = directory.parent
        seen.setdefault(".".join(reversed(parts)), []).append(_relative(path))
    collisions = {name: paths for name, paths in seen.items() if len(paths) > 1}
    assert not collisions, f"colliding test module names: {collisions}"
