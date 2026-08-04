# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard the gate's Isaac Lab imports against the checked-out source tree.

``perf_runtime.py`` is the only gate module that imports Isaac Lab, and it runs
exclusively inside the CI container -- so a stale import path is invisible to
every other test in this suite and to every local check. That is not
hypothetical: the gate was authored against ``isaaclab.test.benchmark``, #6564
moved the package to the public ``isaaclab.benchmark``, and the mismatch turned
all nine buckets into ``HARD_FAILURE(phase=import)`` while the bench jobs still
reported success.

These tests resolve each framework import statically against ``source/`` in the
current checkout, so the suite fails on the machine that rebases rather than on
the GPU runner an hour later. Static resolution is deliberate: importing
``isaaclab`` for real needs Isaac Sim, which this suite is specifically built
not to require.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
_SOURCE = _REPO_ROOT / "source"

# Gate modules that import the Isaac Lab framework. Everything else in
# tools/perf_smoke_test/ is deliberately framework-free so it can be unit tested
# without Isaac Sim; if that ever changes, add the module here.
_FRAMEWORK_IMPORTERS = ("perf_runtime.py",)

_FRAMEWORK_ROOTS = ("isaaclab", "isaacsim", "omni", "newton", "warp")


def _module_to_path(dotted: str) -> Path | None:
    """Resolve ``a.b.c`` to its file or package directory under ``source/``.

    Isaac Lab lays every package out as ``source/<pkg>/<pkg>/...``.
    """
    parts = dotted.split(".")
    pkg_root = _SOURCE / parts[0] / parts[0]
    if not pkg_root.is_dir():
        return None
    candidate = pkg_root.joinpath(*parts[1:]) if len(parts) > 1 else pkg_root
    if candidate.is_dir():
        return candidate
    py = candidate.with_suffix(".py")
    return py if py.is_file() else None


def _defines(path: Path, name: str) -> bool:
    """Whether ``path`` (a module or package) exposes ``name``."""
    if path.is_dir():
        # A submodule satisfies `from pkg import name` via the import system's
        # fromlist fallback, even when the package uses a lazy __getattr__.
        if (path / name).is_dir() or (path / f"{name}.py").is_file():
            return True
        # Otherwise it must be declared in the package's stub or __init__.
        sources = [path / "__init__.pyi", path / "__init__.py"]
    else:
        sources = [path]

    for src in sources:
        if not src.is_file():
            continue
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return True
            if isinstance(node, ast.ImportFrom):
                if any(alias.asname == name or alias.name == name for alias in node.names):
                    return True
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
                    # `__all__ = [...]` entries count as declared exports.
                    if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and elt.value == name:
                                return True
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                return True
    return False


def _framework_imports(module: str) -> list[tuple[str, tuple[str, ...], int]]:
    """Return ``(module, names, lineno)`` for framework imports in a gate module."""
    tree = ast.parse((_GATE_DIR / module).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in _FRAMEWORK_ROOTS:
                found.append((node.module, tuple(a.name for a in node.names), node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FRAMEWORK_ROOTS:
                    found.append((alias.name, (), node.lineno))
    return found


@pytest.mark.parametrize("module", _FRAMEWORK_IMPORTERS)
def test_framework_modules_exist_in_checkout(module: str) -> None:
    """Every Isaac Lab module the gate imports must exist in this checkout."""
    missing = [
        f"{module}:{lineno} imports '{dotted}', which does not exist under source/"
        for dotted, _names, lineno in _framework_imports(module)
        if _module_to_path(dotted) is None
    ]
    assert not missing, "stale framework import path(s):\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("module", _FRAMEWORK_IMPORTERS)
def test_framework_symbols_exist_in_checkout(module: str) -> None:
    """Every name imported from an Isaac Lab module must be resolvable there."""
    missing = []
    for dotted, names, lineno in _framework_imports(module):
        target = _module_to_path(dotted)
        if target is None:
            continue  # reported by the module-level test
        for name in names:
            if not _defines(target, name):
                missing.append(f"{module}:{lineno} imports '{name}' from '{dotted}', which does not provide it")
    assert not missing, "stale framework symbol(s):\n  " + "\n  ".join(missing)


def test_retired_benchmark_namespace_is_not_referenced() -> None:
    """``isaaclab.test.benchmark`` was removed in #6564 and must not come back.

    Covers comments and docstrings too: the stale path survived in six of them
    after the import statements themselves were first written.
    """
    offenders = []
    for path in sorted(_GATE_DIR.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "isaaclab.test.benchmark" in line:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "retired isaaclab.test.benchmark namespace referenced:\n  " + "\n  ".join(offenders)


def test_source_tree_is_present() -> None:
    """Fail loudly if the resolver is silently checking nothing."""
    assert (_SOURCE / "isaaclab" / "isaaclab" / "benchmark").is_dir(), (
        f"expected the Isaac Lab benchmark package under {_SOURCE}; the import guards above are vacuous without it"
    )
