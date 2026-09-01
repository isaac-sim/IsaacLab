# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that every test file's Kit markers agree with what the file actually does.

Kit-dependence is a property of *importing* a test module: a module that imports ``omni`` at
module scope, or constructs :class:`~isaaclab.app.AppLauncher` there, needs Isaac Sim running
by the time pytest imports it -- before any fixture exists. :mod:`isaaclab.test.kit` turns
that property into a declaration, reading a module's ``pytestmark`` out of its source and
booting Kit before the import, so files sharing a launch configuration can share one app.

A declaration is only useful if it cannot drift from what the file does, which is what this
test enforces:

* At most one module-scope ``pytestmark`` assignment, since a second assignment rebinds the
  name and silently discards the markers from the first.
* At most one launch marker per file. ``kit`` and ``kit_cameras`` are alternatives, so there
  is no single app that satisfies both.
* A file declaring a launch marker does not also construct ``AppLauncher`` or
  ``SimulationApp``, which would boot a second, unshared app inside the shared one.
* A file marked ``unit`` neither declares a launch marker nor imports a Kit runtime package at
  module scope, which turns that marker's registered description ("does not launch the
  simulator") into a checked invariant.
* Within :data:`_MIGRATED_ROOTS`, a module-scope Kit runtime import is backed by something
  that actually starts Kit -- a launch marker, or the file's own ``AppLauncher``.
* ``solo`` appears only alongside a launch marker. It is the only case where it changes
  anything, and writing it alone reads like a launch marker while starting no app at all.

The checks are AST-based rather than text-based because a source-text search cannot tell an
``AppLauncher`` reference in a docstring from a real call -- several Kit-free files mention
``AppLauncher`` only to document that they do not use it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from isaaclab.test.kit import KIT_MARKERS, SOLO_MARKER, module_markers

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCAN_ROOTS = ("source", "scripts")

_EXCLUDED_PARTS = frozenset(
    {
        # Own pytest.ini / rootdir; deliberately excluded from the main collector too.
        "install_ci",
        # Vendored copies of the source tree produced by the wheel builder.
        "build",
        # Virtual environments and the Isaac Sim symlink.
        ".venv",
        "env_isaaclab",
        "_isaac_sim",
    }
)

# Packages that only exist inside a running Kit application. ``pxr`` is deliberately absent:
# OpenUSD is importable Kit-lessly through the ``usd-core`` wheel, so importing it says nothing
# about whether Kit is running.
_KIT_RUNTIME_PREFIXES = ("omni", "carb", "isaacsim")

# Names whose construction starts an app the plugin does not own.
_LAUNCHER_NAMES = ("AppLauncher", "SimulationApp")

# Directories where a module-scope Kit runtime import must be backed by a launch marker or by
# the file's own AppLauncher. Grows one package at a time as files are migrated.
_MIGRATED_ROOTS = ("source/isaaclab/test/sim/",)


class _FileFacts:
    """What a single test file declares and what it actually does at module scope."""

    def __init__(self, path: Path, source: str, tree: ast.Module):
        self.path = path
        self.markers = set(module_markers(source))
        self.pytestmark_lines: list[int] = []
        self.module_scope_launchers: list[tuple[str, int]] = []
        self.kit_runtime_imports: list[tuple[str, int]] = []

        for node in _module_scope_nodes(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
            ):
                self.pytestmark_lines.append(node.lineno)

            name = _call_name(node)
            if name in _LAUNCHER_NAMES:
                self.module_scope_launchers.append((name, node.lineno))

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _KIT_RUNTIME_PREFIXES:
                        self.kit_runtime_imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.split(".")[0] in _KIT_RUNTIME_PREFIXES:
                    self.kit_runtime_imports.append((node.module, node.lineno))

        # Decorator markers (e.g. a per-test ``@pytest.mark.unit``) count toward the file's
        # marker set even though they resolve too late to influence the launch.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for decorator in node.decorator_list:
                    self.markers.update(_decorator_marker_names(decorator))

    @property
    def rel(self) -> str:
        """Repo-relative POSIX path, as it appears in an assertion message."""
        return self.path.relative_to(_REPO_ROOT).as_posix()

    @property
    def launch_markers(self) -> list[str]:
        """The launch markers this file declares, sorted."""
        return sorted(self.markers & KIT_MARKERS.keys())

    @property
    def launchers(self) -> str:
        """The file's own app constructions, formatted for an assertion message."""
        return ", ".join(f"{name} at line {line}" for name, line in self.module_scope_launchers)


def _module_scope_nodes(tree: ast.Module):
    """Yield every node that executes at module import, without entering callables.

    Descends through module-level control flow (``if`` / ``try`` / ``with``) because those
    bodies still run at import, but stops at function, class, and lambda boundaries because
    those bodies only run when called.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _call_name(node: ast.AST) -> str | None:
    """Return the called function's bare name, for ``f()`` and ``mod.f()`` alike."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _decorator_marker_names(node: ast.AST) -> list[str]:
    """Return the marker names in a ``@pytest.mark.<name>`` decorator expression."""
    if isinstance(node, ast.Call):
        return _decorator_marker_names(node.func)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
        return [node.attr]
    return []


def _iter_test_files():
    for root in _SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("test_*.py")):
            if _EXCLUDED_PARTS.isdisjoint(path.parts):
                yield path


@pytest.fixture(scope="module")
def facts() -> list[_FileFacts]:
    """Parse every test file once and return the extracted facts."""
    collected = []
    for path in _iter_test_files():
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"{path.relative_to(_REPO_ROOT).as_posix()} failed to parse: {exc}")
        collected.append(_FileFacts(path, source, tree))
    assert collected, f"no test files discovered under {_SCAN_ROOTS} -- the scan roots are wrong"
    return collected


def test_pytestmark_is_assigned_at_most_once(facts: list[_FileFacts]):
    """A second module-scope ``pytestmark`` rebinds the name and drops the first one's markers."""
    offenders = [f"{f.rel}: lines {sorted(f.pytestmark_lines)}" for f in facts if len(f.pytestmark_lines) > 1]
    assert not offenders, (
        "These files assign `pytestmark` more than once at module scope. The later assignment"
        " replaces the earlier one, so the markers declared first are silently lost:\n  "
        + "\n  ".join(offenders)
        + "\n\nFix: merge them into a single list, e.g. `pytestmark = [pytest.mark.a, pytest.mark.b]`."
    )


def test_launch_markers_are_mutually_exclusive(facts: list[_FileFacts]):
    """A file runs in exactly one launch configuration, so it declares at most one."""
    offenders = [f"{f.rel}: {', '.join(f.launch_markers)}" for f in facts if len(f.launch_markers) > 1]
    assert not offenders, (
        f"These files declare more than one of {', '.join(sorted(KIT_MARKERS))}, which are"
        " alternatives rather than nested configurations, so no single app satisfies both:\n  " + "\n  ".join(offenders)
    )


def test_solo_accompanies_a_launch_marker(facts: list[_FileFacts]):
    """`solo` only changes anything for a file that gets an app booted for it.

    A file with no launch marker already runs on its own, so `solo` alone is not merely
    redundant -- it reads like a launch marker while starting no app, which is how a file ends
    up importing ``omni`` into a process where Kit was never started.
    """
    offenders = [f.rel for f in facts if SOLO_MARKER in f.markers and not f.launch_markers]
    assert not offenders, (
        f"These files declare `{SOLO_MARKER}` without one of {', '.join(sorted(KIT_MARKERS))}, so"
        " nothing boots an app for them and the marker changes nothing:\n  "
        + "\n  ".join(offenders)
        + f"\n\nFix: add the launch marker the file needs, or drop `{SOLO_MARKER}` -- an unmarked"
        " file is never grouped with another anyway."
    )


def test_launch_marked_files_do_not_build_their_own_app(facts: list[_FileFacts]):
    """A marked file is handed the process app; building another one defeats the sharing."""
    offenders = [
        f"{f.rel}: declares `{f.launch_markers[0]}` but constructs {f.launchers}"
        for f in facts
        if f.launch_markers and f.module_scope_launchers
    ]
    assert not offenders, (
        "These files declare a launch marker and also construct their own app:\n  "
        + "\n  ".join(offenders)
        + "\n\nFix: drop the AppLauncher construction and let `isaaclab.test.kit` launch for the"
        " marker, or drop the marker and keep the file on a process of its own."
    )


def test_unit_files_do_not_touch_kit(facts: list[_FileFacts]):
    """A `unit` file must run in a process where Kit was never started."""
    offenders = []
    for f in facts:
        if "unit" not in f.markers:
            continue
        if f.module_scope_launchers:
            offenders.append(f"{f.rel}: constructs {f.launchers}")
        if f.launch_markers:
            offenders.append(f"{f.rel}: declares `{f.launch_markers[0]}`")
        if f.kit_runtime_imports:
            where = ", ".join(f"`{name}` at line {line}" for name, line in f.kit_runtime_imports)
            offenders.append(f"{f.rel}: imports {where} at module scope")

    assert not offenders, (
        "These files are marked `unit` but depend on a running Kit:\n  "
        + "\n  ".join(offenders)
        + f"\n\nKit runtime packages: {_KIT_RUNTIME_PREFIXES}."
        "\nFix: mark the file `integration` and declare `kit`, or move the Kit import inside the"
        " test function so it is not paid at collection."
    )


def test_migrated_files_start_kit_before_importing_it(facts: list[_FileFacts]):
    """In a migrated package, a Kit import at module scope needs something that booted Kit.

    Either the file declares a launch marker, in which case the plugin boots for it, or it
    still constructs its own ``AppLauncher``. A file with neither imports ``omni`` into a
    process where Kit was never started, which fails at collection.
    """
    offenders = [
        f"{f.rel}: imports `{f.kit_runtime_imports[0][0]}` at line {f.kit_runtime_imports[0][1]}"
        for f in facts
        if f.rel.startswith(_MIGRATED_ROOTS)
        and f.kit_runtime_imports
        and not f.launch_markers
        and not f.module_scope_launchers
    ]
    assert not offenders, (
        "These files import a Kit runtime package at module scope but neither declare one of"
        f" {', '.join(sorted(KIT_MARKERS))} nor construct an app themselves, so nothing starts"
        " Kit before pytest imports them:\n  " + "\n  ".join(offenders)
    )
