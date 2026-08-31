# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Marker-driven Kit startup for the Isaac Lab test suite.

A test module that needs Isaac Sim declares it with a marker and nothing else::

    import pytest

    from pxr import Usd

    import isaaclab.sim as sim_utils

    pytestmark = [pytest.mark.kit, pytest.mark.integration]

This module is a pytest plugin, loaded from the repo-root ``conftest.py``. It reads that
marker out of the file's source *before* pytest imports the module and boots Kit if it is
there. Reading the source rather than the imported module is what makes the marker usable at
all: a Kit-dependent test module imports ``pxr``, ``omni``, ... at module scope, so Kit has to
be running by the time the module is imported -- which is before any fixture, and before the
module's own ``pytestmark`` exists.

Kit is booted once per process. The first marked module collected pays startup; every later
one is imported into the app that is already running, so a pytest process covering a directory
of such files boots Kit once instead of once per file.

:data:`KIT_MARKERS` gives the launch configurations. They are alternatives, not nested: a
camera-enabled app is not a superset of a plain one, because cameras cannot be enabled after
startup and some tests assert that offscreen rendering is off. A process that is handed both
kinds of file raises rather than importing one of them into the wrong app.

Tests that need the app object itself request the :func:`kit_app` fixture.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from isaacsim import SimulationApp

KIT_MARKERS: dict[str, bool] = {"kit": False, "kit_cameras": True}
"""The launch markers, mapped to the ``enable_cameras`` setting each one asks for."""

SOLO_MARKER = "kit_solo"
"""Marker that keeps a file in a process of its own, never grouped with other files."""

_app: SimulationApp | None = None
"""The app booted for this process, or None before the first marked module is collected."""

_cameras: bool = False
"""Whether :data:`_app` was booted with cameras enabled."""


"""
Marker inspection.
"""


def module_markers(source: str) -> frozenset[str]:
    """Return the marker names a test module declares, without importing it.

    Only module-scope ``pytestmark`` assignments are read, because only those are known
    before the module is imported and can therefore influence how Kit is launched. Assignments
    inside module-level ``if`` / ``try`` blocks count; per-test ``@pytest.mark`` decorators do
    not.

    Args:
        source: The module's text.

    Returns:
        Every marker name found, or an empty set if the module declares none or does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()

    names: set[str] = set()
    for node in _module_scope_nodes(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            names.update(_marker_names(node.value))
    return frozenset(names)


def kit_marker(source: str) -> str | None:
    """Return the launch marker a test module declares, or None if it needs no Kit app.

    Args:
        source: The module's text.

    Returns:
        A key of :data:`KIT_MARKERS`, or None.

    Raises:
        ValueError: If the module declares more than one launch marker. They are alternatives,
            so there is no configuration that satisfies both.
    """
    declared = sorted(module_markers(source) & KIT_MARKERS.keys())
    if len(declared) > 1:
        raise ValueError(f"a test module declares more than one launch marker: {', '.join(declared)}")
    return declared[0] if declared else None


def kit_marker_of_file(path: str | os.PathLike[str]) -> str | None:
    """Return the launch marker declared by the test file at ``path``.

    Args:
        path: Path to a test file.

    Returns:
        A key of :data:`KIT_MARKERS`, or None when the file declares none or cannot be read.
    """
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return kit_marker(source)


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


def _marker_names(node: ast.AST) -> list[str]:
    """Return the marker names in a ``pytest.mark.<name>`` expression, or a list of them."""
    if isinstance(node, ast.List | ast.Tuple):
        return [name for element in node.elts for name in _marker_names(element)]
    if isinstance(node, ast.Call):
        return _marker_names(node.func)
    # pytest.mark.<name>, i.e. an attribute whose parent attribute is `mark`
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
        return [node.attr]
    return []


"""
Launching.
"""


def _launch(*, cameras: bool) -> SimulationApp:
    """Boot this process's Kit app, or return the one already running.

    Args:
        cameras: Whether the app must have camera and render extensions enabled.

    Returns:
        The running ``SimulationApp``.

    Raises:
        RuntimeError: If Kit is already running in a configuration other than the one asked
            for, or was started by something other than this plugin. Both mean the files
            sharing this process do not share a launch configuration and must be split up.
    """
    global _app, _cameras

    if _app is not None:
        if cameras != _cameras:
            wanted, running = ("with", "without") if cameras else ("without", "with")
            raise RuntimeError(
                f"a `{_marker_for(cameras)}` file wants a Kit app {wanted} cameras, but this process is"
                f" already running one {running} them, and that cannot be changed after startup."
                f" `{_marker_for(False)}` and `{_marker_for(True)}` files need separate processes: a"
                " camera-enabled app is not a drop-in replacement for a plain one, because"
                " test_simulation_context.py::test_headless_mode asserts that offscreen rendering"
                " is off."
            )
        return _app

    from isaaclab.utils import has_kit

    if has_kit():
        raise RuntimeError(
            "Kit is already running but was not started by this plugin, so its launch"
            " configuration is unknown. Another test file in this process constructs AppLauncher"
            " itself; mark that file `kit_solo` so it keeps a process of its own."
        )

    from isaaclab.app import AppLauncher

    from .utils import resolve_test_sim_device

    _app = AppLauncher(headless=True, enable_cameras=cameras, device=resolve_test_sim_device()).app
    _cameras = cameras
    return _app


def _marker_for(cameras: bool) -> str:
    """Return the marker name that asks for the given ``enable_cameras`` setting."""
    return next(name for name, wants in KIT_MARKERS.items() if wants == cameras)


"""
Pytest plugin.
"""


def pytest_collectstart(collector: pytest.Collector) -> None:
    """Boot Kit before pytest imports a module that declares it needs one.

    ``Module.collect()`` is what imports the module, and this hook runs immediately before it.
    That is the last point at which the app can still be started early enough for the module's
    own ``pxr`` / ``omni`` imports to succeed.
    """
    if isinstance(collector, pytest.Module):
        marker = kit_marker_of_file(collector.path)
        if marker is not None:
            _launch(cameras=KIT_MARKERS[marker])


@pytest.fixture(scope="session")
def kit_app() -> SimulationApp:
    """The Kit app shared by every launch-marked module in this pytest process.

    Returns:
        The running ``SimulationApp``.

    Raises:
        RuntimeError: If the requesting module declares no launch marker, so nothing booted
            an app for it.
    """
    if _app is None:
        raise RuntimeError(
            "the `kit_app` fixture was requested but no Kit app is running: add one of"
            f" {', '.join(sorted(KIT_MARKERS))} to the test module's `pytestmark`."
        )
    return _app
