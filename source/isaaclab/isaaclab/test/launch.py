# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kit launch helper for Isaac Lab tests.

Test modules that need Isaac Sim call :func:`launch_kit` at module scope in place of
constructing :class:`~isaaclab.app.AppLauncher` directly::

    from isaaclab.test.launch import launch_kit

    launch_kit()  # or launch_kit(cameras=True)

The call must stay at module scope: a test module's own imports (``pxr``, ``omni``,
``isaaclab_physx``, ...) run during pytest collection, before any fixture executes, so Kit
must already be running by then.

:func:`launch_kit` is idempotent within a process. The first test module to call it boots
Kit; every later module gets the running app back. A pytest process covering several test
files therefore pays Kit startup once rather than once per file.

Declare the matching marker on the module so the test runner can group files that share a
launch configuration into one process::

    pytestmark = pytest.mark.kit  # launch_kit()
    pytestmark = pytest.mark.kit_cameras  # launch_kit(cameras=True)

The two groups cannot be merged. Cameras cannot be enabled after startup, so a plain ``kit``
file cannot run in a process a ``kit_cameras`` file will later join; and a camera-enabled app
is not a drop-in replacement for a plain one either, because some tests assert that offscreen
rendering is off. :func:`launch_kit` therefore raises on any mismatch rather than handing back
an app whose configuration is not the one the caller asked for.
"""

from __future__ import annotations

from typing import Any

_app: Any = None
"""The Kit application booted by :func:`launch_kit`, or None before the first call."""

_cameras: bool = False
"""Whether :attr:`_app` was booted with camera and render extensions enabled."""


def launch_kit(*, cameras: bool = False) -> Any:
    """Boot the shared Kit app for this process, or return the one already running.

    Args:
        cameras: Whether the app must be booted with camera and render extensions enabled.
            Passed through to :paramref:`~isaaclab.app.AppLauncher.enable_cameras`.

    Returns:
        The running ``SimulationApp``.

    Raises:
        RuntimeError: If the running app was booted with a different ``cameras`` setting, or if
            Kit was started by something other than this function. Both mean the test files
            sharing this process do not share a launch configuration and must be split across
            processes.
    """
    global _app, _cameras

    if _app is not None:
        if cameras != _cameras:
            wanted = "with" if cameras else "without"
            running = "with" if _cameras else "without"
            raise RuntimeError(
                f"launch_kit(cameras={cameras}) wants an app {wanted} cameras, but Kit is already"
                f" running in this process {running} them, and that cannot be changed after"
                " startup. Files marked `kit` and `kit_cameras` need separate processes. A"
                " camera-enabled app is not a drop-in replacement for a plain one:"
                " test_simulation_context.py::test_headless_mode asserts that offscreen"
                " rendering is off."
            )
        return _app

    from isaaclab.utils import has_kit

    if has_kit():
        raise RuntimeError(
            "Kit is already running but was not started by launch_kit(), so its launch"
            " configuration is unknown. Another test file in this process still constructs"
            " AppLauncher directly; run that file in its own process."
        )

    from isaaclab.app import AppLauncher

    from .utils import resolve_test_sim_device

    _app = AppLauncher(headless=True, enable_cameras=cameras, device=resolve_test_sim_device()).app
    _cameras = cameras
    return _app
