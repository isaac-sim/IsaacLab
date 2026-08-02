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
        RuntimeError: If a camera-enabled app is requested but Kit is already running in
            this process without cameras, or if Kit was started by something other than
            this function. Both mean the test files sharing this process do not share a
            launch configuration and must be split across processes.
    """
    global _app, _cameras

    if _app is not None:
        if cameras and not _cameras:
            raise RuntimeError(
                "launch_kit(cameras=True) was called, but Kit is already running in this process"
                " without cameras. Camera extensions cannot be enabled after startup. Mark this"
                " file `pytest.mark.kit_cameras` so it is grouped with other camera tests instead"
                " of with plain `pytest.mark.kit` files."
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
