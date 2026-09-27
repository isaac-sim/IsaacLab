# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start the simulation runtime for a test module."""

from __future__ import annotations

import contextlib

from .devices import resolve_test_sim_device

_RUNTIME = contextlib.ExitStack()


def launch_test_simulation(**launcher_args):
    """Start Isaac Sim / Kit for the rest of the test process through :func:`~isaaclab.app.launch_simulation`.

    Call it at module level, before importing modules that need Kit. The runtime stays up until the
    process exits, when the Kit launcher closes it with the process's exit status.

    Args:
        **launcher_args: Launcher arguments, for example ``device`` or ``enable_cameras``. ``device``
            defaults to :func:`~isaaclab.test.utils.resolve_test_sim_device`.

    Returns:
        The running Kit application, for tests that pump it with ``update()``.
    """
    from isaaclab.app import launch_simulation

    if "device" not in launcher_args:
        launcher_args["device"] = resolve_test_sim_device()
    _RUNTIME.enter_context(launch_simulation(None, {"require_kit": True, "headless": True, **launcher_args}))
    import omni.kit.app

    return omni.kit.app.get_app()
