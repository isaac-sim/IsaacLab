# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start the simulation runtime for a test module."""

from __future__ import annotations

import contextlib

from .devices import resolve_test_sim_device

_RUNTIME = contextlib.ExitStack()


def launch_test_simulation(cfg=None, **launcher_args) -> None:
    """Start the runtime *cfg* needs for the rest of the test process through :func:`~isaaclab.app.launch_simulation`.

    Call it at module level, before importing modules that need the runtime. The runtime stays up
    until the process exits, when its launcher closes it with the process's exit status.

    Args:
        cfg: Config tree whose physics, renderers, and sensors select the runtime. Defaults to
            :class:`~isaaclab.sim.SimulationCfg`, whose default physics backend selects the runtime.
        **launcher_args: Launcher arguments, for example ``device`` or ``enable_cameras``. ``device``
            defaults to :func:`~isaaclab.test.utils.resolve_test_sim_device`.
    """
    # sim_launcher loads the backend configs (~1 s); keep this module cheap for tests that only use test_devices
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    if "device" not in launcher_args:
        launcher_args["device"] = resolve_test_sim_device()
    cfg = SimulationCfg() if cfg is None else cfg
    _RUNTIME.enter_context(launch_simulation(cfg, launcher_args))
