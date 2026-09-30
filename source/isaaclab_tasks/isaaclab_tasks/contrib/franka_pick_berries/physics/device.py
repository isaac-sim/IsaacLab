# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp device selection for the MPM solver and everything that draws its state."""

import functools

import warp as wp


def resolve_mpm_device(cfg, sim_device) -> str:
    """Warp device of the MPM solver: ``cfg.mpm_device``, else the simulation device when CUDA, else ``cuda:0``.

    The robot runs on the CPU MuJoCo backend by default, so the simulation device alone cannot host the solver.
    """
    if cfg.mpm_device is not None:
        return cfg.mpm_device
    device = str(sim_device)
    return device if device.startswith("cuda") else "cuda:0"


def on_device(attribute: str = "mpm_device"):
    """Run a method with ``getattr(self, attribute)`` as the Warp device, instead of changing the process default.

    A scope is needed because the robot simulation, and therefore Warp's default device, is on the CPU.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with wp.ScopedDevice(getattr(self, attribute)):
                return method(self, *args, **kwargs)

        return wrapper

    return decorator
