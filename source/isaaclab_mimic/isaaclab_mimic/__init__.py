# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Package containing implementation of Isaac Lab Mimic data generation."""

# ---------------------------------------------------------------------------
# Compatibility shim: re-expose ``warp.torch`` after warp-lang 1.13 dropped it
#
# Newton ``v1.2.0rc2`` requires ``warp-lang>=1.13``. Warp 1.13 collapsed the
# ``warp.torch`` submodule into the top-level ``warp`` namespace, so e.g.
# ``wp.torch.device_from_torch`` is now ``wp.device_from_torch``. cuRobo
# (NVlabs/curobo) still uses the old ``wp.torch.*`` form (verified at
# ``ebb71702f`` and on ``main`` as of 2026-05-07) and raises
# ``AttributeError: module 'warp' has no attribute 'torch'`` at
# ``MotionGenConfig.load_from_robot_config(...)`` time.
#
# This shim runs at ``isaaclab_mimic`` import — which Python evaluates before
# any submodule, including
# :mod:`isaaclab_mimic.motion_planners.curobo.curobo_planner` — so curobo
# sees a ``warp.torch`` namespace whose members forward to the relocated
# top-level ``warp.*`` callables. Idempotent: a no-op once warp ships
# ``wp.torch`` again or curobo migrates.
#
# TODO: remove this shim once the cuRobo pin in ``docker/Dockerfile.curobo``
# bumps to a commit that uses ``wp.from_torch``/``wp.device_from_torch``/
# etc. directly. Tracking upstream at https://github.com/NVlabs/curobo —
# follow up on the open issue / PR there to confirm the migration landed
# before deleting this block.
import sys as _sys
import types as _types

import warp as _wp

if not hasattr(_wp, "torch"):
    _wp_torch_shim = _types.ModuleType("warp.torch")
    for _name in (
        "from_torch",
        "to_torch",
        "device_from_torch",
        "device_to_torch",
        "dtype_from_torch",
        "dtype_to_torch",
        "stream_from_torch",
        "stream_to_torch",
    ):
        if hasattr(_wp, _name):
            setattr(_wp_torch_shim, _name, getattr(_wp, _name))
    _wp.torch = _wp_torch_shim
    _sys.modules["warp.torch"] = _wp_torch_shim
    del _wp_torch_shim, _name

del _sys, _types, _wp


import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_mimic")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
