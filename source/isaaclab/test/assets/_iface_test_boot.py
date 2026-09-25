# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kit and kitless bootstrap for backend interface tests."""

import os
import sys
from unittest.mock import MagicMock

_kitless = "ovphysx" in os.environ.get("LD_PRELOAD", "") or (
    os.environ.get("LD_PRELOAD", "") == "" and "EXP_PATH" not in os.environ
)

if not _kitless:
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
else:
    simulation_app = None
    # ``omni`` is a real namespace package in OvPhysX kitless runs. Install
    # missing submodules in both ``sys.modules`` and the namespace attributes.
    import omni as _omni

    for _mod in ("physics", "physics.tensors", "physx", "timeline", "usd"):
        _stub = MagicMock()
        sys.modules[f"omni.{_mod}"] = _stub
        setattr(_omni, _mod.split(".", 1)[0], _stub)
    for _mod in ("isaacsim.core", "isaacsim.core.simulation_manager"):
        sys.modules.setdefault(_mod, MagicMock())
