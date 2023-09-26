# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for creating prims in Omniverse.

Usage:
    .. code-block:: python

        import omni.isaac.orbit.sim as sim_utils
        from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

        # spawn from USD file
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
        prim_path = "/World/myAsset"

        # Option 1: spawn using the function from the module
        sim_utils.spawn_from_usd(prim_path, cfg)

        # Option 2: use the `func` reference in the config class
        cfg.func(prim_path, cfg)
"""

from __future__ import annotations

from .from_files import *  # noqa: F401, F403
from .lights import *  # noqa: F401, F403
from .materials import *  # noqa: F401, F403
from .sensors import *  # noqa: F401, F403
from .shapes import *  # noqa: F401, F403
from .spawner_cfg import *  # noqa: F401, F403
