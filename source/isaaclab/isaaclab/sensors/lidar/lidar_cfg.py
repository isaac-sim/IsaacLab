# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg


@configclass
class LidarCfg(SensorBaseCfg):
    """Base configuration for a rendered lidar sensor."""

    @configclass
    class OffsetCfg:
        """Offset of the lidar frame from its parent prim."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation relative to the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` relative to the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

    offset: OffsetCfg = OffsetCfg()
    """Pose of the lidar frame relative to its parent. Defaults to identity."""

    spawn: SpawnerCfg | None = None
    """Optional spawner for the lidar prim."""

    renderer_cfg: RendererCfg = MISSING
    """Renderer used to capture this sensor."""
