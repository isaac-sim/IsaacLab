# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-origin support for the asset-rich conveyor playback variant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from .conveyor_franka_asset_env_cfg import _ElevatedGroundPlaneCfg


class ConveyorFrankaGroundPlaneTerrainImporter(TerrainImporter):
    """Plane terrain whose environment origins follow an elevated workspace."""

    def __init__(self, cfg: _ElevatedGroundPlaneCfg):
        """Create the ground plane and translate its policy-facing origins."""
        super().__init__(cfg)
        self.env_origins.add_(self.env_origins.new_tensor(cfg.workspace_origin_offset))
