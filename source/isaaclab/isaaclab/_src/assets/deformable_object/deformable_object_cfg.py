# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab._src.assets.asset_base_cfg import AssetBaseCfg
from isaaclab._src.markers import VisualizationMarkersCfg
from isaaclab._src.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from isaaclab._src.utils.configclass import configclass

if TYPE_CHECKING:
    from .deformable_object import DeformableObject


@configclass
class DeformableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a deformable object."""

    class_type: type[DeformableObject] | str = "{DIR}.deformable_object:DeformableObject"

    visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/DeformableTarget"
    )
    """The configuration object for the visualization markers. Defaults to DEFORMABLE_TARGET_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
