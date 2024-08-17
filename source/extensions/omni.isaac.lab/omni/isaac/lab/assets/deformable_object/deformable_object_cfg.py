# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from omni.isaac.lab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .deformable_object import DeformableObject


@configclass
class DeformableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a deformable object."""

    class_type: type = DeformableObject

    visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/DeformableTarget"
    )
    """The configuration object for the visualization markers. Defaults to DEFORMABLE_TARGET_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
