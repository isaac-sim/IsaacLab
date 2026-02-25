# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.string import DeferredClass


@configclass
class DeformableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a deformable object."""

    class_type: type | DeferredClass = DeferredClass(
        "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject"
    )

    visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/DeformableTarget"
    )
    """The configuration object for the visualization markers. Defaults to DEFORMABLE_TARGET_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
