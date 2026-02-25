# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast camera sensor."""

import logging

from isaaclab.utils import DeferredClass, configclass

from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from .ray_caster_camera_cfg import RayCasterCameraCfg

# import logger
logger = logging.getLogger(__name__)


@configclass
class MultiMeshRayCasterCameraCfg(RayCasterCameraCfg, MultiMeshRayCasterCfg):
    """Configuration for the multi-mesh ray-cast camera sensor."""

    class_type: type | DeferredClass = DeferredClass("isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera:MultiMeshRayCasterCamera")

    def __post_init__(self):
        super().__post_init__()

        # Camera only supports 'base' ray alignment. Ensure this is set correctly.
        if self.ray_alignment != "base":
            logger.warning(
                "Ray alignment for MultiMeshRayCasterCameraCfg only supports 'base' alignment. Overriding from"
                f"'{self.ray_alignment}' to 'base'."
            )
            self.ray_alignment = "base"
