# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton BVH ray-cast sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .newton_raycast_sensor import NewtonRaycastSensor


@configclass
class NewtonRaycastSensorCfg(RayCasterCfg):
    """Configuration for the Newton BVH ray-cast sensor.

    Unlike :class:`~isaaclab.sensors.RayCasterCfg`, no target meshes are
    configured: rays are cast against every collision shape in the Newton
    scene through the model's shape BVH, including dynamic bodies.
    """

    class_type: type[NewtonRaycastSensor] | str = "{DIR}.newton_raycast_sensor:NewtonRaycastSensor"

    mesh_prim_paths: list[str] = []
    """Unused. Rays hit every shape in the Newton scene BVH."""
