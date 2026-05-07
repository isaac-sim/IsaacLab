# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.camera import CameraData
from isaaclab.utils.backend_utils import FactoryBase

from .base_ray_caster_camera import BaseRayCasterCamera

if TYPE_CHECKING:
    from isaaclab_newton.sensors.ray_caster import RayCasterCamera as NewtonRayCasterCamera
    from isaaclab_physx.sensors.ray_caster import RayCasterCamera as PhysXRayCasterCamera


class RayCasterCamera(FactoryBase, BaseRayCasterCamera):
    """Factory for creating ray-caster camera sensor instances."""

    data: CameraData

    def __new__(cls, *args, **kwargs) -> BaseRayCasterCamera | NewtonRayCasterCamera | PhysXRayCasterCamera:
        """Create a new instance of a ray-caster camera based on the backend."""
        return super().__new__(cls, *args, **kwargs)
