# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.ray_caster import MultiMeshRayCasterCamera as NewtonMultiMeshRayCasterCamera
    from isaaclab_physx.sensors.ray_caster import MultiMeshRayCasterCamera as PhysXMultiMeshRayCasterCamera


class MultiMeshRayCasterCamera(FactoryBase, BaseMultiMeshRayCasterCamera):
    """Factory for creating multi-mesh ray-caster camera sensor instances."""

    data: MultiMeshRayCasterCameraData

    def __new__(
        cls, *args, **kwargs
    ) -> BaseMultiMeshRayCasterCamera | NewtonMultiMeshRayCasterCamera | PhysXMultiMeshRayCasterCamera:
        """Create a new instance of a multi-mesh ray-caster camera based on the backend."""
        return super().__new__(cls, *args, **kwargs)
