# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.ray_caster import MultiMeshRayCaster as NewtonMultiMeshRayCaster
    from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster as PhysXMultiMeshRayCaster


class MultiMeshRayCaster(FactoryBase, BaseMultiMeshRayCaster):
    """Factory for creating multi-mesh ray-caster sensor instances."""

    data: MultiMeshRayCasterData

    def __new__(cls, *args, **kwargs) -> BaseMultiMeshRayCaster | NewtonMultiMeshRayCaster | PhysXMultiMeshRayCaster:
        """Create a new instance of a multi-mesh ray-caster based on the backend."""
        return super().__new__(cls, *args, **kwargs)
