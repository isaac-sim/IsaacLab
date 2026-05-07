# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_ray_caster import BaseRayCaster
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.ray_caster import RayCaster as NewtonRayCaster
    from isaaclab_physx.sensors.ray_caster import RayCaster as PhysXRayCaster


class RayCaster(FactoryBase, BaseRayCaster):
    """Factory for creating ray-caster sensor instances."""

    data: RayCasterData

    def __new__(cls, *args, **kwargs) -> BaseRayCaster | NewtonRayCaster | PhysXRayCaster:
        """Create a new instance of a ray-caster based on the backend."""
        return super().__new__(cls, *args, **kwargs)
