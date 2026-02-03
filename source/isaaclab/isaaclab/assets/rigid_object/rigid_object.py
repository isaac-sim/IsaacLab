# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_rigid_object import BaseRigidObject
from .base_rigid_object_data import BaseRigidObjectData

if TYPE_CHECKING:
    from isaaclab_newton.assets.rigid_object import RigidObject as NewtonRigidObject
    from isaaclab_newton.assets.rigid_object import RigidObjectData as NewtonRigidObjectData


class RigidObject(FactoryBase, BaseRigidObject):
    """Factory for creating rigid object instances."""

    data: BaseRigidObjectData | NewtonRigidObjectData

    def __new__(cls, *args, **kwargs) -> BaseRigidObject | NewtonRigidObject:
        """Create a new instance of a rigid object based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific rigid object class,
        # which is guaranteed to be a subclass of `BaseArticulation` by convention.
        return super().__new__(cls, *args, **kwargs)
