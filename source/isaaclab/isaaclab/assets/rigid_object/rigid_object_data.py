# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_rigid_object_data import BaseRigidObjectData

if TYPE_CHECKING:
    from isaaclab_physx.assets.rigid_object.rigid_object_data import RigidObjectData as PhysXRigidObjectData


class RigidObjectData(FactoryBase):
    """Factory for creating rigid object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectData | PhysXRigidObjectData:
        """Create a new instance of a rigid object data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
