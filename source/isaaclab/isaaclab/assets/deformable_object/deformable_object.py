# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_deformable_object import BaseDeformableObject
from .base_deformable_object_data import BaseDeformableObjectData

if TYPE_CHECKING:
    from isaaclab_physx.assets.deformable_object import DeformableObject as PhysXDeformableObject
    from isaaclab_physx.assets.deformable_object import DeformableObjectData as PhysXDeformableObjectData


class DeformableObject(FactoryBase, BaseDeformableObject):
    """Factory for creating deformable object instances."""

    data: BaseDeformableObjectData | PhysXDeformableObjectData

    def __new__(cls, *args, **kwargs) -> BaseDeformableObject | PhysXDeformableObject:
        """Create a new instance of a deformable object based on the backend."""
        return super().__new__(cls, *args, **kwargs)
