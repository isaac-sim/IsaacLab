# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_deformable_object_data import BaseDeformableObjectData

if TYPE_CHECKING:
    from isaaclab_physx.assets.deformable_object.deformable_object_data import (
        DeformableObjectData as PhysXDeformableObjectData,
    )


class DeformableObjectData(FactoryBase):
    """Factory for creating deformable object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseDeformableObjectData | PhysXDeformableObjectData:
        """Create a new instance of a deformable object data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
