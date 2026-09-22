# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_rigid_object_collection_data import BaseRigidObjectCollectionData

if TYPE_CHECKING:
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as PhysXRigidObjectCollectionData,
    )


class RigidObjectCollectionData(FactoryBase):
    """Factory for creating rigid object collection data instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectCollectionData | PhysXRigidObjectCollectionData:
        """Create a new instance of a rigid object collection data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
