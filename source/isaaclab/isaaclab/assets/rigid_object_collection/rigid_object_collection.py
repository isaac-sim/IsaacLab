# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_rigid_object_collection import BaseRigidObjectCollection
from .base_rigid_object_collection_data import BaseRigidObjectCollectionData

if TYPE_CHECKING:
    from isaaclab_physx.assets.rigid_object_collection import RigidObjectCollection as PhysXRigidObjectCollection
    from isaaclab_physx.assets.rigid_object_collection import (
        RigidObjectCollectionData as PhysXRigidObjectCollectionData,
    )


class RigidObjectCollection(FactoryBase, BaseRigidObjectCollection):
    """Factory for creating rigid object collection instances."""

    data: BaseRigidObjectCollectionData | PhysXRigidObjectCollectionData

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectCollection | PhysXRigidObjectCollection:
        """Create a new instance of a rigid object collection based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific rigid object collection class,
        # which is guaranteed to be a subclass of `BaseRigidObjectCollection` by convention.
        return super().__new__(cls, *args, **kwargs)
