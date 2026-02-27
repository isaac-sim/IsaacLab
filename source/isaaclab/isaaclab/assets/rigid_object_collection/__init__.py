# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid object collection."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .base_rigid_object_collection import BaseRigidObjectCollection
    from .base_rigid_object_collection_data import BaseRigidObjectCollectionData
    from .rigid_object_collection import RigidObjectCollection
    from .rigid_object_collection_cfg import RigidObjectCollectionCfg
    from .rigid_object_collection_data import RigidObjectCollectionData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("base_rigid_object_collection", "BaseRigidObjectCollection"),
    ("base_rigid_object_collection_data", "BaseRigidObjectCollectionData"),
    ("rigid_object_collection", "RigidObjectCollection"),
    ("rigid_object_collection_cfg", "RigidObjectCollectionCfg"),
    ("rigid_object_collection_data", "RigidObjectCollectionData"),
)
