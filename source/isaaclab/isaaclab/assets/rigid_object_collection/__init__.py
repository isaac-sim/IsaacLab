# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid object collection."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base_rigid_object_collection": ["BaseRigidObjectCollection"],
        "base_rigid_object_collection_data": ["BaseRigidObjectCollectionData"],
        "rigid_object_collection": ["RigidObjectCollection"],
        "rigid_object_collection_cfg": ["RigidObjectCollectionCfg"],
        "rigid_object_collection_data": ["RigidObjectCollectionData"],
    },
)
