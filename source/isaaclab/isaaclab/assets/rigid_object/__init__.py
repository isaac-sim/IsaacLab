# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid object assets."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .base_rigid_object import BaseRigidObject
    from .base_rigid_object_data import BaseRigidObjectData
    from .rigid_object import RigidObject
    from .rigid_object_cfg import RigidObjectCfg
    from .rigid_object_data import RigidObjectData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("base_rigid_object", "BaseRigidObject"),
    ("base_rigid_object_data", "BaseRigidObjectData"),
    ("rigid_object", "RigidObject"),
    ("rigid_object_cfg", "RigidObjectCfg"),
    ("rigid_object_data", "RigidObjectData"),
)
