# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .articulation import Articulation, ArticulationData
    from .rigid_object import RigidObject, RigidObjectData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("articulation", ["Articulation", "ArticulationData"]),
    ("rigid_object", ["RigidObject", "RigidObjectData"]),
)
