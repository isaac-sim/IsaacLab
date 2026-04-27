# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn assets from files or in-memory mesh data.

Currently, the following spawners are supported:

* :class:`UsdFileCfg`: Spawn an asset from a USD file.
* :class:`UrdfFileCfg`: Spawn an asset from a URDF file.
* :class:`MeshFileCfg`: Spawn a mesh file path or in-memory mesh data.
* :class:`GroundPlaneCfg`: Spawn a ground plane using the grid-world USD file.

"""

from isaaclab.utils.module import lazy_export

lazy_export()
