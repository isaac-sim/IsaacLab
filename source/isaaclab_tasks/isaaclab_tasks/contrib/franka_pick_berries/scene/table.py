# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a USD table as a kinematic rigid body, so the coupled solvers can see it as a collider."""

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import define_rigid_body_properties
from isaaclab.sim.spawners.from_files import spawn_from_usd


def spawn_kinematic_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a USD asset and define a kinematic rigid body on its root prim.

    The stock USD spawner only modifies rigid bodies that already exist, and the lab-table asset has none.
    """
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
    return prim
