# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid objects of the VLA benchmark tasks."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from . import VLA_BENCHMARKS_ASSETS_DIR


def library_object(
    name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    kinematic: bool = False,
) -> RigidObjectCfg:
    """Return an object of the asset package's library, spawned under the environment prim as ``name``.

    Args:
        name: Object name; the package holds the object as ``objects/<name>.usda``.
        pos: Start position in the environment frame [m].
        rot: Start orientation as a quaternion in (x, y, z, w) order.
        kinematic: Whether the object is kinematic, like the furniture that the objects stand on.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{VLA_BENCHMARKS_ASSETS_DIR}/objects/{name}.usda",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True) if kinematic else None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )
