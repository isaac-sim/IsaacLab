# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:
* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0

Reference:
* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
"""


import math
from scipy.spatial.transform import Rotation

from omni.isaac.orbit.actuators.config.anydrive import ANYMAL_C_DEFAULT_GROUP_CFG
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..legged_robot import LeggedRobotCfg

__all__ = ["ANYMAL_B_CFG", "ANYMAL_C_CFG"]

##
# Helper functions
##


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


##
# Configuration
##

_ANYMAL_B_INSTANCEABLE_USD = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmalB/anymal_b_instanceable.usd"
_ANYMAL_C_INSTANCEABLE_USD = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmalC/anymal_c_minimal_instanceable.usd"


ANYMAL_B_CFG = LeggedRobotCfg(
    meta_info=LeggedRobotCfg.MetaInfoCfg(usd_path=_ANYMAL_B_INSTANCEABLE_USD, soft_dof_pos_limit_factor=0.95),
    feet_info={
        "LF_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="LF_SHANK",
            pos_offset=(0.1, -0.02, -0.3215),
        ),
        "RF_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="RF_SHANK",
            pos_offset=(0.1, 0.02, -0.3215),
        ),
        "LH_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="LH_SHANK",
            pos_offset=(-0.1, -0.02, -0.3215),
        ),
        "RH_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="RH_SHANK",
            pos_offset=(-0.1, 0.02, -0.3215),
        ),
    },
    init_state=LeggedRobotCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
        dof_vel={".*": 0.0},
    ),
    rigid_props=LeggedRobotCfg.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    collision_props=LeggedRobotCfg.CollisionPropertiesCfg(
        contact_offset=0.02,
        rest_offset=0.0,
    ),
    articulation_props=LeggedRobotCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
    ),
    actuator_groups={"base_legs": ANYMAL_C_DEFAULT_GROUP_CFG},
)
"""Configuration of ANYmal-B robot using actuator-net."""

ANYMAL_C_CFG = LeggedRobotCfg(
    meta_info=LeggedRobotCfg.MetaInfoCfg(usd_path=_ANYMAL_C_INSTANCEABLE_USD, soft_dof_pos_limit_factor=0.95),
    feet_info={
        "LF_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="LF_SHANK",
            pos_offset=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
            rot_offset=quat_from_euler_rpy(0, 0, -math.pi / 2),
        ),
        "RF_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="RF_SHANK",
            pos_offset=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
            rot_offset=quat_from_euler_rpy(0, 0, math.pi / 2),
        ),
        "LH_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="LH_SHANK",
            pos_offset=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(-0.08795, 0.01305, -0.33797)),
            rot_offset=quat_from_euler_rpy(0, 0, -math.pi / 2),
        ),
        "RH_FOOT": LeggedRobotCfg.FootFrameCfg(
            body_name="RH_SHANK",
            pos_offset=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(-0.08795, -0.01305, -0.33797)),
            rot_offset=quat_from_euler_rpy(0, 0, math.pi / 2),
        ),
    },
    init_state=LeggedRobotCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
        dof_vel={".*": 0.0},
    ),
    rigid_props=LeggedRobotCfg.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    collision_props=LeggedRobotCfg.CollisionPropertiesCfg(
        contact_offset=0.02,
        rest_offset=0.0,
    ),
    articulation_props=LeggedRobotCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
    ),
    actuator_groups={"base_legs": ANYMAL_C_DEFAULT_GROUP_CFG},
)
"""Configuration of ANYmal-C robot using actuator-net."""
