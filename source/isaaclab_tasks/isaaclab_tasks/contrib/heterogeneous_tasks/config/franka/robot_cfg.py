# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot of the heterogeneous tasks: a Franka arm with a Robotiq 2F-85 gripper."""

import math

from isaaclab_isaacsim.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyCfg, NewtonArticulationCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

FRANKA_ROBOTIQ_CFG = clone(FRANKA_ROBOTIQ_GRIPPER_CFG)
FRANKA_ROBOTIQ_CFG.prim_path = "{ENV_REGEX_NS}/robot"
# TODO: Temporary location. Replace the Hugging Face URL with the Isaac Lab Nucleus path once the asset is uploaded
# there. The asset is the Isaac Sim Franka with the Isaac Lab Franka arm inertias; its Robotiq gripper variant sits on
# the flange ``panda_link8`` with rubber finger pads.
FRANKA_ROBOTIQ_CFG.spawn.usd_path = (
    "https://huggingface.co/datasets/LiFanxing/franka_robotiq/resolve/47163169a7decfe1131d3734b51007f74fe0f594"
    "/franka.usd"
)
# PhysX switches gravity off per body, while Newton reads gravity switches only on the physics scene, so MuJoCo Warp
# compensates gravity on the bodies instead.
FRANKA_ROBOTIQ_CFG.spawn.rigid_props = [
    PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
    MujocoRigidBodyCfg(gravcomp=1.0),
]
# Effort limits [N*m] of the Franka arm joints.
FRANKA_ROBOTIQ_CFG.actuators["panda_shoulder"].joint_effort_limit = 87.0
FRANKA_ROBOTIQ_CFG.actuators["panda_forearm"].joint_effort_limit = 12.0
FRANKA_ROBOTIQ_CFG.spawn.activate_contact_sensors = True
FRANKA_ROBOTIQ_CFG.spawn.articulation_props = [
    PhysxArticulationCfg(
        enabled_self_collisions=False, solver_position_iteration_count=64, solver_velocity_iteration_count=0
    ),
    NewtonArticulationCfg(self_collision_enabled=False),
]
# MuJoCo Warp applies the gravity compensation through the joint actuators, within their effort limits.
FRANKA_ROBOTIQ_CFG.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
FRANKA_ROBOTIQ_CFG.init_state = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": -math.pi / 5,
        "panda_joint3": 0.0,
        "panda_joint4": -4 * math.pi / 5,
        "panda_joint5": 0.0,
        "panda_joint6": 3 * math.pi / 5,
        "panda_joint7": 0.0,
        "finger_joint": 0.0,
        "right_outer.*": 0.0,
        "left_inner.*": 0.0,
        "right_inner.*": 0.0,
    },
)
for actuator in ("panda_shoulder", "panda_forearm"):
    FRANKA_ROBOTIQ_CFG.actuators[actuator].stiffness = 400.0
    FRANKA_ROBOTIQ_CFG.actuators[actuator].damping = 80.0
    FRANKA_ROBOTIQ_CFG.actuators[actuator].armature = 0.1
"""Franka arm with a Robotiq 2F-85 gripper, at the environment origin with the start pose and arm gains of the RoboLab
DROID setup."""
