# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot of the VLA benchmark tasks: a Franka arm with a Robotiq 2F-85 gripper."""

import math
import os

from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots import franka

# TODO: Point to the Isaac Lab Nucleus once the robot is uploaded there.
FRANKA_ROBOTIQ_ASSETS_DIR = os.environ.get(
    "FRANKA_ROBOTIQ_ASSETS_DIR", "https://huggingface.co/datasets/LiFanxing/franka_robotiq/resolve/main"
)
"""Root of the robot's complete USD file. Defaults to the franka_robotiq dataset on Hugging Face; the environment
variable of the same name overrides it, e.g. with a local copy.
"""

# TODO: Confirm whether to use the default Franka robot.
FRANKA_ROBOTIQ_GRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FRANKA_ROBOTIQ_ASSETS_DIR}/franka_robotiq_2f_85_isaaclab_inertia.usd",
        activate_contact_sensors=True,
        # Gravity compensation holds the arm, as the PhysX robot disables gravity on its links.
        rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
        joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)],
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=64, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
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
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            joint_effort_limit=87.0,
            joint_velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
            armature=0.1,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            joint_effort_limit=12.0,
            joint_velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
            armature=0.1,
        ),
        **{
            name: franka.FRANKA_ROBOTIQ_GRIPPER_CFG.actuators[name].copy()
            for name in ("gripper_drive", "gripper_finger", "gripper_passive")
        },
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka arm with a Robotiq 2F-85 gripper, as in the RoboLab DROID setup with the Isaac Lab arm inertia."""
