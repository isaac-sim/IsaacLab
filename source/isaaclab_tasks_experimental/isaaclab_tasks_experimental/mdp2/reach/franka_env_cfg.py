# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka joint-position Reach on MDP2."""

import math

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp
from isaaclab_isaacsim.sim.schemas import PhysxRigidBodyCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.reach.reach_env_cfg import ReachSceneCfg

from isaaclab_assets import FRANKA_PANDA_CFG

from .reach_env_cfg import REACH_TASK, ReachMDP2EnvCfg, joint_limits, joint_state, pose_tracking, robot

franka_arm = robot.joints("panda_joint.*")

FRANKA_ROBOT_CFG = replace(
    FRANKA_PANDA_CFG,
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=replace(
        FRANKA_PANDA_CFG.spawn,
        variants={"Physics": "mujoco", "Colliders": "primitives"},
        rigid_props=[
            PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            MujocoRigidBodyCfg(gravcomp=None),
        ],
    ),
    actuators={
        **FRANKA_PANDA_CFG.actuators,
        # Joint inertia varies substantially along the arm; damp each drive accordingly.
        "panda_arm": replace(
            FRANKA_PANDA_CFG.actuators["panda_arm"],
            damping={
                "panda_joint1": 40.0,
                "panda_joint2": 60.0,
                "panda_joint3": 35.0,
                "panda_joint4": 60.0,
                "panda_joint[5-6]": 15.0,
                "panda_joint7": 12.0,
            },
        ),
    },
)
"""The Franka Panda with its MuJoCo physics variant and joint-wise arm damping."""

FRANKA_TASK = replace(
    REACH_TASK,
    actions={
        "arm_action": mdp2.DirectAction(
            robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos, subset=franka_arm
        ),
    },
    commands={
        # The Franka end effector is along its z axis.
        "ee_pose": mdp2.Command(
            mdp.uniform_pose_command,
            resampling_time_range=(4.0, 4.0),
            params={
                "pose_range": ((0.35, 0.65), (-0.2, 0.2), (0.15, 0.5), (0.0, 0.0), (math.pi, math.pi), (-3.14, 3.14)),
                "make_quat_unique": False,
            },
        ),
    },
    quantities={
        "position_error": mdp2.Quantity(
            mdp.position_command_error, inputs={**pose_tracking, "body_ids": robot.bodies("panda_hand")}, log=True
        ),
        "orientation_error": mdp2.Quantity(
            mdp.orientation_command_error, inputs={**pose_tracking, "body_ids": robot.bodies("panda_hand")}, log=True
        ),
    },
    rewards={
        **REACH_TASK.rewards,
        "joint_vel": mdp2.Reward(
            mdp.joint_vel_l2, inputs={"joint_vel": robot.joint_vel, "joint_ids": franka_arm}, weight=-0.0001
        ),
    },
    events={
        # Offsets bound the arm away from its joint limits, and the mimic fingers start together.
        "reset_robot_joints": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={**joint_limits, "joint_ids": franka_arm},
            outputs=joint_state,
            params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
        ),
        "reset_robot_fingers": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={**joint_limits, "joint_ids": robot.joints("panda_finger_joint.*")},
            outputs=joint_state,
            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
        ),
    },
)
"""The MDP of the Franka Reach task."""


@configclass
class FrankaReachMDP2EnvCfg(ReachMDP2EnvCfg):
    """Franka joint-position Reach on MJWarp."""

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5, robot=FRANKA_ROBOT_CFG)
    task: mdp2.TaskCfg = FRANKA_TASK
