# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 joint-position Reach on MDP2."""

import math

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.reach.reach_env_cfg import ReachSceneCfg

from isaaclab_assets import UR10_CFG

from .reach_env_cfg import REACH_TASK, ReachMDP2EnvCfg, joint_limits, joint_state, joints, pose_tracking, robot

UR10_TASK = replace(
    REACH_TASK,
    actions={
        "arm_action": mdp2.DirectAction(
            robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos, subset=joints
        ),
    },
    commands={
        # The UR10 end effector is along its x axis.
        "ee_pose": mdp2.Command(
            mdp.uniform_pose_command,
            resampling_time_range=(4.0, 4.0),
            params={
                "pose_range": (
                    (0.35, 0.65),
                    (-0.2, 0.2),
                    (0.15, 0.5),
                    (0.0, 0.0),
                    (math.pi / 2, math.pi / 2),
                    (-3.14, 3.14),
                ),
                "make_quat_unique": False,
            },
        ),
    },
    quantities={
        "position_error": mdp2.Quantity(
            mdp.position_command_error, inputs={**pose_tracking, "body_ids": robot.bodies("ee_link")}, log=True
        ),
        "orientation_error": mdp2.Quantity(
            mdp.orientation_command_error, inputs={**pose_tracking, "body_ids": robot.bodies("ee_link")}, log=True
        ),
    },
    rewards={
        **REACH_TASK.rewards,
        "joint_vel": mdp2.Reward(
            mdp.joint_vel_l2, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}, weight=-0.0001
        ),
    },
    events={
        "reset_robot_joints": mdp2.Event(
            mdp.reset_joints_by_scale,
            inputs={**joint_limits, "joint_ids": joints},
            outputs=joint_state,
            params={"position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
        ),
    },
)
"""The MDP of the UR10 Reach task."""


@configclass
class UR10ReachMDP2EnvCfg(ReachMDP2EnvCfg):
    """UR10 joint-position Reach on MJWarp."""

    scene: ReachSceneCfg = ReachSceneCfg(
        num_envs=4096, env_spacing=2.5, robot=replace(UR10_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    task: mdp2.TaskCfg = UR10_TASK
