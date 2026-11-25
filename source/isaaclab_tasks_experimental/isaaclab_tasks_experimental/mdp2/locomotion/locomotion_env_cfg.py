# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant and Humanoid locomotion: run toward a target far along the x axis, without falling.

The robots share :data:`LOCOMOTION_TASK`, as the direct Ant and Humanoid environments share their environment class.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from . import mdp as locomotion

robot = mdp2.Entity("robot")
joint_wrench = mdp2.Entity("joint_wrench")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root = {"root_link_pose_w": robot.root_link_pose_w}
velocity = {**root, "root_com_vel_w": robot.root_com_vel_w}
joint_pos = {"joint_pos": robot.joint_pos, "soft_joint_pos_limits": robot.soft_joint_pos_limits, "joint_ids": joints}
to_target = {**root, "env_origins": scene.env_origins}
target = {"target_x": 1000.0, "target_y": 0.0}  # Target position in each world [m]

LOCOMOTION_TASK = mdp2.TaskCfg(
    actions={"joint_effort": None},
    quantities={
        "up": mdp2.Quantity(locomotion.base_up_proj, inputs={**root, "gravity_w": scene.gravity_w}),
        "heading": mdp2.Quantity(locomotion.base_heading_proj, inputs=to_target, params=target),
        "success_rate": mdp2.Quantity(locomotion.survival, inputs={"time_out": mdp2.time_out}, log=True),
    },
    observations={
        "policy": {
            "base_height": mdp2.Observation(locomotion.base_height, inputs=root),
            "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=velocity),
            "base_ang_vel": None,
            "base_yaw_roll": mdp2.Observation(locomotion.base_yaw_roll, inputs=root),
            "base_angle_to_target": mdp2.Observation(locomotion.base_angle_to_target, inputs=to_target, params=target),
            "base_up_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.up}),
            "base_heading_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.heading}),
            "joint_pos_norm": mdp2.Observation(mdp.joint_pos_limit_normalized, inputs=joint_pos),
            "joint_vel_rel": None,
            "feet_body_forces": None,
            "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
        },
    },
    rewards={
        "progress": mdp2.Reward(
            locomotion.progress_reward, inputs=to_target, params={**target, "step_dt": mdp2.step_dt}, weight=1.0
        ),
        "alive": None,
        "upright": mdp2.Reward(
            locomotion.upright_posture_bonus, inputs={"up": mdp2.quantities.up}, params={"threshold": 0.93}, weight=0.1
        ),
        "move_to_target": mdp2.Reward(
            locomotion.move_to_target_bonus,
            inputs={"heading": mdp2.quantities.heading},
            params={"threshold": 0.8},
            weight=0.5,
        ),
        "action_l2": None,
        "energy": None,
        "joint_pos_limits": None,
        "terminating": None,
    },
    terminations={"time_out": None, "torso_height": None},
    events={
        "reset_base": mdp2.Event(
            mdp.reset_root_state_uniform,
            inputs={
                "default_root_pose": robot.default_root_pose,
                "default_root_vel": robot.default_root_vel,
                "env_origins": scene.env_origins,
            },
            outputs=velocity,
            params={"pose_range": ((0.0, 0.0),) * 6, "velocity_range": ((0.0, 0.0),) * 6},
        ),
        "reset_robot_joints": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={
                "default_joint_pos": robot.default_joint_pos,
                "default_joint_vel": robot.default_joint_vel,
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                "soft_joint_vel_limits": robot.soft_joint_vel_limits,
                "joint_ids": joints,
            },
            outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
            params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
        ),
    },
)
"""The MDP of both robots. Each robot's task sets the uses left ``None``: its actions, the scales of its velocity and
foot force observations, its reward weights, its time limit, and its minimum torso height."""
