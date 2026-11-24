# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cassie velocity tracking on flat ground."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg

from isaaclab_assets.robots.cassie import CASSIE_CFG

from ..velocity_flat import (
    FLAT_TERRAIN_SCENE,
    FlatVelocityEnvCfg,
    contact_forces,
    flat_velocity_sim,
    flat_velocity_task,
    joints,
    robot,
    root_state,
    scene,
    velocity_tracking,
)

toes = contact_forces.bodies(".*toe")
toe_joints = robot.joints("toe_joint_.*")
feet_command = {"command": mdp2.commands.base_velocity, "body_ids": toes}

_FLAT_TASK = flat_velocity_task("pelvis", action_rate_weight=-0.015)

TASK = replace(
    _FLAT_TASK,
    actions={
        "joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos),
    },
    rewards={
        "track_lin_vel_xy_exp": mdp2.Reward(
            mdp.track_lin_vel_xy_exp, inputs=velocity_tracking, params={"std": 0.5}, weight=2.0
        ),
        "track_ang_vel_z_exp": mdp2.Reward(
            mdp.track_ang_vel_z_exp, inputs=velocity_tracking, params={"std": 0.5}, weight=1.0
        ),
        "lin_vel_z_l2": mdp2.Reward(mdp.lin_vel_z_l2, inputs=root_state, weight=-2.0),
        "ang_vel_xy_l2": mdp2.Reward(mdp.ang_vel_xy_l2, inputs=root_state, weight=-0.05),
        "dof_torques_l2": mdp2.Reward(
            mdp.joint_torques_l2, inputs={"applied_effort": robot.applied_effort, "joint_ids": joints}, weight=-5.0e-6
        ),
        "dof_acc_l2": mdp2.Reward(
            mdp.joint_acc_l2, inputs={"joint_acc": robot.joint_acc, "joint_ids": joints}, weight=-3.75e-7
        ),
        "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.015),
        "feet_air_time": mdp2.Reward(
            mdp.feet_air_time_positive_biped,
            inputs={
                **feet_command,
                "current_air_time": contact_forces.current_air_time,
                "current_contact_time": contact_forces.current_contact_time,
            },
            params={"threshold": 0.3, "overstay_scale": 0.0},
            weight=5.0,
        ),
        "flat_orientation_l2": mdp2.Reward(
            mdp.flat_orientation_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
            weight=-2.5,
        ),
        "air_time_variance": mdp2.Reward(
            mdp.feet_air_time_variance,
            inputs={
                **feet_command,
                "last_air_time": contact_forces.last_air_time,
                "last_contact_time": contact_forces.last_contact_time,
            },
            params={"max_time": 0.5},
            weight=-10.0,
        ),
        "termination_penalty": mdp2.Reward(mdp.is_terminated, inputs={"terminated": mdp2.terminated}, weight=-200.0),
        "joint_deviation_hip": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={
                "joint_pos": robot.joint_pos,
                "default_joint_pos": robot.default_joint_pos,
                "joint_ids": robot.joints("hip_rotation_.*"),
            },
            weight=-0.2,
        ),
        "joint_deviation_toes": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={
                "joint_pos": robot.joint_pos,
                "default_joint_pos": robot.default_joint_pos,
                "joint_ids": toe_joints,
            },
            weight=-0.2,
        ),
        "dof_pos_limits": mdp2.Reward(
            mdp.joint_pos_limits,
            inputs={
                "joint_pos": robot.joint_pos,
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                "joint_ids": toe_joints,
            },
            weight=-1.0,
        ),
    },
    events={
        **_FLAT_TASK.events,
        "add_base_mass": replace(
            _FLAT_TASK.events["add_base_mass"],
            params={"mass_distribution_params": (1.0, 1.25), "operation": "scale", "distribution": "log_uniform"},
        ),
        "base_com": None,
        "reset_robot_joints": replace(
            _FLAT_TASK.events["reset_robot_joints"], params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
        ),
    },
)
"""The MDP of the task, adapted to Cassie's biped gait and reset."""


@configclass
class CassieFlatMDP2EnvCfg(FlatVelocityEnvCfg):
    """Cassie's flat velocity task on MJWarp."""

    sim: SimulationCfg = flat_velocity_sim(njmax=52, nconmax=15)
    scene: MySceneCfg = replace(FLAT_TERRAIN_SCENE, robot=replace(CASSIE_CFG, prim_path="{ENV_REGEX_NS}/Robot"))
    task: mdp2.TaskCfg = TASK
