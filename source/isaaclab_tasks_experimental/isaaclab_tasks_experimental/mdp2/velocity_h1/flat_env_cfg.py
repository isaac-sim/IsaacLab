# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree H1 velocity tracking on flat ground."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg

from isaaclab_assets import H1_MINIMAL_CFG

from ..velocity_flat import (
    FLAT_TERRAIN_SCENE,
    FLAT_VELOCITY_TASK,
    FlatVelocityEnvCfg,
    FlatVelocityPhysicsCfg,
    contact_forces,
    joints,
    robot,
    root_state,
    scene,
    velocity_tracking,
)

world_yaw_tracking = {"command": mdp2.commands.base_velocity, "root_com_vel_w": robot.root_com_vel_w}

_COMMAND = FLAT_VELOCITY_TASK.commands["base_velocity"]
_BASE_CONTACT = FLAT_VELOCITY_TASK.terminations["base_contact"]
_BASE_MASS = FLAT_VELOCITY_TASK.events["add_base_mass"]
_FEET = ".*ankle_link"

TASK = replace(
    FLAT_VELOCITY_TASK,
    actions={
        "joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos, subset=joints)
    },
    commands={
        "base_velocity": replace(
            _COMMAND, params={**_COMMAND.params, "lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0)}
        ),
    },
    rewards={
        "track_lin_vel_xy_exp": mdp2.Reward(
            mdp.track_lin_vel_xy_yaw_frame_exp, inputs=velocity_tracking, params={"std": 0.5}, weight=1.0
        ),
        "track_ang_vel_z_exp": mdp2.Reward(
            mdp.track_ang_vel_z_world_exp, inputs=world_yaw_tracking, params={"std": 0.5}, weight=1.0
        ),
        "ang_vel_xy_l2": mdp2.Reward(mdp.ang_vel_xy_l2, inputs=root_state, weight=-0.05),
        "dof_acc_l2": mdp2.Reward(
            mdp.joint_acc_l2, inputs={"joint_acc": robot.joint_acc, "joint_ids": joints}, weight=-1.25e-7
        ),
        "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.005),
        "feet_air_time": mdp2.Reward(
            mdp.feet_air_time_positive_biped,
            inputs={
                "body_ids": contact_forces.bodies(_FEET),
                "command": mdp2.commands.base_velocity,
                "current_air_time": contact_forces.current_air_time,
                "current_contact_time": contact_forces.current_contact_time,
            },
            params={"threshold": 0.6, "overstay_scale": 0.0},
            weight=1.0,
        ),
        "flat_orientation_l2": mdp2.Reward(
            mdp.flat_orientation_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
            weight=-1.0,
        ),
        # Penalize the ankle joint limits.
        "dof_pos_limits": mdp2.Reward(
            mdp.joint_pos_limits,
            inputs={
                "joint_pos": robot.joint_pos,
                "joint_ids": robot.joints(".*_ankle"),
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
            },
            weight=-1.0,
        ),
        "termination_penalty": mdp2.Reward(mdp.is_terminated, inputs={"terminated": mdp2.terminated}, weight=-200.0),
        "feet_slide": mdp2.Reward(
            mdp.feet_slide,
            inputs={
                "body_com_vel_w": robot.body_com_vel_w,
                "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                "body_ids": robot.bodies(_FEET),
                "sensor_ids": contact_forces.bodies(_FEET),
            },
            weight=-0.25,
        ),
        # Penalize the deviation from default of the joints that are not essential for locomotion.
        "joint_deviation_hip": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={
                "joint_pos": robot.joint_pos,
                "joint_ids": robot.joints(".*_hip_yaw", ".*_hip_roll"),
                "default_joint_pos": robot.default_joint_pos,
            },
            weight=-0.2,
        ),
        "joint_deviation_arms": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={
                "joint_pos": robot.joint_pos,
                "joint_ids": robot.joints(".*_shoulder_.*", ".*_elbow"),
                "default_joint_pos": robot.default_joint_pos,
            },
            weight=-0.2,
        ),
        "joint_deviation_torso": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={
                "joint_pos": robot.joint_pos,
                "joint_ids": robot.joints("torso"),
                "default_joint_pos": robot.default_joint_pos,
            },
            weight=-0.1,
        ),
        # Fixes the tilted gait.
        "air_time_variance": mdp2.Reward(
            mdp.feet_air_time_variance,
            inputs={
                "body_ids": contact_forces.bodies(_FEET),
                "command": mdp2.commands.base_velocity,
                "last_air_time": contact_forces.last_air_time,
                "last_contact_time": contact_forces.last_contact_time,
            },
            params={"max_time": 0.5},
            weight=-2.0,
        ),
    },
    terminations={
        **FLAT_VELOCITY_TASK.terminations,
        "base_contact": mdp2.Termination(
            mdp.illegal_contact,
            inputs={
                "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                "body_ids": contact_forces.bodies(".*torso_link"),
            },
            params={"threshold": 1.0},
        ),
    },
    events={
        **FLAT_VELOCITY_TASK.events,
        "add_base_mass": replace(_BASE_MASS, inputs={**_BASE_MASS.inputs, "body_ids": robot.bodies("torso_link")}),
        "base_com": None,
        "reset_robot_joints": replace(
            FLAT_VELOCITY_TASK.events["reset_robot_joints"],
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        ),
    },
)
"""The MDP of the Unitree H1 task."""

_FLAT_DEFAULTS = FlatVelocityEnvCfg()
_FLAT_MJWARP = _FLAT_DEFAULTS.sim.physics.newton_mjwarp
_H1_MJWARP = replace(_FLAT_MJWARP, solver_cfg=replace(_FLAT_MJWARP.solver_cfg, nconmax=15))


@configclass
class H1FlatMDP2EnvCfg(FlatVelocityEnvCfg):
    """Flat-ground velocity tracking with the Unitree H1."""

    sim: SimulationCfg = replace(
        _FLAT_DEFAULTS.sim, physics=FlatVelocityPhysicsCfg(newton_mjwarp=_H1_MJWARP, default=_H1_MJWARP)
    )
    scene: MySceneCfg = replace(FLAT_TERRAIN_SCENE, robot=replace(H1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot"))
    task: mdp2.TaskCfg = TASK
