# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position reaching: move an arm's end effector to commanded poses above a table.

The arms share :data:`REACH_TASK` and :class:`ReachMDP2EnvCfg`, as the manager Reach tasks share their base config.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.reach import reach_env_cfg as core_reach
from isaaclab_tasks.utils import PresetCfg

robot = mdp2.Entity("robot")
joints = robot.joints(".*")

pose_tracking = {
    "command": mdp2.commands.ee_pose,
    "root_link_pose_w": robot.root_link_pose_w,
    "body_link_pose_w": robot.body_link_pose_w,
}
joint_limits = {
    "default_joint_pos": robot.default_joint_pos,
    "default_joint_vel": robot.default_joint_vel,
    "soft_joint_pos_limits": robot.soft_joint_pos_limits,
    "soft_joint_vel_limits": robot.soft_joint_vel_limits,
}
joint_state = {"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel}


@configclass
class ReachPhysicsCfg(PresetCfg):
    """Newton physics presets of the Reach tasks: the MJWarp preset of the manager Reach tasks."""

    newton_mjwarp: NewtonCfg = core_reach.ReachPhysicsCfg().newton_mjwarp
    default: NewtonCfg = newton_mjwarp


REACH_TASK = mdp2.TaskCfg(
    actions={"arm_action": None},
    commands={"ee_pose": None},
    quantities={"position_error": None, "orientation_error": None},
    observations={
        "policy": {
            "joint_pos": mdp2.Observation(
                mdp.joint_pos_rel,
                inputs={
                    "joint_pos": robot.joint_pos,
                    "default_joint_pos": robot.default_joint_pos,
                    "joint_ids": joints,
                },
                noise=mdp2.UniformNoise(-0.01, 0.01),
            ),
            "joint_vel": mdp2.Observation(
                mdp.joint_vel_rel,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "default_joint_vel": robot.default_joint_vel,
                    "joint_ids": joints,
                },
                noise=mdp2.UniformNoise(-0.01, 0.01),
            ),
            "pose_command": mdp2.Observation(mdp.generated_commands, inputs={"command": mdp2.commands.ee_pose}),
            "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
        }
    },
    rewards={
        "end_effector_position_tracking": mdp2.Reward(
            mdp.value, inputs={"value": mdp2.quantities.position_error}, weight=-0.2
        ),
        "end_effector_orientation_tracking": mdp2.Reward(
            mdp.value, inputs={"value": mdp2.quantities.orientation_error}, weight=-0.1
        ),
        "success": mdp2.Reward(
            mdp.is_terminated_before_time_out,
            inputs={"terminated": mdp2.terminated, "time_out": mdp2.time_out},
            weight=10.0,
        ),
        "action_rate": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.0001),
        "action_magnitude": mdp2.Reward(mdp.action_l2, inputs={"actions": mdp2.last_action}, weight=-0.005),
        "joint_vel": None,
    },
    terminations={
        "success": mdp2.Termination(
            mdp.pose_command_success,
            inputs={
                "position_error": mdp2.quantities.position_error,
                "orientation_error": mdp2.quantities.orientation_error,
            },
            params={"position_threshold": 0.05, "orientation_threshold": 0.2},
        ),
        "time_out": mdp2.TimeLimit(episode_length_s=12.0),
    },
    events={"reset_robot_joints": None},
)
"""The MDP of the arms. Each arm's task sets the uses left ``None``: its action, its end-effector pose command and
errors, the joints of its joint velocity penalty, and its joint resets."""


@configclass
class ReachMDP2EnvCfg(MDP2RLEnvCfg):
    """Joint-position reaching with reset-driven reward weight changes."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics=ReachPhysicsCfg(),
        visualizer_cfgs=VisualizerCfg(eye=(3.5, 3.5, 3.5)),
    )
    decimation: int = 4
    curriculum_after_steps: int = 4500
    curriculum_action_rate: float = -0.005
    curriculum_joint_vel: float = -0.001
