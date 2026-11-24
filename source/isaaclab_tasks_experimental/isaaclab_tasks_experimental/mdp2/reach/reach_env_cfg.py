# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position reaching: move an arm's end effector to commanded poses above a table."""

import math

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_isaacsim.sim.schemas import PhysxRigidBodyCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.reach.reach_env_cfg import ReachSceneCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import FRANKA_PANDA_CFG, UR10_CFG

SIM_DT = 1 / 120
DECIMATION = 4
EPISODE_LENGTH_S = 12.0

robot = mdp2.Entity("robot")
joints = robot.joints(".*")
franka_arm = robot.joints("panda_joint.*")

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


@configclass
class ReachPhysicsCfg(PresetCfg):
    """Newton physics presets of the Reach tasks."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=100, nconmax=20, cone="pyramidal", integrator="implicitfast", impratio=1, update_data_interval=2
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


OBSERVATIONS = {
    "policy": {
        "joint_pos": mdp2.Observation(
            mdp.joint_pos_rel,
            inputs={"joint_pos": robot.joint_pos, "default_joint_pos": robot.default_joint_pos, "joint_ids": joints},
            noise=mdp2.UniformNoise(-0.01, 0.01),
        ),
        "joint_vel": mdp2.Observation(
            mdp.joint_vel_rel,
            inputs={"joint_vel": robot.joint_vel, "default_joint_vel": robot.default_joint_vel, "joint_ids": joints},
            noise=mdp2.UniformNoise(-0.01, 0.01),
        ),
        "pose_command": mdp2.Observation(mdp.generated_commands, inputs={"command": mdp2.commands.ee_pose}),
        "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
    }
}
"""Observations of both arms: the joint state, the commanded pose, and the last action."""

REWARDS = {
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
}
"""Rewards of both arms, except the joint velocity penalty, whose joints differ."""

TERMINATIONS = {
    "success": mdp2.Termination(
        mdp.pose_command_success,
        inputs={
            "position_error": mdp2.quantities.position_error,
            "orientation_error": mdp2.quantities.orientation_error,
        },
        params={"position_threshold": 0.05, "orientation_threshold": 0.2},
    ),
    "time_out": mdp2.TimeLimit(episode_length_s=EPISODE_LENGTH_S),
}
"""Terminations of both arms: reaching the commanded pose, and the time limit."""

UR10_TASK = mdp2.TaskCfg(
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
    observations=OBSERVATIONS,
    rewards={
        **REWARDS,
        "joint_vel": mdp2.Reward(
            mdp.joint_vel_l2, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}, weight=-0.0001
        ),
    },
    terminations=TERMINATIONS,
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

FRANKA_TASK = mdp2.TaskCfg(
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
    observations=OBSERVATIONS,
    rewards={
        **REWARDS,
        "joint_vel": mdp2.Reward(
            mdp.joint_vel_l2, inputs={"joint_vel": robot.joint_vel, "joint_ids": franka_arm}, weight=-0.0001
        ),
    },
    terminations=TERMINATIONS,
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
class ReachMDP2EnvCfg(MDP2RLEnvCfg):
    """Joint-position reaching with reset-driven reward weight changes."""

    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=ReachPhysicsCfg(),
        visualizer_cfgs=VisualizerCfg(eye=(3.5, 3.5, 3.5)),
    )
    decimation: int = DECIMATION
    curriculum_after_steps: int = 4500
    curriculum_action_rate: float = -0.005
    curriculum_joint_vel: float = -0.001


@configclass
class UR10ReachMDP2EnvCfg(ReachMDP2EnvCfg):
    """UR10 joint-position Reach on MJWarp."""

    scene: ReachSceneCfg = ReachSceneCfg(
        num_envs=4096, env_spacing=2.5, robot=replace(UR10_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    task: mdp2.TaskCfg = UR10_TASK


@configclass
class FrankaReachMDP2EnvCfg(ReachMDP2EnvCfg):
    """Franka joint-position Reach on MJWarp."""

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5, robot=FRANKA_ROBOT_CFG)
    task: mdp2.TaskCfg = FRANKA_TASK
