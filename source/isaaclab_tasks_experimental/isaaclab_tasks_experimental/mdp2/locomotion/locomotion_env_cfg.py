# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant and Humanoid locomotion: run toward a target far along the x axis, without falling."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.core.locomotion.ant.ant_manager_env_cfg import AntSceneCfg
from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidSceneCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp as locomotion

SIM_DT = 1 / 120
DECIMATION = 2
EPISODE_LENGTH_S = 16.0
GROUND_MATERIAL = sim_utils.RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

HUMANOID_JOINT_GEARS = {
    ".*_waist.*": 67.5,
    ".*_upper_arm.*": 67.5,
    "pelvis": 67.5,
    ".*_lower_arm": 45.0,
    ".*_thigh:0": 45.0,
    ".*_thigh:1": 135.0,
    ".*_thigh:2": 45.0,
    ".*_shin": 90.0,
    ".*_foot.*": 22.5,
}
"""Gear ratios of the Humanoid joints, which scale its actions to joint efforts [N m]."""

robot = mdp2.Entity("robot")
joint_wrench = mdp2.Entity("joint_wrench")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root = {"root_link_pose_w": robot.root_link_pose_w}
velocity = {**root, "root_com_vel_w": robot.root_com_vel_w}
joint_pos = {"joint_pos": robot.joint_pos, "soft_joint_pos_limits": robot.soft_joint_pos_limits, "joint_ids": joints}
to_target = {**root, "env_origins": scene.env_origins}
target = {"target_x": 1000.0, "target_y": 0.0}  # Target position in each world [m]


@configclass
class AntPhysicsCfg(PresetCfg):
    """Newton physics presets of the Ant task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=45, nconmax=25, cone="pyramidal", integrator="implicitfast", impratio=1),
        num_substeps=1,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    """Newton physics presets of the Humanoid task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=80, nconmax=25, cone="pyramidal", update_data_interval=2, integrator="implicitfast", impratio=1
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


QUANTITIES = {
    "up": mdp2.Quantity(locomotion.base_up_proj, inputs={**root, "gravity_w": scene.gravity_w}),
    "heading": mdp2.Quantity(locomotion.base_heading_proj, inputs=to_target, params=target),
    "success_rate": mdp2.Quantity(locomotion.survival, inputs={"time_out": mdp2.time_out}, log=True),
}
"""Quantities of both robots: the upright and target-heading projections, and the logged survival."""

EVENTS = {
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
}
"""Reset events of both robots: the default root state, and joint offsets."""

ANT_TASK = mdp2.TaskCfg(
    actions={
        # The effort is clipped at the gear magnitude: unbounded joint efforts drive the solver to NaN.
        "joint_effort": mdp2.DirectAction(target=robot.joint_effort_target, scale=7.5, clip={".*": (-7.5, 7.5)}),
    },
    quantities=QUANTITIES,
    observations={
        "policy": {
            "base_height": mdp2.Observation(locomotion.base_height, inputs=root),
            "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=velocity),
            "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=velocity),
            "base_yaw_roll": mdp2.Observation(locomotion.base_yaw_roll, inputs=root),
            "base_angle_to_target": mdp2.Observation(locomotion.base_angle_to_target, inputs=to_target, params=target),
            "base_up_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.up}),
            "base_heading_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.heading}),
            "joint_pos_norm": mdp2.Observation(mdp.joint_pos_limit_normalized, inputs=joint_pos),
            "joint_vel_rel": mdp2.Observation(
                mdp.joint_vel_rel,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "default_joint_vel": robot.default_joint_vel,
                    "joint_ids": joints,
                },
                scale=0.2,
            ),
            "feet_body_forces": mdp2.Observation(
                mdp.body_incoming_wrench,
                inputs={
                    "force": joint_wrench.force,
                    "torque": joint_wrench.torque,
                    "body_ids": joint_wrench.bodies(
                        "front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"
                    ),
                },
                scale=0.1,
            ),
            "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
        },
    },
    rewards={
        "progress": mdp2.Reward(
            locomotion.progress_reward, inputs=to_target, params={**target, "step_dt": mdp2.step_dt}, weight=1.0
        ),
        "alive": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=0.5),
        "upright": mdp2.Reward(
            locomotion.upright_posture_bonus, inputs={"up": mdp2.quantities.up}, params={"threshold": 0.93}, weight=0.1
        ),
        "move_to_target": mdp2.Reward(
            locomotion.move_to_target_bonus,
            inputs={"heading": mdp2.quantities.heading},
            params={"threshold": 0.8},
            weight=0.5,
        ),
        "action_l2": mdp2.Reward(mdp.action_l2, inputs={"actions": mdp2.last_action}, weight=-0.005),
        "energy": mdp2.Reward(
            locomotion.power_consumption,
            inputs={"actions": mdp2.last_action, "joint_vel": robot.joint_vel, "joint_ids": joints},
            params={"gear_ratio": {".*": 15.0}},
            weight=-0.05,
        ),
        "joint_pos_limits": mdp2.Reward(
            locomotion.joint_pos_limits_penalty_ratio,
            inputs=joint_pos,
            params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
            weight=-0.1,
        ),
        "terminating": mdp2.Reward(
            locomotion.terminated_penalty,
            inputs={"terminated": mdp2.terminated},
            params={"step_dt": mdp2.step_dt},
            weight=-2.0,
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(EPISODE_LENGTH_S),
        "torso_height": mdp2.Termination(mdp.root_height_below_minimum, inputs=root, params={"minimum_height": 0.31}),
    },
    events=EVENTS,
)
"""The MDP of the Ant task."""

HUMANOID_TASK = mdp2.TaskCfg(
    actions={
        # The effort is clipped at the gear magnitude: unbounded joint efforts drive the solver to NaN.
        "joint_effort": mdp2.DirectAction(
            target=robot.joint_effort_target,
            scale=HUMANOID_JOINT_GEARS,
            clip={name: (-gear, gear) for name, gear in HUMANOID_JOINT_GEARS.items()},
        ),
    },
    quantities=QUANTITIES,
    observations={
        "policy": {
            "base_height": mdp2.Observation(locomotion.base_height, inputs=root),
            "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=velocity),
            "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=velocity, scale=0.25),
            "base_yaw_roll": mdp2.Observation(locomotion.base_yaw_roll, inputs=root),
            "base_angle_to_target": mdp2.Observation(locomotion.base_angle_to_target, inputs=to_target, params=target),
            "base_up_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.up}),
            "base_heading_proj": mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.heading}),
            "joint_pos_norm": mdp2.Observation(mdp.joint_pos_limit_normalized, inputs=joint_pos),
            "joint_vel_rel": mdp2.Observation(
                mdp.joint_vel_rel,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "default_joint_vel": robot.default_joint_vel,
                    "joint_ids": joints,
                },
                scale=0.1,
            ),
            "feet_body_forces": mdp2.Observation(
                mdp.body_incoming_wrench,
                inputs={
                    "force": joint_wrench.force,
                    "torque": joint_wrench.torque,
                    "body_ids": joint_wrench.bodies("left_foot", "right_foot"),
                },
                scale=0.01,
            ),
            "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
        },
    },
    rewards={
        "progress": mdp2.Reward(
            locomotion.progress_reward, inputs=to_target, params={**target, "step_dt": mdp2.step_dt}, weight=1.0
        ),
        "alive": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=2.0),
        "upright": mdp2.Reward(
            locomotion.upright_posture_bonus, inputs={"up": mdp2.quantities.up}, params={"threshold": 0.93}, weight=0.1
        ),
        "move_to_target": mdp2.Reward(
            locomotion.move_to_target_bonus,
            inputs={"heading": mdp2.quantities.heading},
            params={"threshold": 0.8},
            weight=0.5,
        ),
        "action_l2": mdp2.Reward(mdp.action_l2, inputs={"actions": mdp2.last_action}, weight=-0.01),
        "energy": mdp2.Reward(
            locomotion.power_consumption,
            inputs={"actions": mdp2.last_action, "joint_vel": robot.joint_vel, "joint_ids": joints},
            params={"gear_ratio": HUMANOID_JOINT_GEARS},
            weight=-0.005,
        ),
        "joint_pos_limits": mdp2.Reward(
            locomotion.joint_pos_limits_penalty_ratio,
            inputs=joint_pos,
            params={"threshold": 0.98, "gear_ratio": HUMANOID_JOINT_GEARS},
            weight=-0.25,
        ),
        "terminating": mdp2.Reward(
            locomotion.terminated_penalty,
            inputs={"terminated": mdp2.terminated},
            params={"step_dt": mdp2.step_dt},
            weight=-1.0,
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(EPISODE_LENGTH_S),
        "torso_height": mdp2.Termination(mdp.root_height_below_minimum, inputs=root, params={"minimum_height": 0.8}),
    },
    events=EVENTS,
)
"""The MDP of the Humanoid task."""


@configclass
class AntMDP2EnvCfg(MDP2RLEnvCfg):
    """Ant locomotion on MJWarp."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=AntPhysicsCfg(), physics_material=GROUND_MATERIAL
    )
    scene: AntSceneCfg = AntSceneCfg(num_envs=4096, env_spacing=5.0)
    decimation: int = DECIMATION
    task: mdp2.TaskCfg = ANT_TASK


@configclass
class HumanoidMDP2EnvCfg(MDP2RLEnvCfg):
    """Humanoid locomotion on MJWarp."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=HumanoidPhysicsCfg(), physics_material=GROUND_MATERIAL
    )
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=5.0)
    decimation: int = DECIMATION
    task: mdp2.TaskCfg = HUMANOID_TASK
