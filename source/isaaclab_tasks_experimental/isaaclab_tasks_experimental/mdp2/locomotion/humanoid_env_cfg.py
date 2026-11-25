# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid locomotion on MDP2, with the constants that the other Humanoid tasks share."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.locomotion.humanoid import humanoid_common
from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidSceneCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp as locomotion
from .locomotion_env_cfg import LOCOMOTION_TASK, joint_pos, joint_wrench, joints, robot, root, velocity


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    """Newton physics presets of the Humanoid task: the MJWarp preset of the other Humanoid tasks."""

    newton_mjwarp: NewtonCfg = humanoid_common.HumanoidPhysicsCfg().newton_mjwarp
    default: NewtonCfg = newton_mjwarp


HUMANOID_TASK = replace(
    LOCOMOTION_TASK,
    actions={
        # The effort is clipped at the gear magnitude: unbounded joint efforts drive the solver to NaN.
        "joint_effort": mdp2.DirectAction(
            target=robot.joint_effort_target,
            scale=humanoid_common.JOINT_GEARS,
            clip=humanoid_common.JOINT_EFFORT_LIMITS,
        ),
    },
    observations={
        "policy": {
            **LOCOMOTION_TASK.observations["policy"],
            "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=velocity, scale=0.25),
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
                    "body_ids": joint_wrench.bodies(*humanoid_common.FEET_BODY_NAMES),
                },
                scale=0.01,
            ),
        },
    },
    rewards={
        **LOCOMOTION_TASK.rewards,
        "alive": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=2.0),
        "action_l2": mdp2.Reward(mdp.action_l2, inputs={"actions": mdp2.last_action}, weight=-0.01),
        "energy": mdp2.Reward(
            locomotion.power_consumption,
            inputs={"actions": mdp2.last_action, "joint_vel": robot.joint_vel, "joint_ids": joints},
            params={"gear_ratio": humanoid_common.JOINT_GEARS},
            weight=-0.005,
        ),
        "joint_pos_limits": mdp2.Reward(
            locomotion.joint_pos_limits_penalty_ratio,
            inputs=joint_pos,
            params={"threshold": 0.98, "gear_ratio": humanoid_common.JOINT_GEARS},
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
        "time_out": mdp2.TimeLimit(16.0),
        "torso_height": mdp2.Termination(mdp.root_height_below_minimum, inputs=root, params={"minimum_height": 0.8}),
    },
)
"""The MDP of the Humanoid task."""


@configclass
class HumanoidMDP2EnvCfg(MDP2RLEnvCfg):
    """Humanoid locomotion on MJWarp."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics=HumanoidPhysicsCfg(),
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=5.0)
    decimation: int = 2
    task: mdp2.TaskCfg = HUMANOID_TASK
