# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow Ant environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.locomotion.ant.ant_common import (
    FEET_BODY_NAMES,
    JOINT_GEARS,
    TERRAIN_CFG,
    WALK_TARGET_POS,
    AntPhysicsCfg,
)

from isaaclab_assets.robots.ant import ANT_CFG


@configclass
class AntDirectSceneCfg(InteractiveSceneCfg):
    """Ant, terrain, sensor, and light constructed through one clone lifecycle."""

    terrain = TERRAIN_CFG
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class AntEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow Ant walking environment."""

    # env
    episode_length_s = 16.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 60
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=AntPhysicsCfg())

    # scene
    scene: AntDirectSceneCfg = AntDirectSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    joint_gears: dict[str, float] = JOINT_GEARS
    feet_body_names: list[str] = FEET_BODY_NAMES

    # walk target, relative to the environment origin
    target_pos: tuple[float, float, float] = WALK_TARGET_POS

    # reset
    initial_joint_pos_range: tuple[float, float] = (-0.2, 0.2)  # [rad]
    initial_joint_vel_range: tuple[float, float] = (-0.1, 0.1)  # [rad/s]

    # reward scales
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    joint_pos_limits_cost_scale: float = 0.1
    joint_pos_limits_threshold: float = 0.99
    death_cost: float = -2.0

    # termination
    termination_height: float = 0.31  # [m]

    # observation scales
    dof_vel_scale: float = 0.2
    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1
