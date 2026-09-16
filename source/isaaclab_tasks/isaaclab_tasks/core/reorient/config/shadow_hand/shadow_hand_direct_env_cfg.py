# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import config_field

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
    ShadowHandRobotCfg,
)

from isaaclab_assets.robots.shadow_hand import (
    FINGERTIP_NAMES,
    JOINT_NAMES,
    TENDON_NAMES,
    TENDON_POSITION_LIMITS,
)


@dataclass
class ShadowHandSceneCfg(InteractiveSceneCfg):
    """Shadow Direct scene defaults."""

    num_envs: Any = config_field(8192)
    env_spacing: Any = config_field(0.75)
    replicate_physics: Any = config_field(True)


@dataclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation: Any = config_field(2)
    episode_length_s: Any = config_field(10.0)
    action_space: Any = config_field(20)
    observation_space: Any = config_field(157)  # (full)
    state_space: Any = config_field(0)
    asymmetric_obs: Any = config_field(False)
    obs_type: Any = config_field("full")

    # simulation — values mirrored by the manager cfg
    sim: SimulationCfg = config_field(
        SimulationCfg(
            dt=1 / 120,
            render_interval=decimation,
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
            physics=PhysicsCfg(),
        )
    )

    # robot
    robot_cfg: ShadowHandRobotCfg = config_field(ShadowHandRobotCfg())
    actuated_joint_names: Any = config_field(JOINT_NAMES)
    actuated_tendon_names: Any = config_field(TENDON_NAMES)
    actuated_tendon_position_limits: Any = config_field(TENDON_POSITION_LIMITS)
    fingertip_body_names: Any = config_field(FINGERTIP_NAMES)

    # in-hand object
    object_cfg: RigidObjectCfg = config_field(CUBE_CFG)
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = config_field(GOAL_OBJECT_CFG)
    # scene
    scene: InteractiveSceneCfg = config_field(ShadowHandSceneCfg())

    # reset
    reset_position_noise: Any = config_field(0.01)  # range of position at reset
    reset_dof_pos_noise: Any = config_field(0.2)  # range of dof pos at reset
    reset_dof_vel_noise: Any = config_field(0.0)  # range of dof vel at reset
    # reward scales
    dist_reward_scale: Any = config_field(-10.0)
    rot_reward_scale: Any = config_field(1.0)
    rot_eps: Any = config_field(0.1)
    action_penalty_scale: Any = config_field(-0.0002)
    reach_goal_bonus: Any = config_field(250.0)
    fall_penalty: Any = config_field(0.0)
    fall_dist: Any = config_field(0.24)
    vel_obs_scale: Any = config_field(0.2)
    success_tolerance: Any = config_field(0.1)
    max_consecutive_success: Any = config_field(0)
    in_hand_pos_offset: tuple[float, float, float] = config_field((0.0, 0.0, -0.04))
    """In-hand goal anchor, relative to the object's default position [m]."""
    goal_marker_position: tuple[float, float, float] = config_field((-0.2, -0.45, 0.68))
    """Fixed goal-marker display position [m], environment frame."""
    av_factor: Any = config_field(0.1)
    act_moving_average: Any = config_field(1.0)
    force_torque_obs_scale: Any = config_field(10.0)
