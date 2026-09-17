# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
)

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@dataclass
class AllegroHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation: Any = 4
    episode_length_s: Any = 10.0
    action_space: Any = 16
    observation_space: Any = 124  # (full)
    state_space: Any = 0
    asymmetric_obs: Any = False
    obs_type: Any = "full"

    # simulation — values mirrored by the manager cfg
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=1 / 120,
            render_interval=4,
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
            physics=PhysicsCfg(),
        )
    )
    # robot
    robot_cfg: ArticulationCfg = field(default_factory=lambda: deepcopy(ALLEGRO_HAND_ROBOT_CFG))

    actuated_joint_names: Any = field(default_factory=lambda: deepcopy(ALLEGRO_ACTUATED_JOINT_NAMES))
    fingertip_body_names: Any = field(default_factory=lambda: deepcopy(ALLEGRO_FINGERTIP_BODY_NAMES))

    # in-hand object
    object_cfg: RigidObjectCfg = field(default_factory=lambda: deepcopy(CUBE_CFG))
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = field(default_factory=lambda: deepcopy(GOAL_OBJECT_CFG))
    # scene
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
            num_envs=8192,
            env_spacing=0.75,
            replicate_physics=True,
        )
    )
    # reset
    reset_position_noise: Any = 0.01  # range of position at reset
    reset_dof_pos_noise: Any = 0.2  # range of dof pos at reset
    reset_dof_vel_noise: Any = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale: Any = -10.0
    rot_reward_scale: Any = 1.0
    rot_eps: Any = 0.1
    action_penalty_scale: Any = -0.0002
    reach_goal_bonus: Any = 250.0
    fall_penalty: Any = 0.0
    fall_dist: Any = 0.24
    vel_obs_scale: Any = 0.2
    success_tolerance: Any = 0.2
    max_consecutive_success: Any = 0
    in_hand_pos_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    """In-hand goal anchor, relative to the object's default position [m]."""
    goal_marker_position: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """Fixed goal-marker display position [m], environment frame."""
    av_factor: Any = 0.1
    act_moving_average: Any = 1.0
    force_torque_obs_scale: Any = 10.0
