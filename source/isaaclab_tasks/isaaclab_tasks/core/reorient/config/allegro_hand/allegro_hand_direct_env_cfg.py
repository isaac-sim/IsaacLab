# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
)

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@configclass
class AllegroHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 16
    observation_space = 124  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation — values mirrored by the manager cfg
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # robot
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_ROBOT_CFG

    actuated_joint_names = ALLEGRO_ACTUATED_JOINT_NAMES
    fingertip_body_names = ALLEGRO_FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: RigidObjectCfg = CUBE_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=0.75,
        replicate_physics=True,
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    in_hand_pos_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    """In-hand goal anchor, relative to the object's default position [m]."""
    goal_marker_position: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """Fixed goal-marker display position [m], environment frame."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
