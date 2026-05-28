# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    ROBOT_CFG,
    ObjectCfg,
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

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # robot
    robot_cfg: ArticulationCfg = ROBOT_CFG

    # Order matches the prior Isaac Allegro layout (per-knuckle across fingers); names follow MuJoCo Menagerie MJCF.
    actuated_joint_names = [
        "ffj0",
        "mfj0",
        "rfj0",
        "thj0",
        "ffj1",
        "mfj1",
        "rfj1",
        "thj1",
        "ffj2",
        "mfj2",
        "rfj2",
        "thj2",
        "ffj3",
        "mfj3",
        "rfj3",
        "thj3",
    ]
    fingertip_body_names = [
        "ff_tip",
        "mf_tip",
        "rf_tip",
        "th_tip",
    ]

    # in-hand object
    object_cfg: ObjectCfg = OBJECT_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
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
    success_count_threshold: int = 1
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
