# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass

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


@configclass
class ShadowHandSceneCfg(InteractiveSceneCfg):
    """Shadow Hand, in-hand object, and shared scene assets."""

    num_envs = 8192
    env_spacing = 0.75
    replicate_physics = True
    ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg())
    robot: ShadowHandRobotCfg = ShadowHandRobotCfg()
    object: RigidObjectCfg = CUBE_CFG
    joint_wrench: JointWrenchSensorCfg | None = None
    goal_object: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157  # (full)
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

    actuated_joint_names = JOINT_NAMES
    actuated_tendon_names = TENDON_NAMES
    actuated_tendon_position_limits = TENDON_POSITION_LIMITS
    fingertip_body_names = FINGERTIP_NAMES

    # scene
    scene: ShadowHandSceneCfg = ShadowHandSceneCfg()

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
    success_tolerance = 0.1
    max_consecutive_success = 0
    in_hand_pos_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    """In-hand goal anchor, relative to the object's default position [m]."""
    goal_marker_position: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """Fixed goal-marker display position [m], environment frame."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
