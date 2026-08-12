# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    ROBOT_CFG,
    ObjectCfg,
    PhysicsCfg,
    ShadowHandEventCfg,
    ShadowHandRobotCfg,
)
from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES


@configclass
class ShadowHandSceneCfg(InteractiveSceneCfg):
    """Shadow Direct scene defaults.

    ``clone_in_fabric`` is the only backend-varying field: PhysX/OvPhysX use Fabric
    cloning for speed; Newton does not support it. The per-backend value is selected
    inline at the scene field via :func:`~isaaclab_tasks.utils.preset`.
    """

    num_envs = 8192
    env_spacing = 0.75
    replicate_physics = True


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

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )

    # robot
    robot_cfg: ShadowHandRobotCfg = ROBOT_CFG
    actuated_joint_names = SHADOW_ACTUATED_JOINT_NAMES
    fingertip_body_names = SHADOW_FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: ObjectCfg = OBJECT_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene — clone_in_fabric is the only backend-varying field (Newton cannot use Fabric cloning)
    scene: InteractiveSceneCfg = preset(
        default=ShadowHandSceneCfg(clone_in_fabric=False),
        physx=ShadowHandSceneCfg(clone_in_fabric=True),
        isaacsim_physx=ShadowHandSceneCfg(clone_in_fabric=True),
        ovphysx=ShadowHandSceneCfg(clone_in_fabric=True),
        newton_mjwarp=ShadowHandSceneCfg(clone_in_fabric=False),
        newton_kamino=ShadowHandSceneCfg(clone_in_fabric=False),
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
    success_tolerance = 0.1
    max_consecutive_success = 0
    success_count_threshold: int = 1
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):
    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 20
    observation_space = 42
    state_space = 187
    asymmetric_obs = True
    obs_type = "openai"

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=3,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
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
    fall_penalty = -50.0
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: ShadowHandEventCfg = ShadowHandEventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = OPENAI_ACTION_NOISE_CFG
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = OPENAI_OBSERVATION_NOISE_CFG
