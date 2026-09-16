# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct configuration for the OpenAI Shadow Hand variant, moved unchanged from the core task."""

from dataclasses import dataclass
from typing import Any

from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import config_field
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    PhysicsCfg,
    ShadowHandRandomizationEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg


@dataclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):
    # env
    decimation: Any = config_field(3)
    episode_length_s: Any = config_field(8.0)
    action_space: Any = config_field(20)
    observation_space: Any = config_field(42)
    state_space: Any = config_field(187)
    asymmetric_obs: Any = config_field(True)
    obs_type: Any = config_field("openai")

    # simulation
    sim: SimulationCfg = config_field(
        SimulationCfg(
            dt=1 / 60,
            render_interval=decimation,
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
            physics=PhysicsCfg(),
        )
    )
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
    fall_penalty: Any = config_field(-50.0)
    vel_obs_scale: Any = config_field(0.2)
    success_tolerance: Any = config_field(0.4)
    max_consecutive_success: Any = config_field(50)
    av_factor: Any = config_field(0.1)
    act_moving_average: Any = config_field(0.3)
    force_torque_obs_scale: Any = config_field(10.0)
    # domain randomization config
    events: ShadowHandRandomizationEventCfg = config_field(ShadowHandRandomizationEventCfg())
    # per-step gaussian noise + reset-sampled bias, as in the paper
    action_noise_model: NoiseModelWithAdditiveBiasCfg = config_field(
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
        )
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = config_field(
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        )
    )
