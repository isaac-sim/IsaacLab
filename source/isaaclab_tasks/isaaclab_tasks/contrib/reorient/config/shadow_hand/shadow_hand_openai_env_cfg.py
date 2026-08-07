# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct configuration for the OpenAI reduced-observation Shadow Hand variant."""

from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    PhysicsCfg,
    ShadowHandEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandOpenAIObsEnvCfg


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandOpenAIObsEnvCfg):
    """The paper's training regime on top of the core task's ``presets=openai``.

    Inherits the reduced actor and privileged critic; everything below is the sim-to-real
    setup that does not generalize to other reorientation work.
    """

    # 20 Hz control, and a shorter budget because it is spent per goal
    decimation = 3
    episode_length_s = 8.0

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # a looser goal, chased repeatedly, with falls punished
    fall_penalty = -50.0
    success_tolerance = 0.4
    max_consecutive_success = 50
    act_moving_average = 0.3
    # domain randomization config
    events: ShadowHandEventCfg = ShadowHandEventCfg()
    # per-step gaussian noise + reset-sampled bias; mirrored by the manager cfg
    # (guarded by the value-parity test)
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
