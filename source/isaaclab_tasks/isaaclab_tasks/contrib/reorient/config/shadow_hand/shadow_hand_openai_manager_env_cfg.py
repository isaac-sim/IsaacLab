# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the OpenAI Shadow Hand reorientation variant.

Builds on the core task's ``presets=openai`` observation architecture and adds only the
training regime of OpenAI et al., "Learning Dexterous In-Hand Manipulation"
(https://arxiv.org/abs/1808.00177): 20 Hz control, action and observation noise, a looser
goal chased repeatedly, and an episode budget spent per goal.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import isaaclab_tasks.contrib.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import PhysicsCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    RewardsCfg,
    ShadowHandOpenAIObservationsCfg,
    ShadowHandOpenAIObsManagerEnvCfg,
    TerminationsCfg,
)


@configclass
class ObservationsCfg(ShadowHandOpenAIObservationsCfg):
    """The core ``presets=openai`` groups, with the paper's actor-observation noise."""

    def __post_init__(self):
        # mirrors ShadowHandOpenAIEnvCfg.observation_noise_model (guarded by the value-parity test)
        self.policy.openai.params["noise_model"] = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        )


@configclass
class OpenAIRewardsCfg(RewardsCfg):
    """The core reward terms, plus the paper's penalty for dropping the cube."""

    object_away_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={"term_keys": "object_out_of_reach"},
    )


@configclass
class OpenAITerminationsCfg(TerminationsCfg):
    """The core terminations, plus a streak cap and an episode budget spent per goal.

    The Direct variant reports both the streak cap and the elapsed-time limit as
    truncations, so both carry ``time_out=True``.
    """

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success,
        time_out=True,
        params={"num_success": 50, "command_name": "object_pose"},
    )
    time_out = DoneTerm(
        func=mdp.reorient_timeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "success_tolerance": 0.4,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ShadowHandOpenAIObsManagerEnvCfg):
    """The paper's training regime on top of the core task's ``presets=openai``."""

    # 20 Hz control, and a shorter budget because it is spent per goal
    decimation = 3
    episode_length_s = 8.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    rewards: OpenAIRewardsCfg = OpenAIRewardsCfg()
    terminations: OpenAITerminationsCfg = OpenAITerminationsCfg()

    enable_domain_randomization: bool = True
    """Apply the domain-randomization event terms.

    On by default: unlike the other reorientation tasks, the OpenAI Direct environment
    randomizes as well, so the two workflows only match with these enabled. ``__post_init__``
    reads it while building the configuration, before Hydra applies command-line overrides, so
    ``env.enable_domain_randomization=false`` has no effect -- set it on the configuration.
    Changing it requires retraining.
    """

    def __post_init__(self):
        super().__post_init__()
        # a looser goal, chased repeatedly, under an exponential-moving-average action filter
        self.commands.object_pose.orientation_success_threshold = 0.4
        self.actions.joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
            asset_name=self.actions.joint_pos.asset_name,
            joint_names=self.actions.joint_pos.joint_names,
            alpha=0.3,
            rescale_to_limits=True,
            # mirrors ShadowHandOpenAIEnvCfg.action_noise_model (guarded by the value-parity test)
            noise_model=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
            ),
        )
