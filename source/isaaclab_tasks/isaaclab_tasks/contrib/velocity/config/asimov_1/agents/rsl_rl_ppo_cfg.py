# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""PPO and AMP configurations."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_assets.robots.asimov_1 import ASIMOV_1_JOINT_NAMES

from ..amp_env_cfg import (
    ASIMOV_1_AMP_OBS_TERMS,
    ASIMOV_1_ANCHOR_NAME,
    ASIMOV_1_KEY_BODY_NAMES,
    ASIMOV_1_MOTION_FILES,
)
from ..motions.motion_dataset import MotionDatasetCfg


@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """RSL-RL PPO configuration extended with AMP discriminator settings."""

    class_name: str = "isaaclab_tasks.contrib.velocity.config.asimov_1.agents.amp_ppo:AMPPPO"

    amp_data: MotionDatasetCfg = MISSING

    amp_obs_key: str = "amp"

    amp_reward_coef: float = 0.3

    amp_task_reward_lerp: float = 0.7

    amp_discr_hidden_dims: list[int] = [256, 256]
    amp_discr_activation: str = "relu"
    amp_feature_normalization: bool = True
    amp_grad_pen_lambda: float = 10.0
    amp_replay_buffer_size: int = 100_000

    amp_discr_trunk_weight_decay: float = 1.0e-3
    amp_discr_head_weight_decay: float = 1.0e-1

    amp_update_interval: int = 1

    amp_reward_command_gate: bool = False

    amp_reward_command_name: str | None = None
    amp_reward_command_threshold: float = 0.0

    amp_rollout_obs_clip: float | None = 500.0

    amp_min_normalized_std: float | None = 0.0


@configclass
class Asimov1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner configuration for the Asimov-1 PPO baseline."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "asimov1_velocity"
    run_name = "ppo"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        optimizer="adam",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Asimov1AMPRunnerCfg(Asimov1PPORunnerCfg):
    """RSL-RL runner configuration for Asimov-1 AMP training."""

    experiment_name = "asimov_velocity_amp"
    run_name = "amp"
    algorithm = AMPPPOAlgorithmCfg(
        optimizer="adam",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_data=MotionDatasetCfg(
            motion_files=ASIMOV_1_MOTION_FILES,
            joint_names=list(ASIMOV_1_JOINT_NAMES),
            body_names=list(ASIMOV_1_KEY_BODY_NAMES),
            amp_obs_terms=list(ASIMOV_1_AMP_OBS_TERMS),
            anchor_name=ASIMOV_1_ANCHOR_NAME,
        ),
        amp_reward_coef=0.3,
        amp_task_reward_lerp=0.7,
        amp_discr_hidden_dims=[256, 256],
        amp_reward_command_gate=True,
        amp_reward_command_name="twist",
        amp_reward_command_threshold=0.1,
    )
