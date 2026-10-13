# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO baseline for normalized cable insertion actions."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class GB300SharpaInsertionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """State-based PPO baseline; no trained-policy success is implied."""

    num_steps_per_env = 96
    max_iterations = 3000
    init_at_random_ep_len = False
    save_interval = 50
    clip_actions = 1.0
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "gb300_sharpa_insertion"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.0001,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GB300SharpaInsertionPPORunnerPresetsCfg(PresetCfg):
    """Disable action clipping only for the absolute scripted action contract."""

    default = GB300SharpaInsertionPPORunnerCfg()
    scripted = GB300SharpaInsertionPPORunnerCfg(clip_actions=None)
