# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .models import SimBaModelCfg


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=6e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.90,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FactoryVariantPPORunnerCfg(FactoryPPORunnerCfg):
    """PPO runner using SimBa and encoded variant point clouds."""

    obs_groups = {"actor": ["policy", "perception"], "critic": ["policy", "perception"]}
    actor = SimBaModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
        encoder_cfg={
            "perception": SimBaModelCfg.MLPEncoderCfg(
                hidden_dims=[256], output_dim=128, activation="elu", last_activation="elu"
            )
        },
    )
    critic = SimBaModelCfg(
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
        encoder_cfg={
            "perception": SimBaModelCfg.MLPEncoderCfg(
                hidden_dims=[256], output_dim=128, activation="elu", last_activation="elu"
            )
        },
    )
