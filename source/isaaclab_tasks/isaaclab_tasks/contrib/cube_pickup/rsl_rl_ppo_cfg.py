# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the sparse pickup reference example."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class CubePickupPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO settings for learning the sparse pickup goal from generated reset states."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "cube_pickup_reset_buffer"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.7),
        obs_normalization=True,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(obs_normalization=True, hidden_dims=[256, 128, 64], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
