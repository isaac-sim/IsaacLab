# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based PPO baseline for learning with the coffee environment."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class CoffeePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Untuned PPO starting point; the scripted example does not use this policy."""

    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 100
    experiment_name = "rizon_sharpa_coffee"
    init_at_random_ep_len = False
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # Keep the discount and GAE time constants comparable to a 10 Hz task.
        gamma=0.99**0.1,
        lam=0.95**0.1,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
