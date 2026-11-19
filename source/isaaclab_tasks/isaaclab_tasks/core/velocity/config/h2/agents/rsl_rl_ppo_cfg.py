# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class H2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for training H2 from scratch on rough terrain."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 1000
    experiment_name = "h2_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
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
class H2FlatPPORunnerCfg(H2RoughPPORunnerCfg):
    """PPO configuration for training H2 from scratch on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "h2_flat"
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]
