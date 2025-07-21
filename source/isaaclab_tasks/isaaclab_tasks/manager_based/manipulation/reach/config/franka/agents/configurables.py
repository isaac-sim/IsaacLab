# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..configurables import EnvConfigurables


@configclass
class AgentConfigurables(EnvConfigurables):
    agent: dict[str, any] = {
        "policy": {
            "large_network": RslRlPpoActorCriticCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[512, 256, 128, 64],
                critic_hidden_dims=[512, 256, 128, 64],
                activation="elu",
            ),
            "medium_network": RslRlPpoActorCriticCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[256, 128, 64],
                activation="elu",
            ),
            "small_network": RslRlPpoActorCriticCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[128, 64],
                critic_hidden_dims=[128, 64],
                activation="elu",
            ),
        },
        "algorithm": {
            "standard": RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.001,
                num_learning_epochs=8,
                num_mini_batches=4,
                learning_rate=1.0e-3,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
            ),
            "small_batch": RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.001,
                num_learning_epochs=8,
                num_mini_batches=16,
                learning_rate=1.0e-4,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
            ),
        },
    }
