# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@dataclass
class NavigationEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(24)
    max_iterations: Any = config_field(1500)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("arl_robot_1_navigation")
    empirical_normalization: Any = config_field(False)
    policy: Any = config_field(
        RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=4.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
