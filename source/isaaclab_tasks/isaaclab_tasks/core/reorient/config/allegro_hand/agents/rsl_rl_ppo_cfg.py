# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AllegroHandManagerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(24)
    max_iterations: Any = config_field(5000)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("allegro_hand")
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.001,
            schedule="adaptive",
            gamma=0.998,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )


@dataclass
class AllegroHandDirectPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(16)
    max_iterations: Any = config_field(10000)
    save_interval: Any = config_field(250)
    experiment_name: Any = config_field("allegro_hand_direct")
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[1024, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[1024, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.016,
            max_grad_norm=1.0,
        )
    )
