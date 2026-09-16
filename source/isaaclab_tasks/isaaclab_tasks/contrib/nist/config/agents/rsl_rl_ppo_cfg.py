# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(32)
    max_iterations: Any = config_field(15000)
    save_interval: Any = config_field(200)
    experiment_name: Any = config_field("factory")
    obs_groups: Any = config_field({"actor": ["policy"], "critic": ["policy"]})
    actor: Any = config_field(
        RslRlMLPModelCfg(
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
            obs_normalization=True,
            hidden_dims=[512, 256, 128, 64],
            activation="elu",
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            obs_normalization=True,
            hidden_dims=[512, 256, 128, 64],
            activation="elu",
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
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
    )
