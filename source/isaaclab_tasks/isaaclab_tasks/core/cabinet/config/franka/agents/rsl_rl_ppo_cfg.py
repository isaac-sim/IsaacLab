# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class CabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(96)
    max_iterations: Any = config_field(400)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("franka_open_drawer")
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=False,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=1e-3,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.02,
            max_grad_norm=1.0,
        )
    )


@dataclass
class FrankaCabinetPPORunnerCfg(CabinetPPORunnerCfg):
    experiment_name: Any = config_field("franka_open_drawer_direct")
