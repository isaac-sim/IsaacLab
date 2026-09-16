# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field, replace_config

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.006,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.98,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@dataclass
class FrankaDeformablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(24)
    max_iterations: Any = config_field(3000)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("franka_soft")
    obs_groups: Any = config_field(
        {
            "actor": ["policy"],
            "critic": ["policy"],
        }
    )
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = config_field(replace_config(ALGO_CFG, learning_rate=1.0e-3))


@dataclass
class FrankaClothPPORunnerCfg(FrankaDeformablePPORunnerCfg):
    experiment_name: Any = config_field("lift_cloth")


@dataclass
class FrankaCablePPORunnerCfg(FrankaDeformablePPORunnerCfg):
    experiment_name: Any = config_field("lift_cable")


@dataclass
class FrankaDeformableCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(24)
    max_iterations: Any = config_field(5000)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("franka_deformable_camera")
    obs_groups: Any = config_field(
        {
            "actor": ["policy", "proprio", "base_image"],
            "critic": ["policy", "proprio", "perception"],
        }
    )
    actor: Any = config_field(
        RslRlCNNModelCfg(
            obs_normalization=True,
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
            cnn_cfg=RslRlCNNModelCfg.CNNCfg(
                output_channels=[32, 64, 64],
                kernel_size=[8, 4, 3],
                stride=[4, 2, 1],
                activation="elu",
            ),
            activation="elu",
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            obs_normalization=True,
            hidden_dims=[512, 256, 128],
            activation="elu",
        )
    )
    algorithm: Any = config_field(replace_config(ALGO_CFG, num_mini_batches=8))


@dataclass
class FrankaCableCameraPPORunnerCfg(FrankaDeformableCameraPPORunnerCfg):
    experiment_name: Any = config_field("lift_cable_camera")
