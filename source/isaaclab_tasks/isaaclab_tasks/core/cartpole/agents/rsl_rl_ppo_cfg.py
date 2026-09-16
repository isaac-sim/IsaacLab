# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import isaaclab_tasks.core.cartpole.mdp.symmetry as symmetry
from isaaclab_tasks.utils import PresetCfg


@dataclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(16)
    max_iterations: Any = config_field(150)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("cartpole")
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[32, 32],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[32, 32],
            activation="elu",
            obs_normalization=False,
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
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )


@dataclass
class CartpoleDirectPPORunnerCfg(CartpolePPORunnerCfg):
    experiment_name: Any = config_field("cartpole_direct")


@dataclass
class CartpolePPORunnerWithSymmetryCfg(CartpolePPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, data_augmentation_func=symmetry.compute_symmetric_states
            ),
        )
    )


@dataclass
class CartpoleCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """CNN policy for the raw RGB/depth camera observation pipelines."""

    num_steps_per_env: Any = config_field(64)
    max_iterations: Any = config_field(200)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("cartpole_camera")
    obs_groups: Any = config_field({"actor": ["policy"], "critic": ["critic"]})
    clip_actions: Any = config_field(1.0)
    actor: Any = config_field(
        RslRlCNNModelCfg(
            cnn_cfg=RslRlCNNModelCfg.CNNCfg(
                output_channels=[8, 16, 16],
                kernel_size=[5, 3, 3],
                stride=[2, 2, 2],
                activation="relu",
            ),
            hidden_dims=[64],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[32, 32],
            activation="elu",
            obs_normalization=False,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=2.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=4,
            num_mini_batches=16,
            learning_rate=3.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
            share_cnn_encoders=False,
        )
    )


@dataclass
class CartpoleCameraDirectPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    experiment_name: Any = config_field("cartpole_camera_direct")


@dataclass
class CartpoleCameraFeaturePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MLP policy for the pretrained-feature pipelines (ResNet18, Theia-Tiny)."""

    num_steps_per_env: Any = config_field(16)
    max_iterations: Any = config_field(200)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("cartpole_features")
    obs_groups: Any = config_field({"actor": ["policy"], "critic": ["policy"]})
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=4.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=8,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
        )
    )


@dataclass
class CartpoleCameraPPORunnerPresetsCfg(PresetCfg):
    """RSL-RL configuration family keyed by camera pipeline presets."""

    default: Any = config_field(CartpoleCameraPPORunnerCfg())
    resnet18: Any = config_field(CartpoleCameraFeaturePPORunnerCfg())
    theia_tiny: Any = config_field(CartpoleCameraFeaturePPORunnerCfg())
