# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import isaaclab_tasks.core.cartpole.mdp.symmetry as symmetry


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole"
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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


@configclass
class CartpoleDirectPPORunnerCfg(CartpolePPORunnerCfg):
    experiment_name = "cartpole_direct"


@configclass
class CartpolePPORunnerWithSymmetryCfg(CartpolePPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
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


@configclass
class CartpoleCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """CNN policy for the raw RGB/depth camera observation pipelines."""

    num_steps_per_env = 16
    max_iterations = 500
    save_interval = 50
    experiment_name = "cartpole_camera"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="relu",
        ),
        hidden_dims=[512],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="relu",
        ),
        hidden_dims=[512],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=True,
    )


@configclass
class CartpoleCameraDirectPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    experiment_name = "cartpole_camera_direct"


@configclass
class CartpoleCameraFeaturePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MLP policy for the pretrained-feature pipelines (ResNet18, Theia-Tiny)."""

    num_steps_per_env = 16
    max_iterations = 200
    save_interval = 50
    experiment_name = "cartpole_features"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
