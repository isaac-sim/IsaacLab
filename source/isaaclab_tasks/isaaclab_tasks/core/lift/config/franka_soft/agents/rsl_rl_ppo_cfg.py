# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
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


@configclass
class FrankaDeformablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "franka_deformable"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = ALGO_CFG


@configclass
class FrankaDeformableCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "franka_deformable_camera"
    obs_groups = {
        "actor": ["policy", "proprio", "base_image"],
        "critic": ["policy", "proprio", "perception"],
    }
    actor = RslRlCNNModelCfg(
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
    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ALGO_CFG.replace(num_mini_batches=8)
