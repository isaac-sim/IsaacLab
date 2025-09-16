# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class UpdatedRslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticRecurrentCfg):
    fixed_sigma = False
    share_weights = False

@configclass
class UpdatedRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    bounds_loss_coef = 0.0

@configclass
class UR10GearAssemblyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 512
    max_iterations = 1500
    save_interval = 10
    experiment_name = "gear_assembly_ur10e"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class UR10GearAssemblyRNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 7858
    num_steps_per_env = 512
    max_iterations = 1500
    save_interval = 50
    experiment_name = "gear_assembly_ur10e"
    empirical_normalization = True
    clip_actions = 1.0
    resume = False
    policy = UpdatedRslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        noise_std_type="log",
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    algorithm = UpdatedRslRlPpoAlgorithmCfg(
        bounds_loss_coef=0.0001,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
