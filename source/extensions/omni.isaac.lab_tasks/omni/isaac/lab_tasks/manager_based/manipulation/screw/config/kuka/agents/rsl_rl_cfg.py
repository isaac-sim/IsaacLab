# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class KukaScrewPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1600
    save_interval = 200
    experiment_name = "kuka_screw"
    run_name = ""
    resume = False
    logger = "wandb"
    wandb_project = "kuka_screw"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        # actor_hidden_dims=[256, 128, 64],
        # critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=10,
        num_mini_batches=8,  # default
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
    # automate
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=2.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.0,
    #     num_learning_epochs=6,
    #     num_mini_batches=16,
    #     learning_rate=1.0e-3,
    #     schedule="fixed",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.016,
    #     max_grad_norm=1.0,
    # )
