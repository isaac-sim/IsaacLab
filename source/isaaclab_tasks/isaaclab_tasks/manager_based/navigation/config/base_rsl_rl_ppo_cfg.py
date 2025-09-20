# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPerceptiveActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class NavBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "nav_base"
    obs_groups = {
        "policy": ["proprioceptive", "exteroceptive"],
        "critic": ["proprioceptive", "exteroceptive"],
    }
    policy = RslRlPerceptiveActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_cnn_config=RslRlPerceptiveActorCriticCfg.CNNConfig(
            out_channels=[32, 64],
            kernel_size=[(7, 7), (5, 5)],
            flatten=False,
            avg_pool=(1, 1),
            max_pool=(True, False),
        ),
        critic_cnn_config=RslRlPerceptiveActorCriticCfg.CNNConfig(
            out_channels=[32, 64],
            kernel_size=[(7, 7), (5, 5)],
            flatten=False,
            avg_pool=(1, 1),
            max_pool=(True, False),
        ),
        activation="elu",
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
