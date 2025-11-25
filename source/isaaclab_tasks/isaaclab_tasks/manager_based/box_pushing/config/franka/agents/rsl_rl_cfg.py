# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BoxPushingPPORunnerCfg_step(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 100
    max_iterations = 1042
    save_interval = 50
    experiment_name = "step_rl_Fancy_Gym_HP"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=10,
        num_mini_batches=40,
        # learning_rate=5.0e-5,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BoxPushingPPORunnerCfg_bbrl(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 1
    max_iterations = 1042
    save_interval = 50
    experiment_name = "bbrl_Fancy_Gym_HP"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=10,
        num_mini_batches=40,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BoxPushingPPORunnerCfg_mp(RslRlOnPolicyRunnerCfg):
    """RSL-RL config tuned for MP rollouts (aligned with fancy_gym hyperparameters)."""

    seed = 42
    num_steps_per_env = 1
    # 152 envs * 1 step/env * 3290 iters ~= 500k transitions
    max_iterations = 3290
    save_interval = 50
    experiment_name = "mp_proDMP_fancy_gym_hp"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        activation="lrelu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=20,
        num_mini_batches=38,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=1.0,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
