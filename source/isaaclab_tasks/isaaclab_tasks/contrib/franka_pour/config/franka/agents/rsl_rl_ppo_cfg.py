# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrankaPourResetDatasetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner calibrated for competence-adaptive reset-dataset training."""

    @configclass
    class ExplorationDistributionCfg(RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg):
        """Bounded state-dependent exploration for contact-rich manipulation."""

        # Keep the configured standard-deviation range below the action clamp.
        std_range: tuple[float, float] = (0.15, 0.75)

    # 96 policy steps span 3.2 s at 30 Hz.
    num_steps_per_env = 96
    max_iterations = 3000
    # Reset-dataset learning requires complete episodes for its first outcome cohort.
    init_at_random_ep_len = False
    save_interval = 25
    clip_actions = 1.0
    logger = "tensorboard"
    obs_groups = {"actor": ["policy", "media"], "critic": ["policy", "media", "privileged"]}
    experiment_name = "franka_pour"
    run_name = "reset_dataset"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        # Observations are physically scaled at their source, so empirical normalization is not
        # needed.
        obs_normalization=False,
        distribution_cfg=ExplorationDistributionCfg(
            init_std=0.60,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=3.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        # Use a fixed learning-rate schedule.
        schedule="fixed",
        # Express the discount and GAE time constants at the task's 30 Hz policy rate.
        gamma=0.99 ** (1.0 / 3.0),
        lam=0.95 ** (1.0 / 3.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
