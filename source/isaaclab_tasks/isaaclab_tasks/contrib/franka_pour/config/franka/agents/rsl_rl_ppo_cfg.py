# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaPourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_pour"
    run_name = "pour"
    clip_actions = 1.0
    # W&B uses the active CLI login (or WANDB_API_KEY); set WANDB_MODE=offline
    # to keep run data local without changing the task config.
    logger = "wandb"
    wandb_project = "franka-pour-mpm"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        # Curriculum stages intentionally change reset offsets and geometry. Running empirical
        # normalization makes the first unseen stage arbitrarily out-of-distribution because
        # stage-constant features have zero variance. Observations are physically scaled in the
        # environment instead, matching the standard Franka Lift policy setup.
        obs_normalization=False,
        # Relative joint increments and the continuous gripper command already map normalized
        # actions into bounded physical ranges. Use the standard state-independent Gaussian with
        # modest initial exploration around the contact-sensitive backward resets.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Retain modest exploration when the curriculum first exposes unseen reset geometry.
        entropy_coef=1.0e-3,
        # Match Isaac Lab's Franka manipulation update depth. The earlier one-fifth-rate,
        # two-epoch update left the actor statistically unchanged after millions of transitions.
        num_learning_epochs=5,
        num_mini_batches=4,
        # The contact-sensitive, low-noise policy produces very small KL early in the backward
        # curriculum. Keep the standard Franka base rate fixed so an adaptive schedule cannot
        # amplify it during those initially short episodes.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FrankaPourResetDatasetPPORunnerCfg(FrankaPourPPORunnerCfg):
    """PPO runner calibrated for adaptive reset-dataset training."""

    @configclass
    class ExplorationDistributionCfg(RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg):
        """State-dependent exploration bounded for independently sampled action noise."""

        std_range: tuple[float, float] = (0.05, 0.75)

    # The environment runs at 30 Hz, so 32 transitions span a 1.067-second rollout.
    num_steps_per_env = 32
    save_interval = 25
    # Keep the 8-action relative-joint policy separate from incompatible Cartesian-IK checkpoints.
    experiment_name = "franka_pour_reset_dataset_joint_rel"
    run_name = "reset_dataset_joint_rel"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        # The adaptive cache deliberately changes its reset support as competence grows. Running
        # statistics would therefore move the policy's coordinate system at every frontier
        # expansion and can destroy behavior on previously mastered rows. The observations are
        # physically scaled by the environment, so keep their representation stationary.
        obs_normalization=False,
        # Mainline RSL-RL has no temporally correlated gSDE. Retain state-dependent exploration,
        # but use the standard Isaac Lab entropy scale below so independently sampled noise does
        # not grow merely because the environment clips actions.
        distribution_cfg=ExplorationDistributionCfg(
            # Start well inside the learned range. Initializing exactly at the former upper clamp
            # pinned both standard deviation and entropy in the failed joint-relative runs.
            init_std=0.25,
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
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        # The stopped run's small early KL let the adaptive schedule raise this to 1.14e-3,
        # coincident with the first curriculum expansion. Keep update size stationary while the
        # reset distribution itself changes.
        schedule="fixed",
        # Preserve the 10 Hz configuration's physical discount and GAE time constants after
        # increasing the policy rate by three.
        gamma=0.99 ** (1.0 / 3.0),
        lam=0.95 ** (1.0 / 3.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Deprecated compatibility name; use ``FrankaPourResetDatasetPPORunnerCfg``.
FrankaPourResetMixturePPORunnerCfg = FrankaPourResetDatasetPPORunnerCfg
