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
        """State-dependent exploration calibrated by the successful reset-dataset run."""

        # Independent noise is resampled at every 30 Hz policy step. Cap the distribution below the
        # action clamp: with a 0.03-rad joint scale, 0.75 already provides substantially more
        # physical exploration than the earlier 0.015-rad diagnostic.
        std_range: tuple[float, float] = (0.15, 0.75)

    # Match the successful pouring setup's 3.2-second physical rollout at the current 30 Hz policy
    # rate. This lets grasp acquisition and transport influence one advantage estimate.
    num_steps_per_env = 96
    max_iterations = 3000
    # Reset-dataset learning requires complete episodes for its first outcome cohort.
    init_at_random_ep_len = False
    save_interval = 25
    clip_actions = 1.0
    logger = "tensorboard"
    obs_groups = {"actor": ["policy", "media"], "critic": ["policy", "media", "privileged"]}
    # Keep the 8-action relative-joint policy separate from incompatible Cartesian checkpoints.
    experiment_name = "franka_pour_reset_dataset_particle_state_joint_rel"
    run_name = "reset_dataset_particle_state_joint_rel"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        # Observations are physically scaled at their source, so empirical normalization is not
        # needed.
        obs_normalization=False,
        # Mainline RSL-RL has no temporally correlated gSDE. Start above the prematurely collapsed
        # 0.35 diagnostic while retaining a strict bound below the action clamp.
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
        # The 1e-4 run collapsed below the useful 0.20--0.30 exploration band before pre-grasp
        # acquisition, while 1e-3 drove std to its upper cap. The geometric midpoint retains
        # contact-scale exploration without destabilizing already learned transport and pouring.
        entropy_coef=3.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Keep the bootstrap checkpoint's optimizer rate. RSL-RL restores optimizer state on
        # resume, so changing only this field would make the configured and effective rates differ.
        learning_rate=1.5e-4,
        # The stopped run's small early KL let the adaptive learning-rate schedule reach 1.14e-3,
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
