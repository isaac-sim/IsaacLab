# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configurations for state and camera-based Franka stacking."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

EXPLORATION_CFG = RslRlMLPModelCfg.GaussianDistributionCfg(
    init_std=0.2,
    std_range=(0.05, 0.3),
    std_type="log",
)

STATE_POLICY_CFG = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=EXPLORATION_CFG,
)

STATE_CRITIC_CFG = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
)

CAMERA_POLICY_CFG = RslRlCNNModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=EXPLORATION_CFG,
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 32],
        kernel_size=[8, 4, 3],
        stride=[4, 2, 1],
        activation="elu",
    ),
)

ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.001,
    num_learning_epochs=5,
    num_mini_batches=16,
    learning_rate=1.0e-4,
    schedule="fixed",
    # At 50 Hz, the full two-pick sequence needs a longer effective horizon
    # than the manipulation-task default.
    gamma=0.999,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class FrankaStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for order-invariant three-cube Franka stacking."""

    num_steps_per_env = 32
    # Reset-table curricula need a complete first outcome for every row.
    init_at_random_ep_len = False
    max_iterations = 7000
    save_interval = 25
    experiment_name = "franka_stack"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = STATE_POLICY_CFG
    critic = STATE_CRITIC_CFG
    algorithm = ALGORITHM_CFG


@configclass
class FrankaStackCameraPPORunnerCfg(FrankaStackPPORunnerCfg):
    """Asymmetric PPO configuration for RGB-based Franka stacking."""

    max_iterations = 15000
    save_interval = 50
    experiment_name = "franka_stack_camera"
    obs_groups = {
        "actor": ["policy", "base_image"],
        "critic": ["privileged"],
    }
    actor = CAMERA_POLICY_CFG
    # Match the maintained KUKA camera policy's conservative encoder update.
    algorithm = ALGORITHM_CFG.replace(
        entropy_coef=0.005,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
    )
