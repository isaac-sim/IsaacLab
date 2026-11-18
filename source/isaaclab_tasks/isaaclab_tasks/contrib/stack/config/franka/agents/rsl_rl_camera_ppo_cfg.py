# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for RGB-based Franka stacking."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import ALGORITHM_CFG, FrankaStackPPORunnerCfg

CAMERA_POLICY_CFG = RslRlCNNModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=0.2,
        std_range=(0.05, 0.3),
        std_type="log",
    ),
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 32],
        kernel_size=[8, 4, 3],
        stride=[4, 2, 1],
        activation="elu",
    ),
)


@configclass
class FrankaStackCameraPPORunnerCfg(FrankaStackPPORunnerCfg):
    """Use RGB and proprioception for the actor and state for the critic."""

    max_iterations = 15000
    save_interval = 50
    experiment_name = "franka_stack_camera"
    obs_groups = {"actor": ["policy", "base_image"], "critic": ["privileged"]}
    actor = CAMERA_POLICY_CFG
    # Use a smaller learning rate and more samples per mini-batch for the visual encoder.
    algorithm = ALGORITHM_CFG.replace(
        entropy_coef=0.005,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
    )
