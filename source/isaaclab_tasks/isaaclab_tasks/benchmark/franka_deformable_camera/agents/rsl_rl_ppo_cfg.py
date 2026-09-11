# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import ALGO_CFG


@configclass
class FrankaDeformableCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
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


@configclass
class FrankaCableCameraPPORunnerCfg(FrankaDeformableCameraPPORunnerCfg):
    experiment_name = "lift_cable_camera"
