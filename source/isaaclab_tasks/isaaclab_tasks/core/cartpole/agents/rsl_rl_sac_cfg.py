# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC settings for the manager-based cartpole task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlQMLPModelCfg,
    RslRlResidualMLPCfg,
    RslRlResidualMLPModelCfg,
    RslRlSacAlgorithmCfg,
    RslRlSacRunnerCfg,
)


@configclass
class CartpoleSACRunnerCfg(RslRlSacRunnerCfg):
    """FlashSAC runner configuration for the cartpole."""

    num_steps_per_env = 8
    max_iterations = 2500
    save_interval = 500
    experiment_name = "cartpole_sac"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlResidualMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        network_cfg=RslRlResidualMLPCfg(hidden_dim=128, num_blocks=2),
        distribution_cfg=RslRlMLPModelCfg.SquashedGaussianDistributionCfg(action_range=(-2.0, 2.0)),
    )
    critic = RslRlQMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        network_cfg=RslRlResidualMLPCfg(hidden_dim=256, num_blocks=2),
    )
    algorithm = RslRlSacAlgorithmCfg(
        gamma=0.99,
        num_updates=64,
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
        learning_rate_schedule="cosine",
        final_learning_rate_ratio=0.5,
        num_schedule_updates=96_000,
        num_warmup_transitions=8192,
        use_cuda_graph=True,
    )
