# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC settings for the existing flat G1 velocity-tracking MDP.

The action bound 2 follows the FlashSAC single-setting benchmark.
"""

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
class G1FlatSACRunnerCfg(RslRlSacRunnerCfg):
    """FlashSAC runner configuration for flat G1 velocity tracking."""

    num_steps_per_env = 8
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_flat_sac"
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlResidualMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        obs_normalization=False,
        network_cfg=RslRlResidualMLPCfg(hidden_dim=128, num_blocks=2, expansion=4, bn_momentum=0.01),
        distribution_cfg=RslRlMLPModelCfg.SquashedGaussianDistributionCfg(
            action_range=(-2.0, 2.0), log_std_range=(-10.0, 2.0)
        ),
    )
    critic = RslRlQMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        obs_normalization=False,
        network_cfg=RslRlResidualMLPCfg(hidden_dim=256, num_blocks=2, expansion=4, bn_momentum=0.01),
    )
    algorithm = RslRlSacAlgorithmCfg(
        num_critics=2,
        replay_buffer_size=10_000_000,
        num_updates=64,
        batch_size=2048,
        n_steps=3,
        gamma=0.99,
        tau=0.01,
        actor_update_interval=2,
        learning_starts=8192,
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
        optimizer="adam",
        weight_decay=0.0,
        max_grad_norm=None,
        init_alpha=0.01,
        target_entropy_std=0.15,
        critic_loss="categorical",
        num_bins=101,
        value_range=(-5.0, 5.0),
        reward_normalization=True,
        reward_normalization_max_return=5.0,
        exploration="held",
        noise_repeat_exponent=2.0,
        noise_repeat_max=16,
        learning_rate_schedule="cosine",
        final_learning_rate_ratio=0.5,
        schedule_updates=96_000,
        use_mixed_precision=False,
        symmetry_cfg=None,
        use_cuda_graph=True,
    )
