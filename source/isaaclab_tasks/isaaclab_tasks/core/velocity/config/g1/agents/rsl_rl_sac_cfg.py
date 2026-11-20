# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC settings for the G1 velocity-tracking tasks.

Settings and iteration counts come from tuning FlashSAC against PPO under the same rewards.
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

from .symmetry_cfg import G1SymmetryCfg


@configclass
class G1FlatSACRunnerCfg(RslRlSacRunnerCfg):
    """FlashSAC runner configuration for flat G1 velocity tracking.

    Uses the FlashSAC defaults of RSL-RL with a higher discount, more updates per iteration, a cosine learning-rate
    schedule and a short warm-up.
    """

    num_steps_per_env = 8
    max_iterations = 6600
    save_interval = 500
    experiment_name = "g1_flat_sac"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlResidualMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        network_cfg=RslRlResidualMLPCfg(hidden_dim=128, num_blocks=2),
        distribution_cfg=RslRlMLPModelCfg.SquashedGaussianDistributionCfg(action_range=(-1.0, 1.0)),
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
        # without mirrored samples SAC learns left/right asymmetric gaits on G1, e.g. one leg held in the air
        symmetry_cfg=G1SymmetryCfg(),
    )


@configclass
class G1RoughSACRunnerCfg(RslRlSacRunnerCfg):
    """FlashSAC runner configuration for rough G1 velocity tracking.

    Uses the FlashSAC defaults of RSL-RL with higher learning rates and a short warm-up. Tuned with 1024
    environments (``--num_envs 1024``).
    """

    num_steps_per_env = 8
    max_iterations = 16000
    save_interval = 500
    experiment_name = "g1_rough_sac"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlResidualMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        network_cfg=RslRlResidualMLPCfg(hidden_dim=128, num_blocks=2),
        distribution_cfg=RslRlMLPModelCfg.SquashedGaussianDistributionCfg(action_range=(-1.0, 1.0)),
    )
    critic = RslRlQMLPModelCfg(
        hidden_dims=[],
        activation="relu",
        network_cfg=RslRlResidualMLPCfg(hidden_dim=256, num_blocks=2),
    )
    algorithm = RslRlSacAlgorithmCfg(
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
        num_warmup_transitions=8192,
        use_cuda_graph=True,
        # without mirrored samples SAC learns left/right asymmetric gaits on G1, e.g. one leg held in the air
        symmetry_cfg=G1SymmetryCfg(),
    )
