# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class TrackPositionNoObstaclesEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = 24
    max_iterations: Any = 1500
    save_interval: Any = 50
    experiment_name: Any = "arl_robot_1_track_position_state_based"
    actor: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        )
    )
    critic: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=False,
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=4.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
