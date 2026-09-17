# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
========================================= IMPORTANT NOTICE =========================================

This file defines the agent configuration used to generate the "Training Performance" table in
https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html.
Ensure that the configurations for the other RL libraries are updated if this one is modified.

====================================================================================================
"""

from dataclasses import dataclass, field
from typing import Any

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = 32
    max_iterations: Any = 1000
    save_interval: Any = 100
    experiment_name: Any = "humanoid"
    actor: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[400, 200, 100],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[400, 200, 100],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=2.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )


@dataclass
class HumanoidDirectPPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name: Any = "humanoid_direct"
