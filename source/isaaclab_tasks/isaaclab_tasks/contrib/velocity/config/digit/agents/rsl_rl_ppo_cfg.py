# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class DigitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = 24
    max_iterations: Any = 3000
    save_interval: Any = 50
    experiment_name: Any = "digit_rough"
    obs_groups: Any = field(default_factory=lambda: {"actor": ["policy"], "critic": ["policy"]})
    actor: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # MJWarp needs a lower entropy floor than PhysX here (0.005 vs the usual 0.01) -- see #7520.
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )


@dataclass
class DigitFlatPPORunnerCfg(DigitRoughPPORunnerCfg):
    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        self.max_iterations = 2000
        self.experiment_name = "digit_flat"

        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]
