# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab.utils import replace_config

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg


@dataclass
class ShadowHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = 16
    max_iterations: Any = 3000
    save_interval: Any = 250
    experiment_name: Any = "shadow_hand"
    actor: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[512, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[512, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.016,
            max_grad_norm=1.0,
        )
    )


@dataclass
class ShadowHandManagerPPORunnerCfg(PresetCfg):
    """``presets=asymmetric`` feeds the critic the privileged observation group."""

    default: Any = field(default_factory=ShadowHandPPORunnerCfg)
    asymmetric: Any = field(
        default_factory=lambda: replace_config(
            ShadowHandPPORunnerCfg(), obs_groups={"actor": ["policy"], "critic": ["critic"]}
        )
    )


@dataclass
class ShadowHandCameraFFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = 64
    max_iterations: Any = 5000
    save_interval: Any = 250
    experiment_name: Any = "shadow_hand_camera"
    obs_groups: Any = field(default_factory=lambda: {"actor": ["policy"], "critic": ["critic"]})
    actor: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[1024, 512, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[1024, 512, 512, 256, 128],
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
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
