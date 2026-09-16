# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.core.velocity.mdp.symmetry import anymal


@dataclass
class AnymalCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(24)
    max_iterations: Any = config_field(1500)
    save_interval: Any = config_field(50)
    experiment_name: Any = config_field("anymal_c_rough")
    obs_groups: Any = config_field({"actor": ["policy"], "critic": ["policy"]})
    actor: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
    )
    critic: Any = config_field(
        RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        )
    )
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
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
class AnymalCFlatPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        self.max_iterations = 300
        self.experiment_name = "anymal_c_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]


@dataclass
class AnymalCFlatPPORunnerWithSymmetryCfg(AnymalCFlatPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
            ),
        )
    )


@dataclass
class AnymalCRoughPPORunnerWithSymmetryCfg(AnymalCRoughPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm: Any = config_field(
        RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
            ),
        )
    )
