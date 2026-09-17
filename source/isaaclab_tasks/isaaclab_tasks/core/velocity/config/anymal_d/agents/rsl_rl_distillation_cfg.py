# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@dataclass
class AnymalDFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env: Any = 120
    max_iterations: Any = 300
    save_interval: Any = 50
    experiment_name: Any = "anymal_d_flat"
    obs_groups: Any = field(default_factory=lambda: {"student": ["policy"], "teacher": ["policy"]})
    student: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        )
    )
    teacher: Any = field(
        default_factory=lambda: RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
        )
    )
    algorithm: Any = field(
        default_factory=lambda: RslRlDistillationAlgorithmCfg(
            num_learning_epochs=2,
            learning_rate=1.0e-3,
            gradient_length=15,
        )
    )


@dataclass
class AnymalDFlatDistillationRunnerRecurrentCfg(AnymalDFlatDistillationRunnerCfg):
    student: Any = field(
        default_factory=lambda: RslRlRNNModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        )
    )
    teacher: Any = field(
        default_factory=lambda: RslRlRNNModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        )
    )
