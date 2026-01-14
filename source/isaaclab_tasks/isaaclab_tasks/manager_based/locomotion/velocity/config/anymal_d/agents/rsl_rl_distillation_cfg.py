# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@configclass
class AnymalDFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 300
    save_interval = 50
    experiment_name = "anymal_d_flat"
    obs_groups = {"policy": ["policy"], "teacher": ["policy"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=0.1,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=0.0,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class AnymalDFlatDistillationRunnerRecurrentCfg(AnymalDFlatDistillationRunnerCfg):
    student = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=0.1,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    teacher = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=0.0,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
