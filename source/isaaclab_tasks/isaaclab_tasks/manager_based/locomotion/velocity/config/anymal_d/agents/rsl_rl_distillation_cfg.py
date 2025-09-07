# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class AnymalDFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 300
    save_interval = 50
    experiment_name = "anymal_d_flat"
    obs_groups = {"policy": ["policy"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )
