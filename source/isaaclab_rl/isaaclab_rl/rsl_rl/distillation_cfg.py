# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg

############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Defaults to Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to. Defaults to None."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use for the student policy. Defaults to adam."""

    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy. Defaults to mse."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name. Defaults to DistillationRunner."""

    student: RslRlMLPModelCfg = MISSING
    """The student configuration."""

    teacher: RslRlMLPModelCfg = MISSING
    """The teacher configuration."""

    algorithm: RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""

    policy: RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `student` and `teacher` model configurations
    instead.
    """


#############################
# Deprecated configurations #
#############################


@configclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = "StudentTeacher"
    """The policy class name. Defaults to StudentTeacher."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the student policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Defaults to scalar."""

    student_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the student network."""

    teacher_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the teacher network."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the student network."""

    teacher_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the teacher network."""

    activation: str = MISSING
    """The activation function for the student and teacher networks."""


@configclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = "StudentTeacherRecurrent"
    """The policy class name. Defaults to StudentTeacherRecurrent."""

    rnn_type: str = MISSING
    """The type of the RNN network. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The hidden dimension of the RNN network."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN network."""

    teacher_recurrent: bool = MISSING
    """Whether the teacher network is recurrent too."""
