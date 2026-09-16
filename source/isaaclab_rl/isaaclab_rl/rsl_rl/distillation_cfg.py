# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Literal

from isaaclab.utils import config_field

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg

############################
# Algorithm configurations #
############################


@dataclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = config_field("Distillation")
    """The algorithm class name. Defaults to Distillation."""

    num_learning_epochs: int = config_field(MISSING)
    """The number of updates performed with each sample."""

    learning_rate: float = config_field(MISSING)
    """The learning rate for the student policy."""

    gradient_length: int = config_field(MISSING)
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = config_field(None)
    """The maximum norm the gradient is clipped to. Defaults to None."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = config_field("adam")
    """The optimizer to use for the student policy. Defaults to adam."""

    loss_type: Literal["mse", "huber"] = config_field("mse")
    """The loss type to use for the student policy. Defaults to mse."""


#########################
# Runner configurations #
#########################


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = config_field("DistillationRunner")
    """The runner class name. Defaults to DistillationRunner."""

    student: RslRlMLPModelCfg = config_field(MISSING)
    """The student configuration."""

    teacher: RslRlMLPModelCfg = config_field(MISSING)
    """The teacher configuration."""

    algorithm: RslRlDistillationAlgorithmCfg = config_field(MISSING)
    """The algorithm configuration."""

    policy: RslRlDistillationStudentTeacherCfg = config_field(MISSING)
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `student` and `teacher` model configurations
    instead.
    """


#############################
# Deprecated configurations #
#############################


@dataclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = config_field("StudentTeacher")
    """The policy class name. Defaults to StudentTeacher."""

    init_noise_std: float = config_field(MISSING)
    """The initial noise standard deviation for the student policy."""

    noise_std_type: Literal["scalar", "log"] = config_field("scalar")
    """The type of noise standard deviation for the policy. Defaults to scalar."""

    student_obs_normalization: bool = config_field(MISSING)
    """Whether to normalize the observation for the student network."""

    teacher_obs_normalization: bool = config_field(MISSING)
    """Whether to normalize the observation for the teacher network."""

    student_hidden_dims: list[int] = config_field(MISSING)
    """The hidden dimensions of the student network."""

    teacher_hidden_dims: list[int] = config_field(MISSING)
    """The hidden dimensions of the teacher network."""

    activation: str = config_field(MISSING)
    """The activation function for the student and teacher networks."""


@dataclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = config_field("StudentTeacherRecurrent")
    """The policy class name. Defaults to StudentTeacherRecurrent."""

    rnn_type: str = config_field(MISSING)
    """The type of the RNN network. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = config_field(MISSING)
    """The hidden dimension of the RNN network."""

    rnn_num_layers: int = config_field(MISSING)
    """The number of layers of the RNN network."""

    teacher_recurrent: bool = config_field(MISSING)
    """Whether the teacher network is recurrent too."""
