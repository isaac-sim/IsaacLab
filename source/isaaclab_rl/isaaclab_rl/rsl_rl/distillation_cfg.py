# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

#########################
# Policy configurations #
#########################


@configclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks."""

    class_name: str = "StudentTeacher"
    """The policy class name. Default is StudentTeacher."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the student policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the student network."""

    teacher_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the teacher network."""

    activation: str = MISSING
    """The activation function for the student and teacher networks."""


@configclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks."""

    class_name: str = "StudentTeacherRecurrent"
    """The policy class name. Default is StudentTeacherRecurrent."""

    rnn_type: str = MISSING
    """The type of the RNN network. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The hidden dimension of the RNN network."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN network."""

    teacher_recurrent: bool = MISSING
    """Whether the teacher network is recurrent too."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Default is Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""
