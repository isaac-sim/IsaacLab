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
    """The algorithm class name. Default is Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use for the student policy."""

    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name. Default is DistillationRunner."""

    student: RslRlMLPModelCfg = MISSING
    """The student configuration."""

    teacher: RslRlMLPModelCfg = MISSING
    """The teacher configuration."""

    algorithm: RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""
