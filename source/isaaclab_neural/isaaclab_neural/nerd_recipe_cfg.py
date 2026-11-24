# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings (:class:`NeRDRecipeCfg`) that say how the neural robot dynamics model of a task is recorded and trained."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .nerd_recipe import NeRDRecipe


@configclass
class NeRDRecordCfg:
    """How much data is recorded for a model."""

    num_envs: int = 4096
    """Robots recorded in parallel."""

    num_frames: int = 2048
    """Frames recorded per robot."""

    episode_frames: tuple[int, int] = (64, 128)
    """Shortest and longest episode [frames]. A robot restarts from a newly sampled state at the end of each."""


@configclass
class NeRDTrainCfg:
    """How a model is fitted to its data."""

    max_steps: int = 20_000
    """Optimizer steps."""

    window: int = 16
    """Consecutive frames per training sample, or ``0`` for single frames."""

    batch_size: int = 1024
    """Training samples per optimizer step."""

    learning_rate: float = 2.0e-3
    """Peak learning rate, annealed to zero along a cosine."""

    noise: float = 0.1
    """Noise added to the input state [standard deviations of its change over one frame]."""

    horizons: tuple[int, ...] = (1, 10)
    """Lengths of the open-loop rollouts whose error is reported after training [frames]."""


@configclass
class NeRDRecipeCfg:
    """Settings of the neural robot dynamics model of a task, read by the scripts under ``scripts/neural``.

    A task registers a subclass as ``nerd_recipe_cfg_entry_point``, next to its agent configurations, with the numbers
    of its model as fields. :attr:`class_type` names the :class:`~isaaclab_neural.NeRDRecipe` that builds the parts
    that are code.
    """

    class_type: type[NeRDRecipe] | str = "{DIR}.nerd_recipe:NeRDRecipe"
    """Recipe that builds the model, its start states and its joint targets from these settings."""

    physics_preset: str = "newton_nerd"
    """Physics preset of the task that runs the model. The trained model is saved where that preset loads it from."""

    reference_preset: str = "newton_mjwarp"
    """Physics preset of the task that the model learns from."""

    robot: str = "robot"
    """Name of the robot in the scene of the task."""

    unseen_events: tuple[str, ...] = ()
    """Events of the task that randomize what the model does not see, such as the mass of the robot, by their names
    in the event settings of the task. They are off when the data is recorded and when policies are evaluated, so
    that both run the robot that the model stands for."""

    record: NeRDRecordCfg = NeRDRecordCfg()
    """Amount of data."""

    train: NeRDTrainCfg = NeRDTrainCfg()
    """Training settings."""
