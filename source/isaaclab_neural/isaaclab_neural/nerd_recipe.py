# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe (:class:`NeRDRecipe`) that builds the neural robot dynamics model of a task and the data it learns from."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from newton_neural.data import CommandBase, SamplerBase, SimulationBase
from newton_neural.models import FlashNeRD, NeRDBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg

    from .nerd_recipe_cfg import NeRDRecipeCfg


class RecordingSimulation(SimulationBase):
    """The environment that records the data of a model, as a recipe sees it.

    ``scripts/neural/record.py`` implements it on a base environment with the scene and the simulation of the task.
    Besides the members of ``newton_neural``'s ``SimulationBase``, such as ``system``, ``model``, ``device`` and
    ``state_bounds()``, it holds what a recipe needs from Isaac Lab.
    """

    env: ManagerBasedEnv
    """Environment that is recorded."""

    asset: Articulation
    """The robot."""

    articulation: list[int]
    """Newton articulation of every robot, which contact inputs take."""

    default_joint_pos: torch.Tensor
    """Default joint positions of the robot ``[actuated_count]`` [m or rad], in the joint order of the system."""


class NeRDRecipe:
    """Builds the parts of a model that are code, from the ``newton-neural`` library.

    The defaults suit a robot in free space. A task subclasses the recipe, overrides what its robot needs, and names
    the subclass as ``class_type`` of its :class:`~isaaclab_neural.NeRDRecipeCfg`.

    Args:
        cfg: Settings of the model.
    """

    cfg: NeRDRecipeCfg

    def __init__(self, cfg: NeRDRecipeCfg):
        self.cfg = cfg

    def configure(self, env_cfg: ManagerBasedEnvCfg) -> None:
        """Change the resolved task configuration ``env_cfg`` in place before the data is recorded.

        The default leaves the task as it is. Override it for what only the recording needs, such as removing an
        object that the robot would hit under random joint targets. Randomization that the model does not see is
        named in :attr:`~isaaclab_neural.NeRDRecipeCfg.unseen_events`, which holds for the evaluation as well.
        """

    def model(self, simulation: RecordingSimulation) -> NeRDBase:
        """Build the untrained model. The default is a ``FlashNeRD`` of the robot without a contact input."""
        return FlashNeRD(simulation.system)

    def start_states(self, simulation: RecordingSimulation) -> SamplerBase | None:
        """Build the sampler of the states that episodes start from, or ``None`` for the recorder's default."""
        return None

    def joint_targets(self, simulation: RecordingSimulation) -> CommandBase | None:
        """Build the generator of joint position targets, or ``None`` for the recorder's default."""
        return None
