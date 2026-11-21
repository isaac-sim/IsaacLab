# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the neural physics backend."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from ..world_model import ModelCfg


@configclass
class NeuralCfg(PhysicsCfg):
    """Neural physics backend, where a world model replaces the physics engine, selected through a task's
    ``physics=`` preset.

    The task declares an articulation and cameras as usual, and the world model supplies their state. The model is
    chosen with :attr:`model_cfg`, as a Newton solver is with ``NewtonCfg.solver_cfg``.
    """

    class_type: type | str = "{DIR}.neural_manager:NeuralManager"

    model_cfg: ModelCfg = MISSING
    """The world model, built by the manager."""

    reset_manifest: str = MISSING
    """JSON file of reset records (``image``, ``state``, ``prompt``, ``split``), see
    :func:`~isaaclab_neural.world_model.load_reset_records`."""

    split: str = "train"
    """Split of the reset records to sample from. ``"train"`` samples at random; other splits are visited in order."""

    envs_per_reset_record: int = 1
    """Consecutive environments that start from the same reset record, ``1`` for independent starts.

    Environment ``i`` belongs to group ``i // envs_per_reset_record``, and each group draws one record. A
    group-relative algorithm such as GRPO sets it to its group size so that a group compares rollouts of one scene;
    other algorithms keep ``1``. Must divide the number of environments."""

    seed: int = 0
    """Seed of the reset sampler and of the per-environment sampling seeds of the model."""
