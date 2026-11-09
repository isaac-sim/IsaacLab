# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled MDP and the data it runs on."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .config import FieldSelector
from .scene import SceneData, SceneSchema

Observations = dict[str, torch.Tensor]
"""Observation buffers by group, each of shape ``(num_worlds, group_dim)``."""


@dataclass(frozen=True)
class ActionOp:
    """Writes a slice of the action to an entity field."""

    name: str
    target: FieldSelector
    columns: slice
    scale: float
    offset: float


@dataclass(frozen=True)
class TermOp:
    """One term use, compiled for a backend."""

    name: str
    compute: Callable
    inputs: Mapping[str, FieldSelector]
    params: Mapping[str, Any]
    weight: float = 1.0
    """Reward weight."""
    truncation: bool = False
    """Whether a termination is a truncation."""
    group: str = ""
    """Observation group."""
    columns: slice = slice(None)
    """Columns of the observation group."""


@dataclass
class MDPState:
    """Data that changes while an MDP runs on a scene."""

    episode_length: torch.Tensor
    """Control steps since the last reset, shape ``(num_worlds,)``."""
    actions: torch.Tensor
    """Last actions, shape ``(num_worlds, action_dim)``."""
    bindings: dict[str, dict[str, torch.Tensor]]
    """Scene data views bound to each op's inputs, by op name."""


@dataclass
class Transition:
    """Results of one step, owned by the caller. Rewards and flags have shape ``(num_worlds,)``."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    observations: Observations


@dataclass(frozen=True)
class MDP:
    """An immutable, compiled MDP. Create it with :meth:`MDPBuilder.finalize`."""

    schema: SceneSchema
    device: str
    backend: str
    action_ops: tuple[ActionOp, ...]
    termination_ops: tuple[TermOp, ...]
    reward_ops: tuple[TermOp, ...]
    observation_ops: tuple[TermOp, ...]
    action_dim: int
    observation_dims: Mapping[str, int]
    max_episode_length: int | None
    """Control steps until truncation, if the task has a time limit."""

    @property
    def num_worlds(self) -> int:
        return self.schema.num_worlds

    @property
    def step_dt(self) -> float:
        return self.schema.step_dt

    @property
    def schedule(self) -> tuple[str, ...]:
        """Op names in execution order."""
        ops = self.action_ops + self.termination_ops + self.reward_ops + self.observation_ops
        return tuple(op.name for op in ops)

    def state(self, scene_data: SceneData) -> MDPState:
        """Create the runtime state, bound to the views of a scene with the same schema."""
        if scene_data.schema != self.schema:
            raise ValueError("The scene schema does not match the schema this MDP was built against.")
        bindings = {op.name: {"target": scene_data.data(op.target.entity, op.target.field)} for op in self.action_ops}
        for op in self.termination_ops + self.reward_ops + self.observation_ops:
            bindings[op.name] = {arg: scene_data.data(sel.entity, sel.field) for arg, sel in op.inputs.items()}
        return MDPState(
            episode_length=torch.zeros(self.num_worlds, dtype=torch.long, device=self.device),
            actions=torch.zeros(self.num_worlds, self.action_dim, device=self.device),
            bindings=bindings,
        )

    def observations(self) -> Observations:
        """Allocate observation buffers."""
        return {
            group: torch.zeros(self.num_worlds, dim, device=self.device) for group, dim in self.observation_dims.items()
        }

    def transition(self) -> Transition:
        """Allocate a transition."""
        flags = dict(size=(self.num_worlds,), dtype=torch.bool, device=self.device)
        return Transition(
            rewards=torch.zeros(self.num_worlds, device=self.device),
            terminated=torch.zeros(**flags),
            truncated=torch.zeros(**flags),
            observations=self.observations(),
        )
