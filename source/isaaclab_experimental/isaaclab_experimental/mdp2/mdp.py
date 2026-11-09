# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled MDP and the data it runs on."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import warp as wp

from isaaclab.utils.warp import ProxyArray

from .config import FieldSelector
from .scene import SceneData, SceneSchema


class Backend(enum.Enum):
    """How term implementations are executed."""

    TORCH = "torch"
    WARP = "warp"


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
    truncation: bool = False
    group: str = ""
    columns: slice = slice(None)
    shape: tuple[int, ...] = ()
    """Output dimensions after the world dimension."""


@dataclass(frozen=True)
class TermBuffers:
    """Stacked outputs of the reward and termination terms, one row per op, for the Warp backend."""

    terminations: wp.array
    """Shape ``(num_termination_ops, num_worlds)``."""
    truncation: wp.array
    rewards: wp.array
    """Shape ``(num_reward_ops, num_worlds)``."""
    reward_scales: wp.array
    """``weight * step_dt`` of each reward op."""


@dataclass
class MDPState:
    """Data that changes while an MDP runs on a scene."""

    episode_length: ProxyArray
    """Control steps since the last reset, shape ``(num_worlds,)``."""
    actions: ProxyArray
    """Last actions, shape ``(num_worlds, action_dim)``."""
    views: dict[FieldSelector, ProxyArray]
    """Scene data views of every selected field."""
    terms: TermBuffers | None = None
    """Output buffers of the Warp backend's reward and termination terms."""


@dataclass
class Transition:
    """Results of one step, owned by the caller. Rewards and flags have shape ``(num_worlds,)``."""

    rewards: ProxyArray
    terminated: ProxyArray
    truncated: ProxyArray
    observations: dict[str, ProxyArray]
    """Buffers by observation group, each of shape ``(num_worlds, group_dim)``."""


@dataclass(frozen=True)
class MDP:
    """An immutable, compiled MDP. Create it with :meth:`MDPBuilder.finalize`."""

    schema: SceneSchema
    device: str
    backend: Backend
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

    @property
    def selectors(self) -> tuple[FieldSelector, ...]:
        """Every selected scene field, in schedule order, without repeats."""
        selectors = [op.target for op in self.action_ops]
        for op in self.termination_ops + self.reward_ops + self.observation_ops:
            selectors.extend(op.inputs.values())
        return tuple(dict.fromkeys(selectors))

    def state(self, scene_data: SceneData) -> MDPState:
        """Create the runtime state, bound to the views of a scene with the same schema."""
        if scene_data.schema != self.schema:
            raise ValueError("The scene schema does not match the schema this MDP was built against.")
        return MDPState(
            episode_length=ProxyArray(wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)),
            actions=ProxyArray(wp.zeros((self.num_worlds, self.action_dim), dtype=wp.float32, device=self.device)),
            views={selector: scene_data.data(selector.entity, selector.field) for selector in self.selectors},
            terms=self._term_buffers() if self.backend is Backend.WARP else None,
        )

    def observations(self) -> dict[str, ProxyArray]:
        return {
            group: ProxyArray(wp.zeros((self.num_worlds, dim), dtype=wp.float32, device=self.device))
            for group, dim in self.observation_dims.items()
        }

    def transition(self) -> Transition:
        return Transition(
            rewards=ProxyArray(wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)),
            terminated=ProxyArray(wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)),
            truncated=ProxyArray(wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)),
            observations=self.observations(),
        )

    def _term_buffers(self) -> TermBuffers:
        truncation = [op.truncation for op in self.termination_ops]
        reward_scales = [op.weight * self.step_dt for op in self.reward_ops]
        return TermBuffers(
            terminations=wp.zeros((len(truncation), self.num_worlds), dtype=wp.bool, device=self.device),
            truncation=wp.array(truncation, dtype=wp.bool, shape=len(truncation), device=self.device),
            rewards=wp.zeros((len(reward_scales), self.num_worlds), dtype=wp.float32, device=self.device),
            reward_scales=wp.array(reward_scales, dtype=wp.float32, shape=len(reward_scales), device=self.device),
        )
