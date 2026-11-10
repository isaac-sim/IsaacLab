# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled MDP and the data it runs on."""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import warp as wp

from isaaclab.utils.warp import ProxyArray

from . import rng
from .config import FieldSelector
from .scene import SceneData, SceneSchema
from .terms import State


class Backend(enum.Enum):
    """How term implementations are executed."""

    TORCH = "torch"
    WARP = "warp"


@dataclass(frozen=True)
class ActionOp:
    """Writes a slice of the action to an entity field."""

    name: str
    target: FieldSelector
    columns: slice  # Columns of the action
    scale: float
    offset: float


@dataclass(frozen=True)
class TermOp:
    """One term use, compiled for a backend."""

    name: str
    compute: Callable
    update: Callable | None
    reset: Callable | None
    inputs: Mapping[str, FieldSelector]
    params: Mapping[str, Any]
    state: Mapping[str, State]  # Shapes resolved to sizes, including the world dimension
    weight: float = 1.0  # Rewards only
    truncation: bool = False  # Terminations only
    group: str = ""  # Observations only
    columns: slice = slice(None)  # Columns of the observation group
    target: FieldSelector | None = None  # Commands only
    shape: tuple[int, ...] = ()  # Output dimensions after the world dimension
    timer: tuple[float, float] | None = None  # Seconds between command resamples or interval events
    stream: int | None = None  # Row of MDPState.rng, for timers and random terms
    random: bool = False  # Whether the term draws from its stream


@dataclass(frozen=True)
class TermOutputs:
    """Stacked outputs of the reward and termination terms, one row per op, for the Warp backend."""

    terminations: wp.array  # (num_termination_ops, num_worlds)
    is_truncation: wp.array  # (num_termination_ops,)
    rewards: wp.array  # (num_reward_ops, num_worlds)
    reward_scales: wp.array  # (num_reward_ops,), weight * step_dt
    observations: dict[str, wp.array]  # Scratch for observe(), (num_worlds, group_dim) per group
    commands: dict[str, wp.array]  # Scratch for samples, (num_worlds, command_dim) per command op
    due: dict[str, wp.array]  # (num_worlds,) per timed op, whose timer expired this step


@dataclass
class MDPState:
    """Data that changes while an MDP runs on a scene."""

    episode_length: ProxyArray  # (num_worlds,), control steps since the last reset
    actions: ProxyArray  # (num_worlds, action_dim), last actions
    fields: dict[FieldSelector, ProxyArray]  # Scene data of every selected field
    term_state: dict[str, dict[str, ProxyArray]]  # By op name, then state name
    rng: ProxyArray  # (num_streams, num_worlds), see rng.py
    timers: dict[str, ProxyArray]  # (num_worlds,) seconds left per timed op
    outputs: TermOutputs | None = None  # Warp backend only


@dataclass
class Transition:
    """Results of one step, owned by the caller."""

    rewards: ProxyArray  # (num_worlds,)
    terminated: ProxyArray  # (num_worlds,)
    truncated: ProxyArray  # (num_worlds,)
    observations: dict[str, ProxyArray]  # Successor observations before resets, (num_worlds, group_dim) per group


@dataclass(frozen=True)
class MDP:
    """An immutable, compiled MDP. Create it with :meth:`MDPBuilder.finalize`."""

    schema: SceneSchema
    device: str
    backend: Backend
    action_ops: tuple[ActionOp, ...]
    termination_ops: tuple[TermOp, ...]
    reward_ops: tuple[TermOp, ...]
    command_ops: tuple[TermOp, ...]
    interval_event_ops: tuple[TermOp, ...]
    reset_event_ops: tuple[TermOp, ...]
    observation_ops: tuple[TermOp, ...]
    action_dim: int
    observation_dims: Mapping[str, int]
    max_episode_length: int | None  # Control steps until truncation, None without a time limit
    streams: tuple[str, ...]  # Op names of the rows of MDPState.rng

    @property
    def num_worlds(self) -> int:
        return self.schema.num_worlds

    @property
    def step_dt(self) -> float:
        return self.schema.step_dt

    @property
    def term_ops(self) -> tuple[TermOp, ...]:
        """Every term op, in schedule order."""
        return (
            self.termination_ops
            + self.reward_ops
            + self.command_ops
            + self.interval_event_ops
            + self.reset_event_ops
            + self.observation_ops
        )

    @property
    def schedule(self) -> tuple[str, ...]:
        """Op names in schedule order."""
        return tuple(op.name for op in self.action_ops + self.term_ops)

    @property
    def selectors(self) -> tuple[FieldSelector, ...]:
        """Every selected scene field, in schedule order, without repeats."""
        selectors = [op.target for op in self.action_ops]
        for op in self.term_ops:
            selectors.extend(op.inputs.values())
        return tuple(dict.fromkeys(selectors))

    def state(self, scene_data: SceneData, seed: int = 0) -> MDPState:
        """Create the runtime state, bound to the views of a scene with the same schema."""
        if scene_data.schema != self.schema:
            raise ValueError("The scene schema does not match the schema this MDP was built against.")
        fields = {
            selector: scene_data.data(selector.entity, selector.field)
            for selector in self.selectors
            if selector.entity != "commands"
        }
        for op in self.command_ops:
            fields[op.target] = ProxyArray(wp.zeros((self.num_worlds, *op.shape), dtype=wp.float32, device=self.device))
        streams = wp.zeros((len(self.streams), self.num_worlds), dtype=wp.int32, device=self.device)
        wp.to_torch(streams).copy_(rng.streams(seed, self.streams, self.num_worlds, self.device))
        return MDPState(
            episode_length=ProxyArray(wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)),
            actions=ProxyArray(wp.zeros((self.num_worlds, self.action_dim), dtype=wp.float32, device=self.device)),
            fields=fields,
            term_state={
                op.name: {
                    name: ProxyArray(wp.full(spec.shape, spec.initial, dtype=wp.float32, device=self.device))
                    for name, spec in op.state.items()
                }
                for op in self.term_ops
            },
            rng=ProxyArray(streams),
            timers={
                op.name: ProxyArray(wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device))
                for op in self.term_ops
                if op.timer is not None
            },
            outputs=self._term_outputs() if self.backend is Backend.WARP else None,
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

    def _term_outputs(self) -> TermOutputs:
        truncation = [op.truncation for op in self.termination_ops]
        reward_scales = [op.weight * self.step_dt for op in self.reward_ops]
        return TermOutputs(
            terminations=wp.zeros((len(truncation), self.num_worlds), dtype=wp.bool, device=self.device),
            is_truncation=wp.array(truncation, dtype=wp.bool, shape=len(truncation), device=self.device),
            rewards=wp.zeros((len(reward_scales), self.num_worlds), dtype=wp.float32, device=self.device),
            reward_scales=wp.array(reward_scales, dtype=wp.float32, shape=len(reward_scales), device=self.device),
            observations={
                group: wp.zeros((self.num_worlds, dim), dtype=wp.float32, device=self.device)
                for group, dim in self.observation_dims.items()
            },
            commands={
                op.name: wp.zeros((self.num_worlds, math.prod(op.shape)), dtype=wp.float32, device=self.device)
                for op in self.command_ops
            },
            due={
                op.name: wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
                for op in self.term_ops
                if op.timer is not None
            },
        )
