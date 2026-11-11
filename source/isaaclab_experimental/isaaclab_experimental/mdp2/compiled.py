# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The compiled MDP and the runtime data it reads and writes."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from isaaclab.utils.warp import ProxyArray

from . import rng
from .config import FieldSelector, Phase, SubsetSelector, episode_length, last_action, terminated
from .plan import ActionOp, ExecutionPlan, Launches, Segment, TermOp, segments
from .scene import SceneData, SceneSchema


class Backend(enum.Enum):
    """Backend that runs the term implementations."""

    TORCH = "torch"
    WARP = "warp"


@dataclass
class MDPState:
    """Data of an MDP that changes as it runs, such as episode lengths, term state, timers, and random streams."""

    episode_length: ProxyArray  # (num_worlds,), control steps since the last reset
    actions: ProxyArray  # (num_worlds, action_dim), last actions
    fields: dict[FieldSelector | SubsetSelector, ProxyArray]  # Every selected field of the scene and of the MDP
    term_state: dict[str, dict[str, ProxyArray]]  # By op name, then state name
    reward_weights: ProxyArray  # (num_worlds, num_rewards), can be changed at runtime, also under graph capture
    rng: ProxyArray  # (num_streams, num_worlds), see :mod:`.rng`
    timers: dict[str, ProxyArray]  # (num_worlds,) seconds left per timed op
    arguments: dict[str, dict[str, Any]]  # By op name: the hook arguments, as arrays of the MDP's backend
    scratch: dict[str, ProxyArray]  # (num_worlds, *shape) outputs of eager terms and of Warp kernel launchers
    histories: dict[str, ProxyArray]  # (num_worlds, history_length, size) per observation with a history
    selected: dict[str, wp.array]  # (num_worlds,) per timed or masked event op: the worlds its hook acts on
    active: dict[str, ProxyArray]  # (num_worlds,) per op active in only some worlds: where it is active
    scene_data: SceneData  # The scene the state is bound to, for eager terms


@dataclass
class Transition:
    """Results of one step, written into buffers that the caller owns."""

    rewards: ProxyArray  # (num_worlds,)
    terminated: ProxyArray  # (num_worlds,)
    truncated: ProxyArray  # (num_worlds,)
    done: ProxyArray  # (num_worlds,), terminated or truncated: the worlds to reset
    observations: dict[str, ProxyArray]  # Successor observations before resets, (num_worlds, group_dim) per group
    reward_terms: ProxyArray | None = None  # (num_worlds, num_rewards) weighted rewards, only with diagnostics
    termination_terms: ProxyArray | None = None  # (num_worlds, num_terminations) flags, only with diagnostics


@dataclass(frozen=True)
class WorldConfig:
    """Worlds that run one task config, and the columns of the padded buffers that its terms use."""

    worlds: range
    action_dim: int  # Columns of the actions it uses, from the first; the others are padding
    observation_dims: Mapping[str, int]  # Columns of each observation group it uses, from the first


@dataclass(frozen=True)
class MDP:
    """An immutable, compiled MDP, created with :meth:`MDPBuilder.finalize`."""

    schema: SceneSchema
    device: str
    backend: Backend
    ops: Mapping[Phase, tuple[ActionOp | TermOp, ...]]  # In schedule order
    plan: ExecutionPlan
    action_dim: int
    observation_dims: Mapping[str, int]
    max_episode_length: int | None  # Control steps until truncation, None without a time limit
    streams: tuple[str, ...]  # Op names of the rows of MDPState.rng
    subsets: Mapping[SubsetSelector, tuple[int, ...]]  # Indices of every selected subset
    diagnostics: bool  # Whether evaluate also writes the reward and termination of every term
    world_configs: tuple[WorldConfig, ...]  # One per replicated builder, or one for every world
    launches: Launches | None = None  # Set by finalize, None for an MDP that was only validated

    @property
    def num_worlds(self) -> int:
        return self.schema.num_worlds

    @property
    def step_dt(self) -> float:
        return self.schema.step_dt

    @property
    def term_ops(self) -> tuple[TermOp, ...]:
        """Ops of every phase except actions, in schedule order."""
        return tuple(op for phase, ops in self.ops.items() if phase is not Phase.ACTIONS for op in ops)

    @property
    def schedule(self) -> tuple[str, ...]:
        """Names of all ops in schedule order."""
        return tuple(op.name for ops in self.ops.values() for op in ops)

    @property
    def segments(self) -> dict[str, tuple[Segment, ...]]:
        """Segments of each runtime function: runs of steps that are captured as one CUDA graph, or run eagerly.

        The Warp backend may need fewer graphs: it runs a hook early when nothing between depends on it.
        """
        return {function: segments(steps) for function, steps in vars(self.plan).items()}

    @property
    def is_fully_capturable(self) -> bool:
        """Whether every runtime function can be captured as a single CUDA graph. Startup runs once, uncaptured."""
        functions = [segments for function, segments in self.segments.items() if function != "startup"]
        return all(segment.capturable for segments in functions for segment in segments)

    def describe(self) -> str:
        """Return the compiled MDP as readable text: ops by phase with their layout, then each function's steps."""
        lines = [
            f"{self.backend.value} MDP on {self.device}: {self.num_worlds} worlds, step_dt {self.step_dt} s,"
            f" action_dim {self.action_dim}, observation_dims {dict(self.observation_dims)},"
            f" max_episode_length {self.max_episode_length}"
        ]
        width = max((len(name) for name in self.schedule), default=0)

        for phase, ops in self.ops.items():
            if ops:
                lines.append(phase.name.lower().replace("_", " "))
            for op in ops:
                if phase is Phase.ACTIONS:
                    details = [f"columns {op.columns.start}:{op.columns.stop} -> {op.target}"]
                    details += [f"scale {op.scale}", f"offset {op.offset}"]
                else:
                    details = [f"{arg}={selector}" for arg, selector in op.inputs.items()]
                    details += [f"{name}={value!r}" for name, value in op.params.items()]
                    details += [f"shape {op.shape}"] if op.shape else []
                    details += [f"{op.group}[{op.columns.start}:{op.columns.stop}]"] if op.group else []
                    if phase is Phase.REWARDS:
                        same = len(set(op.weights)) == 1
                        details += [f"weight {op.weights[0]}" if same else f"weights {op.weights}"]
                    details += ["time out"] if op.time_out else []
                    details += [f"every {op.timer[0]}-{op.timer[1]} s"] if op.timer is not None else []
                    details += [f"state {', '.join(op.state)}"] if op.state else []
                    details += [f"stream {op.stream}"] if op.stream is not None else []
                if op.worlds is not None:
                    first, last = op.worlds[0], op.worlds[-1]
                    contiguous = op.worlds == tuple(range(first, last + 1))
                    details.append(f"worlds {first}-{last}" if contiguous else f"worlds {list(op.worlds)}")
                lines.append(f"  {op.name:{width}}  {', '.join(details)}")

        for function, function_segments in self.segments.items():
            if function_segments:
                lines.append(f"{function} steps")
            for segment in function_segments:
                lines.append(f"  {'graph' if segment.capturable else 'eager'}")
                for step in segment.steps:
                    lines.append(f"    {step.kind.name.lower().replace('_', ' '):14}  {step.op.name}")

        return "\n".join(lines)

    def state(self, scene_data: SceneData, seed: int = 0) -> MDPState:
        """Create the runtime state, bound to the data views of a scene with the same schema."""
        if scene_data.schema != self.schema:
            raise ValueError("The scene schema does not match the schema this MDP was built against.")

        # Fields the MDP owns, then the scene fields that actions and terms select.
        fields = {
            last_action: self._zeros(self.num_worlds, self.action_dim),
            episode_length: self._zeros(self.num_worlds, dtype=wp.int32),
            terminated: self._zeros(self.num_worlds, dtype=wp.bool),
        }
        for op in self.ops[Phase.COMMANDS]:
            fields[op.writes[0]] = self._zeros(self.num_worlds, *op.shape)
        for subset, indices in self.subsets.items():
            fields[subset] = ProxyArray(wp.array(indices, dtype=wp.int32, device=self.device))

        selectors = [op.target for op in self.ops[Phase.ACTIONS]]
        selectors += [op.offset for op in self.ops[Phase.ACTIONS] if isinstance(op.offset, FieldSelector)]
        selectors += [selector for op in self.term_ops for selector in op.inputs.values()]
        for selector in selectors:
            if selector not in fields:
                fields[selector] = scene_data.data(selector.entity, selector.field)

        # Term state, random streams, and the hook arguments that view them.
        term_state = {
            op.name: {
                name: ProxyArray(wp.full(spec.shape, spec.initial, dtype=wp.float32, device=self.device))
                for name, spec in op.state.items()
            }
            for op in self.term_ops
        }
        streams = ProxyArray(wp.from_torch(rng.streams(seed, self.streams, self.num_worlds, self.device)))
        weights = np.zeros((self.num_worlds, len(self.ops[Phase.REWARDS])), dtype=np.float32)
        for index, config in enumerate(self.world_configs):
            weights[config.worlds.start : config.worlds.stop] = [op.weights[index] for op in self.ops[Phase.REWARDS]]

        arguments = {
            op.name: {
                **{arg: self._view(fields[selector]) for arg, selector in op.inputs.items()},
                **{name: self._view(value) for name, value in term_state[op.name].items()},
                **({"rng": self._view(streams)[op.stream]} if op.random else {}),
                **op.params,
            }
            for op in self.term_ops
        }

        return MDPState(
            episode_length=fields[episode_length],
            actions=fields[last_action],
            fields=fields,
            term_state=term_state,
            reward_weights=ProxyArray(wp.array(weights, dtype=wp.float32, device=self.device)),
            rng=streams,
            timers={op.name: self._zeros(self.num_worlds) for op in self.term_ops if op.timer is not None},
            arguments=arguments,
            histories={
                op.name: self._zeros(self.num_worlds, op.history_length, math.prod(op.shape))
                for op in self.ops[Phase.OBSERVATIONS]
                if op.history_length
            },
            scratch={
                op.name: self._zeros(self.num_worlds, *op.shape, dtype=op.dtype)
                for op in self.term_ops
                if op.phase not in (Phase.STARTUP_EVENTS, Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS)
                and (op.eager or self.backend is Backend.WARP and not isinstance(op.compute, wp.Function))
            },
            selected={
                op.name: wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
                for op in self.term_ops
                if op.timer is not None
                or (op.worlds is not None and op.phase in (Phase.STARTUP_EVENTS, Phase.RESET_EVENTS))
            },
            active={
                op.name: ProxyArray(
                    wp.array(np.isin(np.arange(self.num_worlds), op.worlds), dtype=wp.bool, device=self.device)
                )
                for ops in self.ops.values()
                for op in ops
                if op.worlds is not None
            },
            scene_data=scene_data,
        )

    def observations(self) -> dict[str, ProxyArray]:
        """Allocate one zeroed observation buffer per group, of shape ``(num_worlds, group_dim)``."""
        return {group: self._zeros(self.num_worlds, dim) for group, dim in self.observation_dims.items()}

    def transition(self) -> Transition:
        """Allocate the zeroed buffers of a transition, with per-term buffers only with diagnostics."""
        reward_terms = termination_terms = None
        if self.diagnostics:
            reward_terms = self._zeros(self.num_worlds, len(self.ops[Phase.REWARDS]))
            termination_terms = self._zeros(self.num_worlds, len(self.ops[Phase.TERMINATIONS]), dtype=wp.bool)

        return Transition(
            rewards=self._zeros(self.num_worlds),
            terminated=self._zeros(self.num_worlds, dtype=wp.bool),
            truncated=self._zeros(self.num_worlds, dtype=wp.bool),
            done=self._zeros(self.num_worlds, dtype=wp.bool),
            observations=self.observations(),
            reward_terms=reward_terms,
            termination_terms=termination_terms,
        )

    def _zeros(self, *shape: int, dtype: type = wp.float32) -> ProxyArray:
        return ProxyArray(wp.zeros(shape, dtype=dtype, device=self.device))

    def _view(self, array: ProxyArray) -> Any:
        return array.torch if self.backend is Backend.TORCH else array.warp
