# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The compiled MDP and the runtime data it reads and writes."""

from __future__ import annotations

import enum
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from isaaclab.utils.warp import ProxyArray

from . import rng
from .config import FieldSelector, Phase, SubsetSelector, episode_length, last_action, term_state, terminated
from .plan import ActionOp, ExecutionPlan, Launches, Segment, TermOp, segments
from .scene import SceneData, SceneSchema


class Backend(enum.Enum):
    """Backend that runs the term implementations."""

    TORCH = "torch"
    WARP = "warp"


@dataclass
class MDPState:
    """Data of an MDP that changes as it runs, such as episode lengths, term state, timers, and random streams.

    Every array is in :attr:`arrays`, under the key that the backends use for it:

    - a field or subset selector: the selected fields of the scene, the fields of the MDP (``mdp2.last_action``,
      ``mdp2.commands.<name>``, ...), and the state of every term use, ``mdp2.term_state(use, name)``;
    - ``"rng"``: the random streams, ``(num_streams, num_worlds)``, see :mod:`.rng`;
    - ``"reward_weights"``: ``(num_worlds, num_rewards)``, which may change while the MDP runs, also in a CUDA graph;
    - ``("timer", op)``: seconds left per world, for a timed op;
    - ``("history", op)``: ``(num_worlds, history_length, size)``, for an observation with a history;
    - ``("scratch", op)``: the output of an eager term or of a Warp hook that launches its own kernels;
    - ``("selected", op)``: the worlds that the hook of a timed or masked event acts on;
    - ``("active", op)``: the worlds where an op is active, for an op that is active in only some.
    """

    arrays: dict[Hashable, ProxyArray]
    arguments: dict[str, dict[str, Any]]  # By op name: the hook arguments, as arrays of the backend of its hooks
    scene_data: SceneData  # The scene the state is bound to, for eager terms
    caches: dict[str, dict] = field(default_factory=dict)  # Host objects that eager terms keep, by op name

    @property
    def episode_length(self) -> ProxyArray:
        """Control steps since the last reset, ``(num_worlds,)``."""
        return self.arrays[episode_length]

    @property
    def actions(self) -> ProxyArray:
        """The last actions, ``(num_worlds, action_dim)``."""
        return self.arrays[last_action]

    @property
    def reward_weights(self) -> ProxyArray:
        return self.arrays["reward_weights"]

    @property
    def rng(self) -> ProxyArray:
        return self.arrays["rng"]


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

    @property
    def reset_writes(self) -> frozenset[FieldSelector]:
        """Scene fields that the reset events write, which the scene commits after a reset."""
        return frozenset(selector for op in self.ops[Phase.RESET_EVENTS] for selector in op.writes if selector.scene)

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
                    if op.observation is not None:
                        columns = op.observation.columns
                        details += [f"{op.observation.group}[{columns.start}:{columns.stop}]"]
                    if phase is Phase.REWARDS:
                        same = len(set(op.weights)) == 1
                        details += [f"weight {op.weights[0]}" if same else f"weights {op.weights}"]
                    details += ["time out"] if op.time_out else []
                    details += [f"every {op.timer[0]}-{op.timer[1]} s"] if op.timer is not None else []
                    details += [f"state {', '.join(op.state)}"] if op.state else []
                    details += [f"stream {op.stream}"] if op.stream is not None else []
                    details += ["torch hooks"] if op.torch_hooks else []
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

        # Fields the MDP owns, the state of every term, then the scene fields that actions and terms select.
        arrays: dict[Hashable, ProxyArray] = {
            last_action: self._zeros(self.num_worlds, self.action_dim),
            episode_length: self._zeros(self.num_worlds, dtype=wp.int32),
            terminated: self._zeros(self.num_worlds, dtype=wp.bool),
        }
        for op in self.ops[Phase.COMMANDS] + self.ops[Phase.QUANTITIES]:
            arrays[op.writes[0]] = self._zeros(self.num_worlds, *op.shape)
        for subset, indices in self.subsets.items():
            arrays[subset] = ProxyArray(wp.array(indices, dtype=wp.int32, device=self.device))
        for op in self.term_ops:
            for name, spec in op.state.items():
                values = wp.full(spec.shape, spec.initial, dtype=wp.float32, device=self.device)
                arrays[term_state(op.name, name)] = ProxyArray(values)

        selectors = [op.target for op in self.ops[Phase.ACTIONS]]
        selectors += [op.offset for op in self.ops[Phase.ACTIONS] if isinstance(op.offset, FieldSelector)]
        selectors += [selector for op in self.term_ops for selector in op.inputs.values()]
        noises = [op.observation.noise for op in self.ops[Phase.OBSERVATIONS]]
        selectors += [noise.scale for noise in noises if noise is not None and noise.scale is not None]
        for selector in selectors:
            if selector not in arrays:
                arrays[selector] = scene_data.data(selector.entity, selector.field)

        # Random streams and reward weights.
        arrays["rng"] = ProxyArray(wp.from_torch(rng.streams(seed, self.streams, self.num_worlds, self.device)))
        weights = np.zeros((self.num_worlds, len(self.ops[Phase.REWARDS])), dtype=np.float32)
        for index, config in enumerate(self.world_configs):
            weights[config.worlds.start : config.worlds.stop] = [op.weights[index] for op in self.ops[Phase.REWARDS]]
        arrays["reward_weights"] = ProxyArray(wp.array(weights, dtype=wp.float32, device=self.device))

        # Buffers of single ops.
        events = (Phase.STARTUP_EVENTS, Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS)
        for op in self.term_ops:
            if op.timer is not None:
                arrays["timer", op.name] = self._zeros(self.num_worlds)
            if op.observation is not None and op.observation.history_length:
                history_shape = (op.observation.history_length, math.prod(op.shape))
                arrays["history", op.name] = self._zeros(self.num_worlds, *history_shape)
            launches = op.eager or self.backend is Backend.WARP and not isinstance(op.compute, wp.Function)
            if launches and op.phase not in events:
                arrays["scratch", op.name] = self._zeros(self.num_worlds, *op.shape, dtype=op.dtype)
            masked = op.worlds is not None and op.phase in (Phase.STARTUP_EVENTS, Phase.RESET_EVENTS)
            if op.timer is not None or masked:
                arrays["selected", op.name] = self._zeros(self.num_worlds, dtype=wp.bool)
        for ops in self.ops.values():
            for op in ops:
                if op.worlds is not None:
                    active = np.isin(np.arange(self.num_worlds), op.worlds)
                    arrays["active", op.name] = ProxyArray(wp.array(active, dtype=wp.bool, device=self.device))

        # The hook arguments view the fields and the state; Torch hooks get Torch views, also in a Warp MDP.
        torch_views = {key: array.torch for key, array in arrays.items()}
        warp_views = {key: array.warp for key, array in arrays.items()}
        arguments = {}
        for op in self.term_ops:
            views = torch_views if self.backend is Backend.TORCH or op.torch_hooks else warp_views
            arguments[op.name] = {
                **{arg: views[selector] for arg, selector in op.inputs.items()},
                **{name: views[term_state(op.name, name)] for name in op.state},
                **({"rng": views["rng"][op.stream]} if op.random else {}),
                **op.params,
            }

        return MDPState(arrays=arrays, arguments=arguments, scene_data=scene_data)

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
