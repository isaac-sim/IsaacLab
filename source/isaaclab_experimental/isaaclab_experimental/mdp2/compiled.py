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
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from . import rng
from .config import (
    Entity,
    FieldSelector,
    Phase,
    SubsetSelector,
    TaskMode,
    episode_length,
    last_action,
    progress_field,
    term_state,
    terminated,
    time_out,
)
from .plan import (
    ActionOp,
    ActionSlot,
    ExecutionPlan,
    Launches,
    ObservationSlot,
    Segment,
    Step,
    TaskTree,
    TermOp,
    TunedValue,
    segments,
)
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
    - ``("scratch", op)``: the output of a Python hook in a Warp MDP;
    - ``("selected", op)``: the worlds that the hook of a timed op, or of an op active in some worlds, acts on;
    - ``("active", op)``: the worlds where an op is active, for an op that is active in only some;
    - ``("on", op)`` and ``("entered", op)``: for an op with ``when``, or of several tasks of a sequence or a choice,
      the worlds where it runs at this step, and those where it turned on at this step;
    - ``"event_worlds"``: the worlds where an interval or start event that writes scene fields ran in the last
      ``evaluate``, for an MDP with such events;
    - ``mdp2.progress_field(task, name)``: the progress of every task, with the internal ``streak`` (steps that a
      task's ``done`` held), ``entered`` (whether a task of a sequence started at this step), and, for a task that
      repeats, ``spent`` (done, and its completion still holds) and ``restarted`` (it restarted at this step);
    - ``mdp2.tunable(use, name)``: the value of a tunable param per world, and, for an MDP with tunable values,
      ``("pending", key)``: the values tuned for the next reset of each array, ``"pending"``: which values wait in
      each world, a bit per value of :attr:`MDP.tuned` in ``int32`` words, ``(num_worlds, words)``, and
      ``"revision"``: the number of times tunings took effect in each world, see :func:`~.tune`.
    """

    arrays: dict[Hashable, ProxyArray]
    arguments: dict[str, dict[str, Any]]  # By op name: the hook arguments, as arrays of the backend of its hooks
    scene_data: SceneData  # The scene the state is bound to, whose entities Python hooks may take
    caches: dict[str, dict] = field(default_factory=dict)  # Host objects that Python hooks keep, by op name

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

    @property
    def revision(self) -> ProxyArray | None:
        """Tunings applied in each world, ``int32`` of shape ``(num_worlds,)``; None without tunable values."""
        return self.arrays.get("revision")


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
    # Observations of the worlds that ended, before their reset, per group; only with final observations. The rows of
    # the other worlds keep older values.
    final_observations: dict[str, ProxyArray] | None = None


@dataclass(frozen=True)
class WorldConfig:
    """Worlds that run one task config, and the columns of the padded buffers that its terms use."""

    worlds: range
    action_dim: int  # Columns of the action up to the last that it uses; see MDP.action_mask for those it uses
    observation_dims: Mapping[str, int]  # Columns of each observation group it uses, from the first


@dataclass(frozen=True)
class Columns:
    """Columns of the action, or of an observation buffer, that the ops of one name take or write, see
    :meth:`MDP.spec`."""

    name: str
    columns: slice  # For every step of the history
    size: int  # Columns of one step of history
    history_length: int
    widths: Mapping[str, int]  # Columns of one step that each op takes or writes, by op name; the others are zero
    # Names of the elements of each op, by op name, such as joint names, for the ops whose inputs name them
    labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Spec:
    """The interface of an MDP by name: the columns of its action and of its observation buffers."""

    actions: tuple[Columns, ...]
    observations: Mapping[str, tuple[Columns, ...]]  # By buffer


@dataclass(frozen=True)
class MDP:
    """An immutable, compiled MDP, created with :meth:`MDPBuilder.finalize`."""

    schema: SceneSchema
    device: str
    backend: Backend
    ops: Mapping[Phase, tuple[ActionOp | TermOp, ...]]  # In schedule order
    plan: ExecutionPlan
    action_dim: int
    observation_dims: Mapping[str, int]  # Elements per world of each observation buffer
    # Per buffer: the shape after the world dimension and the element type. A concatenated group is (dim,) float32;
    # a term of a group that is not concatenated is the buffer "group/term", with the term's own shape and type.
    observation_layouts: Mapping[str, tuple[tuple[int, ...], type]]
    max_episode_length: int | None  # Control steps until truncation, None without a time limit
    streams: tuple[str, ...]  # Use names of the rows of MDPState.rng, with "/noise" for observation noise
    subsets: Mapping[SubsetSelector, tuple[int, ...]]  # Indices of every selected subset
    diagnostics: bool  # Whether evaluate also writes the reward and termination of every term
    final_observations: bool  # Whether evaluate also writes the observations of the worlds that ended
    world_configs: tuple[WorldConfig, ...]  # One per replicated builder, or one for every world
    # The task tree of each world config, if the MDP tracks progress: some task has a completion or a sequence
    tasks: tuple[TaskTree, ...] = ()
    slots: tuple[ObservationSlot, ...] = ()  # Observation slots that some op writes only in some steps
    action_slots: tuple[ActionSlot, ...] = ()  # The columns of the action of each name, in order
    tuned: tuple[TunedValue, ...] = ()  # Values declared Tunable, which tune changes
    launches: Launches | None = None  # Set by finalize, None for an MDP that was only validated

    @property
    def num_worlds(self) -> int:
        return self.schema.num_worlds

    @property
    def step_dt(self) -> float:
        return self.schema.step_dt

    @property
    def term_ops(self) -> tuple[TermOp, ...]:
        """Ops of term uses, of every phase, in schedule order: all but the direct actions."""
        return tuple(op for ops in self.ops.values() for op in ops if isinstance(op, TermOp))

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
    def startup_writes(self) -> frozenset[FieldSelector]:
        """Scene fields that the startup events write, which the scene commits after startup."""
        return self._scene_writes(Phase.STARTUP_EVENTS)

    @property
    def reset_writes(self) -> frozenset[FieldSelector]:
        """Scene fields that the reset and start events write, which the scene commits after a reset."""
        return self._scene_writes(Phase.RESET_EVENTS)

    @property
    def step_writes(self) -> frozenset[FieldSelector]:
        """Scene fields that events write during a step, the interval events and the start events of tasks that start
        during an episode, which the scene commits in the worlds where one ran."""
        starts = [op for op in self.ops[Phase.RESET_EVENTS] if op.entry_key is not None]
        return self._scene_writes(Phase.INTERVAL_EVENTS) | frozenset(
            selector for op in starts for selector in op.writes if selector.scene
        )

    def _scene_writes(self, phase: Phase) -> frozenset[FieldSelector]:
        return frozenset(selector for op in self.ops[phase] for selector in op.writes if selector.scene)

    def spec(self) -> Spec:
        """Return the columns of the action and of every observation buffer by name, with the names of their
        elements where known, so that learners, policy servers, and exporters need not know their offsets.

        Actions, and observations, of the same name share their columns, also across world configs and tasks of a
        sequence; see :meth:`action_mask` and :meth:`observation_mask` for the columns that the worlds take and write
        at a step.
        """

        def labels(ops: list) -> Mapping[str, tuple[str, ...]]:
            return MappingProxyType({op.name: op.labels for op in ops if op.labels is not None})

        named: dict[str, dict[str, list[TermOp]]] = {}
        for op in self.ops[Phase.OBSERVATIONS]:
            named.setdefault(op.observation.group, {}).setdefault(op.observation.slot, []).append(op)
        observations = {
            buffer: tuple(
                Columns(
                    name=name,
                    columns=ops[0].observation.columns,
                    size=ops[0].observation.size,
                    history_length=ops[0].observation.history_length,
                    widths=MappingProxyType({op.name: math.prod(op.shape) for op in ops}),
                    labels=labels(ops),
                )
                for name, ops in slots.items()
            )
            for buffer, slots in named.items()
        }
        actions = tuple(
            Columns(
                name=slot.name,
                columns=slot.columns,
                size=slot.columns.stop - slot.columns.start,
                history_length=0,
                widths=MappingProxyType({op.name: op.columns.stop - op.columns.start for op in slot.ops}),
                labels=labels(list(slot.ops)),
            )
            for slot in self.action_slots
        )
        return Spec(actions=actions, observations=MappingProxyType(observations))

    def action_mask(self, state: MDPState) -> torch.Tensor:
        """Return which columns of the action each world takes at this step, ``(num_worlds, action_dim)`` bool.

        The others pad an action to the widest of its name, or belong to actions of other world configs or of tasks of
        a sequence that do not run: the MDP ignores them, and stores them as zero in the last actions. Torch operations
        only, so it can run in a graph.
        """
        mask = torch.zeros(self.num_worlds, self.action_dim, dtype=torch.bool, device=self.device)
        for op in self.ops[Phase.ACTIONS]:
            taken = True if op.active_key is None else state.arrays[op.active_key].torch[:, None]
            mask[:, op.columns] |= taken
        return mask

    def observation_mask(self, state: MDPState, buffer: str) -> torch.Tensor:
        """Return which columns of an observation buffer each world writes at this step, ``(num_worlds, dim)`` bool.

        The others are zero: they pad an observation to the widest of its name, or belong to observations of other
        world configs or of tasks of a sequence that do not run. Torch operations only, so it can run in a graph.
        """
        mask = torch.zeros(self.num_worlds, self.observation_dims[buffer], dtype=torch.bool, device=self.device)
        for op in self.ops[Phase.OBSERVATIONS]:
            if op.observation.group != buffer:
                continue
            columns = op.observation.columns
            width = torch.arange(columns.stop - columns.start, device=self.device) % op.observation.size
            written = (width < math.prod(op.shape))[None, :]
            if op.active_key is not None:
                written = written & state.arrays[op.active_key].torch[:, None]
            mask[:, columns] |= written
        return mask

    def describe(self) -> str:
        """Return the compiled MDP as readable text: ops by phase with their layout, then each function's steps."""
        lines = [
            f"{self.backend.value} MDP on {self.device}: {self.num_worlds} worlds, step_dt {self.step_dt} s,"
            f" action_dim {self.action_dim}, observation_dims {dict(self.observation_dims)},"
            f" max_episode_length {self.max_episode_length}"
        ]
        width = max((len(name) for name in self.schedule), default=0)

        if self.tuned:
            lines.append(f"tunable {', '.join(value.path for value in self.tuned)}")
        for tree in self.tasks:
            worlds = f" (worlds {tree.worlds.start}-{tree.worlds.stop - 1})" if len(self.tasks) > 1 else ""
            lines.append(f"tasks{worlds}")
            for node in tree.nodes:
                details = [node.mode.name.lower()] if node.children else []
                details += [f"done {node.done}"] if node.done is not None else []
                details += [f"hold {node.hold}"] if node.hold > 1 else []
                details += [f"require {node.require}"] if node.require is not None else []
                details += [f"final {node.final}"] if node.final is not None else []
                details += ["staged"] if node.staged else []
                details += ["repeat"] if node.repeat else []
                lines.append(f"  {node.path or '(root)'}  {', '.join(details)}".rstrip())

        for phase, ops in self.ops.items():
            if ops:
                lines.append(phase.name.lower().replace("_", " "))
            for op in ops:
                if isinstance(op, ActionOp):
                    details = [f"columns {op.columns.start}:{op.columns.stop} -> {op.target}"]
                    details += [f"scale {op.scale}", f"offset {op.offset}"]
                    if op.clip is not None:
                        details.append(f"clip {op.clip}")
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
                    details += [f"during {op.stage}"] if op.stage else []
                    details += [f"during {' or '.join(op.stages)}"] if op.stages else []
                    details += [f"when {op.when}"] if op.when is not None else []
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
                    lines.append(f"    {step.kind.name.lower().replace('_', ' '):14}  {_target(step)}")

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
            time_out: self._zeros(self.num_worlds, dtype=wp.bool),
        }
        arrays |= self._progress_arrays()
        for op in self.ops[Phase.COMMANDS] + self.ops[Phase.QUANTITIES]:
            arrays[op.writes[0]] = self._zeros(self.num_worlds, *op.shape)
        dtypes = {value.key: value.dtype for value in self.tuned}
        for op in self.term_ops:
            # Each world config starts the tunable values of its worlds at its own default.
            defaults = {op.inputs[name]: default for name, default in op.tunables.items()}
            defaults |= {op.when: op.when_default} if op.when_default is not None else {}
            for key, default in defaults.items():
                if key not in arrays:
                    arrays[key] = self._zeros(self.num_worlds, dtype=dtypes[key])
                worlds = slice(None) if op.worlds is None else list(op.worlds)
                arrays[key].torch[worlds] = torch.tensor(default, dtype=arrays[key].torch.dtype, device=self.device)
            if op.when is not None or op.stages:
                arrays["on", op.name] = self._zeros(self.num_worlds, dtype=wp.bool)
                arrays["entered", op.name] = self._zeros(self.num_worlds, dtype=wp.bool)
        for subset, indices in self.subsets.items():
            arrays[subset] = ProxyArray(wp.array(indices, dtype=wp.int32, device=self.device))
        arrays |= self._direct_action_arrays()
        # State belongs to the use: the ops that other world configs define differently share it.
        for op in self.term_ops:
            for name, spec in op.state.items():
                key = term_state(op.use, name)
                if key not in arrays:
                    values = wp.full(spec.shape, spec.initial, dtype=wp.float32, device=self.device)
                    arrays[key] = ProxyArray(values)
                elif op.worlds is not None and spec.shape[:1] == (self.num_worlds,):
                    # Each world config starts the shared state of its worlds at its own initial value.
                    arrays[key].torch[list(op.worlds)] = spec.initial

        direct = [op for op in self.ops[Phase.ACTIONS] if isinstance(op, ActionOp)]
        selectors = [op.target for op in direct]
        selectors += [op.offset for op in direct if isinstance(op.offset, FieldSelector)]
        selectors += [
            selector for op in self.term_ops for selector in op.inputs.values() if not isinstance(selector, Entity)
        ]
        noises = [op.observation.noise for op in self.ops[Phase.OBSERVATIONS]]
        selectors += [noise.scale for noise in noises if noise is not None and noise.scale is not None]
        selectors += [op.when for op in self.term_ops if op.when is not None]
        for selector in selectors:
            if selector not in arrays:
                arrays[selector] = scene_data.data(selector.entity, selector.field)

        # Random streams and reward weights.
        arrays["rng"] = ProxyArray(wp.from_torch(rng.streams(seed, self.streams, self.num_worlds, self.device)))
        weights = np.zeros((self.num_worlds, len(self.ops[Phase.REWARDS])), dtype=np.float32)
        for index, config in enumerate(self.world_configs):
            weights[config.worlds.start : config.worlds.stop] = [op.weights[index] for op in self.ops[Phase.REWARDS]]
        arrays["reward_weights"] = ProxyArray(wp.array(weights, dtype=wp.float32, device=self.device))
        if self.tuned:
            # Values tuned for the next reset wait in a copy of their array, in the worlds whose bit of it is set.
            arrays["pending"] = self._zeros(self.num_worlds, (len(self.tuned) + 31) // 32, dtype=wp.int32)
            arrays["revision"] = self._zeros(self.num_worlds, dtype=wp.int32)
            for key in dict.fromkeys(value.key for value in self.tuned):
                arrays["pending", key] = ProxyArray(wp.clone(arrays[key].warp))
        if self.step_writes:
            arrays["event_worlds"] = self._zeros(self.num_worlds, dtype=wp.bool)

        # Buffers of single ops.
        events = (Phase.STARTUP_EVENTS, Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS)
        for op in self.term_ops:
            if op.timer is not None:
                arrays["timer", op.name] = self._zeros(self.num_worlds)
            if op.observation is not None and op.observation.history_length:
                history_shape = (op.observation.history_length, math.prod(op.shape))
                arrays["history", op.name] = self._zeros(self.num_worlds, *history_shape)
            launches = self.backend is Backend.WARP and not isinstance(op.compute, wp.Function)
            if launches and op.phase not in (*events, Phase.ACTIONS):
                arrays["scratch", op.name] = self._zeros(self.num_worlds, *op.shape, dtype=op.dtype)
            # Hooks of ops active in only some worlds get the reset worlds among those.
            if op.timer is not None or op.active_key is not None:
                arrays["selected", op.name] = self._zeros(self.num_worlds, dtype=wp.bool)
        for ops in self.ops.values():
            for op in ops:
                if op.worlds is not None:
                    active = np.isin(np.arange(self.num_worlds), op.worlds)
                    arrays["active", op.name] = ProxyArray(wp.array(active, dtype=wp.bool, device=self.device))

        # The hook arguments view the fields and the state; Torch hooks get Torch views, also in a Warp MDP. Python
        # hooks also get the scene entities bound to them, and a cache for host objects.
        torch_views = {key: array.torch for key, array in arrays.items()}
        warp_views = {key: array.warp for key, array in arrays.items()}
        arguments, caches = {}, {}
        for op in self.term_ops:
            torch_hooks = self.backend is Backend.TORCH or op.torch_hooks
            views = torch_views if torch_hooks else warp_views
            python = not isinstance(op.compute, wp.Function)
            arguments[op.name] = {
                **{
                    arg: scene_data.entity(selector.name) if isinstance(selector, Entity) else views[op.array_key(arg)]
                    for arg, selector in op.inputs.items()
                },
                # Torch hooks get a tunable tuple as a tuple of per-world tensors.
                **{
                    name: tuple(torch_views[op.inputs[name]].unbind(1))
                    for name, default in op.tunables.items()
                    if torch_hooks and isinstance(default, tuple)
                },
                **({"cache": caches.setdefault(op.name, {})} if python else {}),
                **{name: views[term_state(op.use, name)] for name in op.state},
                **({"rng": views["rng"][op.stream]} if op.random else {}),
                **op.params,
            }

        return MDPState(arrays=arrays, arguments=arguments, scene_data=scene_data, caches=caches)

    def _direct_action_arrays(self) -> dict[Hashable, ProxyArray]:
        """Allocate the per-element scales and clip ranges of the direct actions that have them."""
        arrays = {}
        for op in self.ops[Phase.ACTIONS]:
            if not isinstance(op, ActionOp):
                continue  # Action terms write their outputs themselves
            if isinstance(op.scale, tuple):
                arrays["action_scale", op.name] = ProxyArray(wp.array(op.scale, dtype=wp.float32, device=self.device))
            if op.clip is not None:
                arrays["action_clip", op.name] = ProxyArray(wp.array(op.clip, dtype=wp.vec2f, device=self.device))
        return arrays

    def _progress_arrays(self) -> dict[Hashable, ProxyArray]:
        """Allocate the progress of every task, as at the start of an episode: only the tasks that run first are
        active, in the worlds of their world config."""
        values: dict[FieldSelector, np.ndarray] = {}

        def array(path: str, name: str, dtype: type, initial: float = 0) -> np.ndarray:
            return values.setdefault(progress_field(path, name), np.full(self.num_worlds, initial, dtype=dtype))

        for tree in self.tasks:
            worlds = slice(tree.worlds.start, tree.worlds.stop)
            for node in tree.nodes:
                for name in ("done", "holding", "active"):
                    array(node.path, name, np.bool_)
                array(node.path, "done_at", np.int32, -1)
                array(node.path, "started_at", np.int32)
                array(node.path, "score", np.float32)
                if node.mode in (TaskMode.SEQUENCE, TaskMode.CHOICE):
                    array(node.path, "stage", np.int32)
                if node.done is not None:
                    array(node.path, "streak", np.int32)
                if node.staged:
                    array(node.path, "entered", np.bool_)
                if node.repeat:
                    array(node.path, "count", np.int32)
                    array(node.path, "spent", np.bool_)
                    array(node.path, "restarted", np.bool_)
                values[progress_field(node.path, "active")][worlds] = tree.runs_at_start(node)

        dtypes = {np.bool_: wp.bool, np.int32: wp.int32, np.float32: wp.float32}
        return {
            key: ProxyArray(wp.array(value, dtype=dtypes[value.dtype.type], device=self.device))
            for key, value in values.items()
        }

    def observations(self) -> dict[str, ProxyArray]:
        """Allocate one zeroed buffer per observation buffer, of shape ``(num_worlds, dim)`` and its element type.

        The buffer of a separate term is flat per world: view it with the shape in :attr:`observation_layouts`.
        """
        return {
            buffer: self._zeros(self.num_worlds, dim, dtype=self.observation_layouts[buffer][1])
            for buffer, dim in self.observation_dims.items()
        }

    def transition(self) -> Transition:
        """Allocate the zeroed buffers of a transition, with the optional buffers that the MDP writes."""
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
            final_observations=self.observations() if self.final_observations else None,
        )

    def _zeros(self, *shape: int, dtype: type = wp.float32) -> ProxyArray:
        return ProxyArray(wp.zeros(shape, dtype=dtype, device=self.device))


def _target(step: Step) -> str:
    """Return what a step works on, for :meth:`MDP.describe`."""
    if isinstance(step.op, ObservationSlot):
        return f"{step.op.buffer}[{step.op.columns.start}:{step.op.columns.stop}]"
    if isinstance(step.op, tuple):
        return "tasks"
    return f"{step.op.name}{' on start' if step.entry else ''}"
