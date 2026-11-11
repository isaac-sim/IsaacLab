# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builder that collects term uses into a schedule and compiles it into an :class:`MDP`, like Newton's ModelBuilder."""

from __future__ import annotations

import collections
import graphlib
import inspect
import itertools
import math
import re
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import warp as wp

from . import torch_backend, warp_backend
from .compiled import MDP, Backend, WorldConfig
from .config import (
    Command,
    DirectAction,
    Event,
    EventMode,
    FieldSelector,
    Observation,
    Phase,
    Quantity,
    Reward,
    SubsetSelector,
    TaskCfg,
    TermCfg,
    Termination,
    TimeLimit,
    commands,
    episode_length,
    last_action,
    quantities,
    step_dt,
    term_state,
    terminated,
)
from .plan import ActionOp, ObservationOutput, Segment, TermOp, build_plan
from .scene import SceneSchema
from .terms import Implementation, hook_arguments


@dataclass(frozen=True)
class ScheduleEntry:
    """Handle to one term use in the schedule, returned by the ``add_*`` methods."""

    path: tuple[str, ...]  # Location in the task, e.g. ("observations", "policy", "joint_pos")
    cfg: DirectAction | Command | Quantity | Observation | Reward | Termination | TimeLimit | Event

    def __str__(self) -> str:
        return ".".join(self.path)

    @property
    def phase(self) -> Phase:
        return self.cfg.phase

    @property
    def reads(self) -> tuple[FieldSelector, ...]:
        """Fields the use reads, which are its field inputs."""
        if not isinstance(self.cfg, TermCfg):
            return ()
        reads = [selector for selector in self.cfg.inputs.values() if isinstance(selector, FieldSelector)]
        if isinstance(self.cfg, Observation) and self.cfg.noise is not None and self.cfg.noise.scale is not None:
            reads.append(self.cfg.noise.scale)
        return tuple(reads)

    @property
    def writes(self) -> tuple[FieldSelector, ...]:
        """Fields the use writes: an action's target, the value of a command or quantity, or an event's outputs."""
        if isinstance(self.cfg, DirectAction):
            return (self.cfg.target,)
        if isinstance(self.cfg, Command):
            return (FieldSelector(commands.name, self.path[1], scene=False),)
        if isinstance(self.cfg, Quantity):
            return (FieldSelector(quantities.name, self.path[1], scene=False),)
        if isinstance(self.cfg, Event):
            return tuple(self.cfg.outputs.values())
        return ()

    @property
    def provides(self) -> tuple[FieldSelector, ...]:
        """Fields of the MDP that other uses can read once this one is compiled: its value and its state."""
        if not isinstance(self.cfg, TermCfg):
            return ()
        state = [term_state(str(self), name) for name in self.cfg.term.state]
        return (*(selector for selector in self.writes if not selector.scene), *state)


@dataclass(frozen=True)
class ArrayLayout:
    """Element type, number of dimensions, and dimension sizes of an array passed to a hook."""

    dtype: type
    ndim: int
    sizes: Mapping[str, int]  # Sizes of the named dimensions
    shape: tuple[int, ...] = ()  # Sizes of the dimensions after the world dimension, named or not


@dataclass
class CompileContext:
    """Ops, layouts, random streams, and errors collected while compiling the schedule."""

    backend: Backend
    errors: list[str]
    ops: dict[Phase, list] = field(default_factory=lambda: {phase: [] for phase in Phase})
    mdp_fields: dict[FieldSelector, ArrayLayout] = field(  # Fields owned by the MDP; uses add theirs as they compile
        default_factory=lambda: {
            last_action: ArrayLayout(wp.float32, 2, {"actions": 0}, (0,)),
            episode_length: ArrayLayout(wp.int32, 1, {}),
            terminated: ArrayLayout(wp.bool, 1, {}),
        }
    )
    subsets: dict[SubsetSelector, tuple[int, ...]] = field(default_factory=dict)
    observation_dims: dict[str, int] = field(default_factory=dict)
    action_dim: int = 0


class MDPBuilder:
    """Collects term uses, lets users edit their order, and compiles them with :meth:`finalize`.

    Uses run phase by phase. Within a phase they run in the order they were added, unless the order is edited with
    :meth:`move_before`, :meth:`move_after`, or :meth:`set_order`. Every edit is checked so that a use that reads a
    field still runs after the uses of its phase that write it. Action and observation columns follow the schedule
    order.
    """

    def __init__(self, schema: SceneSchema):
        self.schema = schema
        self._entries: dict[tuple[str, ...], ScheduleEntry] = {}
        self._world_configs: list[tuple[tuple[ScheduleEntry, ...], range]] = []  # Schedules of replicated builders

    def replicate(self, builder: MDPBuilder, num_worlds: int) -> None:
        """Run the task of another builder in the next ``num_worlds`` worlds.

        The builder's schedule is copied, so later edits to it do not apply. Once worlds are assigned, every world
        must be, and term uses are added to the replicated builders rather than to this one.
        """
        if builder.schema != self.schema:
            raise ValueError("A replicated builder must be built against the same scene schema.")

        start = self._world_configs[-1][1].stop if self._world_configs else 0
        self._world_configs.append((builder.schedule(), range(start, start + num_worlds)))

    def add_task(self, task: TaskCfg) -> None:
        """Add all term uses of a task in declaration order, skipping uses set to ``None``."""
        for name, action in task.actions.items():
            if action is not None:
                self.add_action(name, action)
        for name, command in task.commands.items():
            if command is not None:
                self.add_command(name, command)
        for name, quantity in task.quantities.items():
            if quantity is not None:
                self.add_quantity(name, quantity)
        for group, observations in task.observations.items():
            for name, observation in observations.terms.items():
                if observation is not None:
                    self.add_observation(group, name, observation)
        for name, reward in task.rewards.items():
            if reward is not None:
                self.add_reward(name, reward)
        for name, termination in task.terminations.items():
            if termination is not None:
                self.add_termination(name, termination)
        for name, event in task.events.items():
            if event is not None:
                self.add_event(name, event)

    def add_action(self, name: str, action: DirectAction) -> ScheduleEntry:
        return self._add(ScheduleEntry(("actions", name), action))

    def add_command(self, name: str, command: Command) -> ScheduleEntry:
        return self._add(ScheduleEntry(("commands", name), command))

    def add_quantity(self, name: str, quantity: Quantity) -> ScheduleEntry:
        return self._add(ScheduleEntry(("quantities", name), quantity))

    def add_event(self, name: str, event: Event) -> ScheduleEntry:
        return self._add(ScheduleEntry(("events", name), event))

    def add_observation(self, group: str, name: str, observation: Observation) -> ScheduleEntry:
        return self._add(ScheduleEntry(("observations", group, name), observation))

    def add_reward(self, name: str, reward: Reward) -> ScheduleEntry:
        return self._add(ScheduleEntry(("rewards", name), reward))

    def add_termination(self, name: str, termination: Termination | TimeLimit) -> ScheduleEntry:
        return self._add(ScheduleEntry(("terminations", name), termination))

    def entry(self, *path: str) -> ScheduleEntry:
        """Return the entry at a path, for example ``builder.entry("rewards", "alive")``."""
        return self._entries[path]

    def schedule(self) -> tuple[ScheduleEntry, ...]:
        """Return all entries in execution order: by phase, then by their order within the phase."""
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.phase))

    def move_before(self, entry: ScheduleEntry, anchor: ScheduleEntry) -> None:
        """Run ``entry`` just before ``anchor``, which must be in the same phase."""
        self._move(entry, anchor, 0)

    def move_after(self, entry: ScheduleEntry, anchor: ScheduleEntry) -> None:
        """Run ``entry`` just after ``anchor``, which must be in the same phase."""
        self._move(entry, anchor, 1)

    def set_order(self, phase: Phase, entries: list[ScheduleEntry]) -> None:
        """Set the execution order of all entries of a phase."""
        current = [entry for entry in self.schedule() if entry.phase is phase]
        if sorted(entry.path for entry in entries) != sorted(entry.path for entry in current):
            raise ValueError(
                f"The order of {phase.name} must list each of its entries once: {', '.join(map(str, current))}."
            )

        self._reorder([entry for entry in self.schedule() if entry.phase is not phase] + list(entries))

    def remove(self, entry: ScheduleEntry) -> None:
        del self._entries[entry.path]

    def describe(self) -> str:
        """Return the schedule as readable text, listing each use with the fields it reads and writes."""
        schedule = self.schedule()
        width = max((len(str(entry)) for entry in schedule), default=0)

        lines = []
        for phase, entries in itertools.groupby(schedule, lambda entry: entry.phase):
            lines.append(phase.name.lower().replace("_", " "))
            for entry in entries:
                fields = [
                    f"{verb} {', '.join(map(str, selectors))}"
                    for verb, selectors in (("reads", entry.reads), ("writes", entry.writes))
                    if selectors
                ]
                lines.append(f"  {str(entry):{width}}  {'; '.join(fields)}".rstrip())
        return "\n".join(lines)

    def inspect(self, entry: ScheduleEntry, backend: Backend | str = Backend.TORCH) -> ActionOp | TermOp | None:
        """Compile the schedule and return the op of an entry, or ``None`` for a time limit."""
        mdp = self._compile("cpu", Backend(backend))
        return {op.name: op for ops in mdp.ops.values() for op in ops}.get(str(entry))

    def preview_segments(self, backend: Backend | str = Backend.TORCH) -> dict[str, tuple[Segment, ...]]:
        """Return how the current order splits each runtime function into CUDA graphs and eager steps."""
        return self._compile("cpu", Backend(backend)).segments

    def validate(self, backend: Backend | str = Backend.TORCH) -> None:
        """Run all checks of :meth:`finalize` without generating kernels."""
        self._compile("cpu", Backend(backend))

    def finalize(self, device: str = "cpu", backend: Backend | str = Backend.TORCH, diagnostics: bool = False) -> MDP:
        """Validate the schedule and compile it into an immutable MDP.

        Raises a single ``ValueError`` that lists every error found. Editing the builder afterwards does not change the
        returned MDP. With ``diagnostics``, :func:`evaluate` also writes the reward and termination flag of each term.
        """
        mdp = self._compile(device, Backend(backend), diagnostics)
        backend_module = warp_backend if mdp.backend is Backend.WARP else torch_backend
        return replace(mdp, launches=backend_module.compile_launches(mdp))

    def _add(self, entry: ScheduleEntry) -> ScheduleEntry:
        if entry.path in self._entries:
            raise ValueError(f"The schedule already has an entry '{entry}'.")
        self._entries[entry.path] = entry
        return entry

    def _move(self, entry: ScheduleEntry, anchor: ScheduleEntry, offset: int) -> None:
        if entry.phase is not anchor.phase:
            raise ValueError(
                f"Cannot move '{entry}' next to '{anchor}': a use stays in its phase"
                f" ({entry.phase.name}, not {anchor.phase.name})."
            )

        order = [other for other in self.schedule() if other.path != entry.path]
        order.insert(order.index(anchor) + offset, entry)
        self._reorder(order)

    def _reorder(self, order: list[ScheduleEntry]) -> None:
        """Apply a new schedule order, or raise if a use would then read a field before it is written."""
        errors = _dependency_errors(sorted(order, key=lambda entry: entry.phase))
        if errors:
            raise ValueError("Invalid schedule:\n  " + "\n  ".join(errors))
        self._entries = {entry.path: entry for entry in order}

    def _compile(self, device: str, backend: Backend, diagnostics: bool = False) -> MDP:
        """Compile the schedule of each world config, then merge their ops, sharing the ones they have in common."""
        configs = self._world_configs or [(self.schedule(), range(self.schema.num_worlds))]
        world_ranges = [worlds for _, worlds in configs]

        errors = []
        if self._world_configs and self._entries:
            errors.append("Add term uses to the replicated builders, not to the builder that replicates them.")
        if world_ranges[-1].stop != self.schema.num_worlds:
            errors.append(f"The world configs cover {world_ranges[-1].stop} of {self.schema.num_worlds} worlds.")

        contexts, time_limits = [], set()
        for index, (schedule, worlds) in enumerate(configs):
            context, time_limit = self._compile_schedule(schedule, backend)
            prefix = f"world config {index} ({worlds.start}-{worlds.stop - 1}): " if len(configs) > 1 else ""
            errors += [prefix + error for error in context.errors]
            contexts.append(context)
            time_limits.add(time_limit)

        if len(time_limits) > 1:
            errors.append(f"The world configs must share one time limit, found {sorted(map(str, time_limits))} s.")
        field_shapes = {}
        for context in contexts:
            for op in context.ops[Phase.COMMANDS] + context.ops[Phase.QUANTITIES]:
                if field_shapes.setdefault(op.writes[0], op.shape) != op.shape:
                    errors.append(f"The world configs give '{op.writes[0]}' different shapes.")
        if errors:
            raise ValueError("Invalid MDP:\n  " + "\n  ".join(errors))

        ops = _merge_ops([(context.ops, worlds) for context, worlds in zip(contexts, world_ranges)], self.schema)

        # Random streams are keyed by use name, so that every world draws what it would in its config alone.
        streams: dict[str, int] = {}
        for phase in Phase:
            if phase is Phase.ACTIONS:
                continue
            ops[phase] = tuple(
                replace(
                    op,
                    stream=streams.setdefault(op.use, len(streams)) if op.random or op.timer is not None else None,
                )
                for op in ops[phase]
            )
        ops[Phase.OBSERVATIONS] = tuple(
            replace(
                op,
                observation=replace(op.observation, noise_stream=streams.setdefault(f"{op.use}/noise", len(streams))),
            )
            if op.observation.noise is not None
            else op
            for op in ops[Phase.OBSERVATIONS]
        )
        ops = MappingProxyType(ops)

        groups = dict.fromkeys(group for context in contexts for group in context.observation_dims)
        observation_dims = {
            group: max(context.observation_dims.get(group, 0) for context in contexts) for group in groups
        }
        subsets = {selector: ids for context in contexts for selector, ids in context.subsets.items()}
        time_limit = time_limits.pop()
        return MDP(
            schema=self.schema,
            device=device,
            backend=backend,
            ops=ops,
            plan=build_plan(ops),
            action_dim=max(context.action_dim for context in contexts),
            observation_dims=MappingProxyType(observation_dims),
            max_episode_length=None if time_limit is None else math.ceil(time_limit / self.schema.step_dt),
            streams=tuple(streams),
            subsets=MappingProxyType(subsets),
            diagnostics=diagnostics,
            world_configs=tuple(
                WorldConfig(worlds, context.action_dim, MappingProxyType(context.observation_dims))
                for context, worlds in zip(contexts, world_ranges)
            ),
        )

    def _compile_schedule(
        self, schedule: tuple[ScheduleEntry, ...], backend: Backend
    ) -> tuple[CompileContext, float | None]:
        """Compile the ops of one world config, and return them with its episode length in seconds."""
        context = CompileContext(backend, _dependency_errors(schedule))
        time_limits = []

        # Actions compile first, for the size of the last action. Every other use compiles after the uses that
        # provide the commands, quantities, and term state it reads.
        entries = {entry.path: entry for entry in schedule}
        providers = {selector: entry.path for entry in schedule for selector in entry.provides}
        reads = {
            path: {providers[selector] for selector in entry.reads if selector in providers} - {path}
            for path, entry in entries.items()
        }
        try:
            order = list(graphlib.TopologicalSorter(reads).static_order())
        except graphlib.CycleError as error:
            context.errors.append(
                f"Uses read each other's fields in a cycle: {', '.join(map('.'.join, error.args[1]))}."
            )
            order = list(entries)

        actions = [entry for entry in schedule if entry.phase is Phase.ACTIONS]
        for entry in (*actions, *(entries[path] for path in order if entries[path].phase is not Phase.ACTIONS)):
            if isinstance(entry.cfg, TimeLimit):
                time_limits.append(entry)
            elif isinstance(entry.cfg, DirectAction):
                self._compile_action(entry, context)
            else:
                self._compile_term(entry, context)

        # Restore the schedule order within each phase, with the positions and observation columns it gives.
        position = {str(entry): k for k, entry in enumerate(schedule)}
        for phase, ops in context.ops.items():
            if phase is Phase.ACTIONS:
                continue
            ops.sort(key=lambda op: position[op.name])
            starts = dict.fromkeys(context.observation_dims, 0)
            for k, op in enumerate(ops):
                observation = op.observation
                if observation is not None:
                    start = starts[observation.group]
                    stop = start + observation.columns.stop - observation.columns.start
                    starts[observation.group] = stop
                    observation = replace(observation, columns=slice(start, stop))
                ops[k] = replace(op, index=k, observation=observation)

        if len(time_limits) > 1:
            context.errors.append(f"Only one time limit is allowed, found {', '.join(map(str, time_limits))}.")
        return context, time_limits[0].cfg.episode_length_s if time_limits else None

    def _compile_action(self, entry: ScheduleEntry, context: CompileContext) -> None:
        """Compile an action use into an op, or record why it cannot be compiled."""
        cfg = entry.cfg
        target = self._resolve(entry, "target", cfg.target, context)
        if target is None:
            return
        if target.dtype is not wp.float32:
            context.errors.append(f"{entry}: the target '{cfg.target}' must be a float32 field.")

        elements = set(range(math.prod(target.sizes.values())))
        if cfg.subset is not None:
            dims = list(target.sizes)
            if cfg.subset.entity != cfg.target.entity or dims != [cfg.subset.dim]:
                context.errors.append(f"{entry}: the subset '{cfg.subset}' must select elements of '{cfg.target}'.")
                return
            if self._resolve(entry, "subset", cfg.subset, context) is None:
                return
            elements = set(context.subsets[cfg.subset])
        for op in context.ops[Phase.ACTIONS]:
            written = set(context.subsets[op.subset]) if op.subset is not None else elements
            if op.target == cfg.target and (op.subset is None or elements & written):
                context.errors.append(f"{entry}: '{cfg.target}' is already written by '{op.name}'.")

        if isinstance(cfg.offset, FieldSelector):
            offset = self._resolve(entry, "offset", cfg.offset, context)
            if offset is not None and offset.sizes != target.sizes:
                context.errors.append(f"{entry}: the offset '{cfg.offset}' must have the shape of '{cfg.target}'.")

        columns = slice(context.action_dim, context.action_dim + len(elements))
        context.action_dim = columns.stop
        context.mdp_fields[last_action] = ArrayLayout(
            wp.float32, 2, {"actions": context.action_dim}, (context.action_dim,)
        )
        action = ActionOp(str(entry), cfg.target, columns, cfg.scale, cfg.offset, subset=cfg.subset)
        context.ops[Phase.ACTIONS].append(action)

    def _compile_term(self, entry: ScheduleEntry, context: CompileContext) -> None:
        """Compile a term use into an op, or record why it cannot be compiled."""
        cfg, errors = entry.cfg, context.errors
        term = cfg.term
        num_errors = len(errors)

        # Check the use, and resolve the shapes of its output and state.
        implementation = _implementation(entry, context.backend, errors)
        outputs = cfg.outputs if isinstance(cfg, Event) else {}
        # A command writes its own field, which every hook gets as ``command``.
        command = {"command": entry.writes[0]} if isinstance(cfg, Command) else {}
        names = [*cfg.inputs, *outputs, *command, *term.state, *cfg.params, *(["rng"] if term.random else [])]
        if len(set(names)) != len(names):
            errors.append(f"{entry}: inputs, outputs, state, params, and rng must have distinct names, got {names}.")

        fields = {
            arg: self._resolve(entry, arg, selector, context) for arg, selector in {**cfg.inputs, **outputs}.items()
        }
        # Int params, such as a number of points, can also size dimensions.
        sizes = {name: value for name, value in cfg.params.items() if type(value) is int}
        sizes |= _dimension_sizes(fields)
        output_shape = _resolve_shape(entry, "output", term.shape, sizes, errors)
        if output_shape and isinstance(cfg, (Reward, Termination)):
            errors.append(f"{entry}: {type(cfg).__name__} outputs must have shape ('worlds',), got {output_shape}.")
        # State without the world dimension is one buffer for every world.
        shared = {name for name, spec in term.state.items() if "worlds" not in spec.shape}
        state_shapes = {}
        for name, spec in term.state.items():
            shape = ("worlds", *spec.shape) if name in shared else spec.shape
            dims = _resolve_shape(entry, f"state {name!r}", shape, sizes, errors)
            if dims is not None:
                state_shapes[name] = dims if name in shared else (self.schema.num_worlds, *dims)
        if isinstance(cfg, Event) and (cfg.mode is EventMode.INTERVAL) != (cfg.interval_range_s is not None):
            errors.append(f"{entry}: interval events need an interval_range_s, and other events cannot have one.")
        if isinstance(cfg, Quantity) and cfg.log and output_shape:
            errors.append(f"{entry}: a logged quantity must have shape ('worlds',), got {output_shape}.")
        if isinstance(cfg, Event) and cfg.commit and cfg.mode is not EventMode.RESET:
            errors.append(f"{entry}: only reset events can ask for a commit.")
        if isinstance(cfg, Command) and implementation is not None and implementation.reset is not None:
            errors.append(f"{entry}: commands write their state when they resample, so they have no reset hook.")
        if isinstance(cfg, Observation) and cfg.noise is not None and cfg.noise.scale is not None:
            layout = self._resolve(entry, "noise scale", cfg.noise.scale, context)
            if layout is not None and (layout.ndim != 1 or layout.dtype is not wp.float32):
                errors.append(f"{entry}: the noise scale '{cfg.noise.scale}' must be a float field of one per world.")

        if len(errors) > num_errors:
            return

        # Check the hooks' signatures. Params and rng are not arrays.
        dtype = wp.bool if isinstance(cfg, Termination) else wp.float32
        arguments = {
            **dict.fromkeys(names),
            **fields,
            **{name: ArrayLayout(wp.float32, len(shape), {}) for name, shape in state_shapes.items()},
            **{name: ArrayLayout(wp.float32, 1 + len(output_shape), {}) for name in command},
        }
        if term.init is not None:
            try:
                inspect.signature(term.init).bind(None, **dict.fromkeys(cfg.params))
            except TypeError as error:
                errors.append(f"{entry}: init {term.init.__name__!r} is called with (ctx, **params): {error}.")
        # Events and commands take the mask of the worlds to write; other Warp functions take the output to write.
        if term.eager is not None:
            try:
                inspect.signature(term.eager).bind(None, **dict.fromkeys(cfg.params))
            except TypeError as error:
                errors.append(f"{entry}: eager term {term.eager.__name__!r} is called with (ctx, **params): {error}.")
        elif isinstance(cfg, (Command, Event)):
            _check_hook(entry, implementation.compute, arguments, errors, extra="mask")
        else:
            extra = "out" if context.backend is Backend.WARP else None
            _check_hook(entry, implementation.compute, arguments, errors, len(output_shape), extra, dtype)
        # A function update hook may also take the worlds where its op runs, as ``mask``.
        update_arguments = arguments if isinstance(implementation.update, wp.Function) else {**arguments, "mask": None}
        _check_hook(entry, implementation.update, update_arguments, errors)
        _check_hook(entry, implementation.reset, arguments, errors, extra="mask")
        if len(errors) > num_errors:
            return

        # Assign its observation columns and its field, then add its op.
        timer = None
        if isinstance(cfg, Command):
            timer = cfg.resampling_time_range
        elif isinstance(cfg, Event):
            timer = cfg.interval_range_s

        observation = None
        if isinstance(cfg, Observation):
            group = entry.path[1]
            start = context.observation_dims.get(group, 0)
            columns = slice(start, start + math.prod(output_shape) * max(cfg.history_length, 1))
            context.observation_dims[group] = columns.stop
            observation = ObservationOutput(group, columns, cfg.noise, cfg.clip, cfg.scale, cfg.history_length)
        if isinstance(cfg, (Command, Quantity)):
            named = {dim: size for dim, size in zip(term.shape[1:], output_shape) if isinstance(dim, str)}
            context.mdp_fields[entry.writes[0]] = ArrayLayout(wp.float32, 1 + len(output_shape), named, output_shape)
        # Other terms can read its state.
        for name, shape in state_shapes.items():
            selector = term_state(str(entry), name)
            context.mdp_fields[selector] = ArrayLayout(wp.float32, len(shape), {}, () if name in shared else shape[1:])

        context.ops[cfg.phase].append(
            TermOp(
                name=str(entry),
                use=str(entry),
                phase=cfg.phase,
                index=len(context.ops[cfg.phase]),
                compute=implementation.compute,
                update=implementation.update,
                reset=implementation.reset,
                inputs=MappingProxyType({**cfg.inputs, **outputs, **command}),
                params=MappingProxyType(
                    {name: self.schema.step_dt if value is step_dt else value for name, value in cfg.params.items()}
                ),
                state=MappingProxyType(
                    {
                        name: replace(spec, shape=state_shapes[name], persistent=spec.persistent or name in shared)
                        for name, spec in term.state.items()
                    }
                ),
                shape=output_shape,
                dtype=dtype,
                writes=entry.writes,
                weights=(cfg.weight,) if isinstance(cfg, Reward) else (),
                time_out=cfg.time_out if isinstance(cfg, Termination) else False,
                observation=observation,
                timer=timer,
                random=term.random,
                capturable=term.capturable and term.eager is None,
                eager=term.eager is not None,
                init=term.init,
                commit=isinstance(cfg, Event) and cfg.commit,
                log=isinstance(cfg, Quantity) and cfg.log,
            )
        )

    def _resolve(
        self, entry: ScheduleEntry, arg: str, selector: FieldSelector | SubsetSelector, context: CompileContext
    ) -> ArrayLayout | None:
        """Look up the array that a selector binds to the argument ``arg``, or report an error and return ``None``."""
        if isinstance(selector, FieldSelector) and not selector.scene:
            if selector not in context.mdp_fields:
                fields = sorted(map(str, context.mdp_fields))
                context.errors.append(f"{entry}: unknown field '{selector}'. Fields of the MDP: {fields}.")
            return context.mdp_fields.get(selector)

        entity = self.schema.entities.get(selector.entity)
        if entity is None:
            context.errors.append(f"{entry}: unknown entity {selector.entity!r} in '{selector}'.")
            return None

        if isinstance(selector, FieldSelector):
            if selector.field not in entity.fields:
                context.errors.append(f"{entry}: unknown field '{selector}'. Fields: {sorted(entity.fields)}.")
                return None
            schema = entity.fields[selector.field]
            sizes = {dim: len(entity.names[dim]) for dim in schema.dims}
            return ArrayLayout(schema.dtype, 1 + len(schema.dims), sizes, tuple(sizes.values()))

        names = entity.names.get(selector.dim, ())
        unmatched = [pattern for pattern in selector.patterns if not any(re.fullmatch(pattern, name) for name in names)]
        if unmatched or not selector.patterns:
            context.errors.append(f"{entry}: '{selector}' matches no name with {unmatched}. Names: {list(names)}.")
            return None
        context.subsets[selector] = tuple(
            index
            for index, name in enumerate(names)
            if any(re.fullmatch(pattern, name) for pattern in selector.patterns)
        )
        return ArrayLayout(wp.int32, 1, {arg: len(context.subsets[selector])})


def _implementation(entry: ScheduleEntry, backend: Backend, errors: list[str]) -> Implementation | None:
    """Return the implementation of a term use for a backend, or record why it has none."""
    term = entry.cfg.term
    if term.eager is not None:
        if term.torch is not None or term.warp is not None:
            errors.append(f"{entry}: an eager term cannot also have Torch or Warp implementations.")
        return Implementation(compute=term.eager)

    implementation = term.torch if backend is Backend.TORCH else term.warp
    if implementation is None:
        errors.append(f"{entry}: the term has no '{backend.value}' implementation.")
    elif not term.capturable and any(isinstance(hook, wp.Function) for hook in vars(implementation).values()):
        errors.append(f"{entry}: wp.func hooks always run inside generated kernels and cannot be non-capturable.")
    return implementation


def _dimension_sizes(fields: Mapping[str, ArrayLayout | None]) -> dict[str, int]:
    """Return the size of each named dimension of a term's inputs.

    The argument name of a subset, or of a field with one dimension after the worlds, is also a dimension. A name
    that inputs give different sizes, such as the bodies of two entities, is left out.
    """
    sizes: dict[str, int] = {}
    ambiguous = set()
    for arg, layout in fields.items():
        if layout is None:
            continue
        named = dict(layout.sizes)
        if len(layout.shape) == 1:
            named[arg] = layout.shape[0]
        for dim, size in named.items():
            if sizes.setdefault(dim, size) != size:
                ambiguous.add(dim)
    return {dim: size for dim, size in sizes.items() if dim not in ambiguous}


def _check_hook(
    entry: ScheduleEntry,
    hook: Callable | None,
    arguments: Mapping[str, ArrayLayout | None],
    errors: list[str],
    indices: int = 0,
    extra: str | None = None,
    returns: type | None = None,
) -> None:
    """Check that a hook's signature matches the arguments it will be called with.

    A ``wp.func`` must take the world index, ``indices`` output indices, and any of ``arguments`` with matching array
    types, and return ``returns``. A Python function must accept ``extra`` and the ``arguments`` it names as keywords.
    """
    if isinstance(hook, wp.Function):
        annotations = dict(list(hook.input_types.items())[1 + indices :])
        unknown = sorted(set(annotations) - set(arguments))
        if len(hook.input_types) < 1 + indices or unknown:
            errors.append(
                f"{entry}: wp.func {hook.key!r} must take the world, {indices} output indices, and any of"
                f" {sorted(arguments)}, got {list(hook.input_types)}."
            )
            return
        for name, annotation in annotations.items():
            bound = arguments[name]
            if bound is None or not isinstance(annotation, wp.array):
                continue
            if annotation.dtype is not bound.dtype or annotation.ndim != bound.ndim:
                errors.append(
                    f"{entry}: wp.func {hook.key!r} takes {name!r} as a {annotation.ndim}-dimensional array of"
                    f" {annotation.dtype.__name__}, but it is a {bound.ndim}-dimensional array of"
                    f" {bound.dtype.__name__}."
                )

        annotation = typing.get_type_hints(hook.func).get("return")
        # An array annotation holds the Warp type of a Python type.
        if returns is not None and annotation is not None and wp.array(dtype=annotation).dtype is not returns:
            errors.append(f"{entry}: wp.func {hook.key!r} must return {returns.__name__}.")
    elif hook is not None:
        names = sorted({*hook_arguments(hook, arguments), extra} - {None})
        try:
            inspect.signature(hook).bind(**dict.fromkeys(names))
        except TypeError as error:
            errors.append(f"{entry}: {hook.__name__!r} is called with {names} of {sorted(arguments)}: {error}.")


def _resolve_shape(
    entry: ScheduleEntry, role: str, shape: tuple[str | int, ...], sizes: dict[str, int], errors: list[str]
) -> tuple[int, ...] | None:
    """Resolve a symbolic shape's dimensions after ``"worlds"`` to sizes, or report an error and return ``None``."""
    if shape[:1] != ("worlds",) or "worlds" in shape[1:]:
        errors.append(f"{entry}: the {role} shape {shape} must start with 'worlds' and contain it once.")
        return None
    unknown = [dim for dim in shape[1:] if isinstance(dim, str) and dim not in sizes]
    if unknown:
        errors.append(
            f"{entry}: the {role} dimensions {unknown} are not dimensions of the term's inputs, or inputs give them"
            " different sizes."
        )
        return None
    return tuple(sizes.get(dim, dim) for dim in shape[1:])


def _dependency_errors(schedule: tuple[ScheduleEntry, ...] | list[ScheduleEntry]) -> list[str]:
    """Report each use that reads a field before a later use of the same phase writes it."""
    errors = []
    for index, reader in enumerate(schedule):
        for writer in itertools.takewhile(lambda entry: entry.phase is reader.phase, schedule[index + 1 :]):
            for selector in dict.fromkeys(reader.reads):
                if selector in writer.writes:
                    errors.append(f"'{reader}' reads '{selector}', so it must run after '{writer}', which writes it.")
    return errors


def _merge_ops(
    configs: list[tuple[dict[Phase, list], range]], schema: SceneSchema
) -> dict[Phase, tuple[ActionOp | TermOp, ...]]:
    """Merge the ops of the world configs, phase by phase, keeping the order of each config.

    An op equal to one of another config that is still ahead in the merged order becomes one op active in the worlds
    of both, with the reward weight of each config. Other ops with the same name get a suffix with the index of their
    config.
    """
    merged: dict[Phase, list[list]] = {phase: [] for phase in Phase}  # Per phase: [op, worlds, config index, weights]
    for index, (ops, worlds) in enumerate(configs):
        for phase in Phase:
            phase_ops, cursor, unshared = merged[phase], 0, []
            for op in ops[phase]:
                # Ops of different configs match regardless of their index and of their reward weight, which is
                # read per world.
                key = op if phase is Phase.ACTIONS else replace(op, index=0, weights=())
                weights = {index: op.weights[0]} if phase is Phase.REWARDS else {}
                match = next((i for i in range(cursor, len(phase_ops)) if phase_ops[i][0] == key), None)
                if match is None:
                    unshared.append([key, list(worlds), index, weights])
                    continue
                # Ops of this config that no other config shares go just before its next shared op.
                phase_ops[match:match] = unshared
                match += len(unshared)
                phase_ops[match][1] += worlds
                phase_ops[match][3] |= weights
                cursor, unshared = match + 1, []
            phase_ops += unshared

    result = {}
    for phase, phase_ops in merged.items():
        counts = collections.Counter(op.name for op, _, _, _ in phase_ops)
        result[phase] = tuple(
            replace(
                op,
                **({} if phase is Phase.ACTIONS else {"index": position}),
                **({"weights": tuple(weights.get(k, 0.0) for k in range(len(configs)))} if weights else {}),
                name=op.name if counts[op.name] == 1 else f"{op.name}[{index}]",
                worlds=None if len(worlds) == schema.num_worlds else tuple(sorted(worlds)),
            )
            for position, (op, worlds, index, weights) in enumerate(phase_ops)
        )
    return result
