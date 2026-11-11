# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builder that collects term uses into a schedule and compiles it into an :class:`MDP`, like Newton's ModelBuilder."""

from __future__ import annotations

import collections
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
from .config import (
    Command,
    DirectAction,
    Event,
    EventMode,
    FieldSelector,
    Observation,
    Phase,
    Reward,
    SubsetSelector,
    TaskCfg,
    TermCfg,
    Termination,
    TimeLimit,
    commands,
    episode_length,
    last_action,
)
from .mdp import MDP, Backend, WorldConfig
from .plan import ActionOp, Segment, TermOp, build_plan
from .scene import SceneSchema
from .terms import Implementation


@dataclass(frozen=True)
class ScheduleEntry:
    """Handle to one term use in the schedule, returned by the ``add_*`` methods."""

    path: tuple[str, ...]  # Location in the task, e.g. ("observations", "policy", "joint_pos")
    cfg: DirectAction | Command | Observation | Reward | Termination | TimeLimit | Event

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
        return tuple(selector for selector in self.cfg.inputs.values() if isinstance(selector, FieldSelector))

    @property
    def writes(self) -> tuple[FieldSelector, ...]:
        """Fields the use writes: an action's target, a command's value, or an event's outputs."""
        if isinstance(self.cfg, DirectAction):
            return (self.cfg.target,)
        if isinstance(self.cfg, Command):
            return (FieldSelector(commands.name, self.path[1], scene=False),)
        return tuple(self.cfg.outputs.values()) if isinstance(self.cfg, Event) else ()


@dataclass(frozen=True)
class ArrayLayout:
    """Element type, number of dimensions, and named dimension sizes of an array passed to a hook."""

    dtype: type
    ndim: int
    sizes: Mapping[str, int]


@dataclass
class CompileContext:
    """Ops, layouts, random streams, and errors collected while compiling the schedule."""

    backend: Backend
    errors: list[str]
    ops: dict[Phase, list] = field(default_factory=lambda: {phase: [] for phase in Phase})
    mdp_fields: dict[FieldSelector, ArrayLayout] = field(  # Fields owned by the MDP; commands are added as they compile
        default_factory=lambda: {
            last_action: ArrayLayout(wp.float32, 2, {"actions": 0}),
            episode_length: ArrayLayout(wp.int32, 1, {}),
        }
    )
    subsets: dict[SubsetSelector, tuple[int, ...]] = field(default_factory=dict)
    observation_dims: dict[str, int] = field(default_factory=dict)
    streams: list[str] = field(default_factory=list)
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

    def add_event(self, name: str, event: Event) -> ScheduleEntry:
        return self._add(ScheduleEntry(("events", name), event))

    def add_observation(self, group: str, name: str, observation: Observation) -> ScheduleEntry:
        return self._add(ScheduleEntry(("observations", group, name), observation))

    def add_reward(self, name: str, reward: Reward) -> ScheduleEntry:
        return self._add(ScheduleEntry(("rewards", name), reward))

    def add_termination(self, name: str, termination: Termination | TimeLimit) -> ScheduleEntry:
        return self._add(ScheduleEntry(("terminations", name), termination))

    def entry(self, *path: str) -> ScheduleEntry:
        """The entry at a path, for example ``builder.entry("rewards", "alive")``."""
        return self._entries[path]

    def schedule(self) -> tuple[ScheduleEntry, ...]:
        """All entries in execution order: by phase, then by their order within the phase."""
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

    def _compile(self, device: str, backend: Backend, diagnostics: bool = False) -> MDP:
        """Compile the schedule of each world config, then merge their ops, sharing the ones they have in common."""
        configs = self._world_configs or [(self.schedule(), range(self.schema.num_worlds))]
        errors = []
        if self._world_configs and self._entries:
            errors.append("Add term uses to the replicated builders, not to the builder that replicates them.")
        if configs[-1][1].stop != self.schema.num_worlds:
            errors.append(f"The world configs cover {configs[-1][1].stop} of {self.schema.num_worlds} worlds.")
        contexts = []
        for index, (schedule, worlds) in enumerate(configs):
            context, time_limit = self._compile_schedule(schedule, backend)
            prefix = f"world config {index} ({worlds.start}-{worlds.stop - 1}): " if len(configs) > 1 else ""
            errors += [prefix + error for error in context.errors]
            contexts.append((context, time_limit, worlds))
        time_limits = {time_limit for _, time_limit, _ in contexts}
        if len(time_limits) > 1:
            errors.append(f"The world configs must share one time limit, found {sorted(map(str, time_limits))} s.")
        command_shapes = {}
        for context, _, _ in contexts:
            for op in context.ops[Phase.COMMANDS]:
                if command_shapes.setdefault(op.writes[0], op.shape) != op.shape:
                    errors.append(f"The world configs give '{op.writes[0]}' different shapes.")
        if errors:
            raise ValueError("Invalid MDP:\n  " + "\n  ".join(errors))
        ops = MappingProxyType(_merge_ops([(context.ops, worlds) for context, _, worlds in contexts], self.schema))
        streams = tuple(
            dict.fromkeys(
                op.use
                for phase, phase_ops in ops.items()
                if phase is not Phase.ACTIONS
                for op in phase_ops
                if op.stream is not None
            )
        )
        ops = MappingProxyType(
            {
                phase: tuple(
                    op if phase is Phase.ACTIONS or op.stream is None else replace(op, stream=streams.index(op.use))
                    for op in phase_ops
                )
                for phase, phase_ops in ops.items()
            }
        )
        time_limit = time_limits.pop()
        return MDP(
            schema=self.schema,
            device=device,
            backend=backend,
            ops=ops,
            plan=build_plan(ops),
            action_dim=max(context.action_dim for context, _, _ in contexts),
            observation_dims=MappingProxyType(
                {
                    group: max(context.observation_dims.get(group, 0) for context, _, _ in contexts)
                    for context, _, _ in contexts
                    for group in context.observation_dims
                }
            ),
            max_episode_length=None if time_limit is None else math.ceil(time_limit / self.schema.step_dt),
            streams=streams,
            subsets=MappingProxyType({k: v for context, _, _ in contexts for k, v in context.subsets.items()}),
            diagnostics=diagnostics,
            world_configs=tuple(
                WorldConfig(worlds, context.action_dim, MappingProxyType(context.observation_dims))
                for context, _, worlds in contexts
            ),
        )

    def _compile_schedule(
        self, schedule: tuple[ScheduleEntry, ...], backend: Backend
    ) -> tuple[CompileContext, float | None]:
        """Compile the ops of one world config, and return them with its episode length in seconds."""
        context = CompileContext(backend, _dependency_errors(schedule))
        time_limits = []
        # Actions and commands compile first, so that terms of earlier phases can read them.
        for entry in sorted(schedule, key=lambda entry: entry.phase not in (Phase.ACTIONS, Phase.COMMANDS)):
            if isinstance(entry.cfg, TimeLimit):
                time_limits.append(entry)
            elif isinstance(entry.cfg, DirectAction):
                self._compile_action(entry, context)
            else:
                self._compile_term(entry, context)
        if len(time_limits) > 1:
            context.errors.append(f"Only one time limit is allowed, found {', '.join(map(str, time_limits))}.")
        return context, time_limits[0].cfg.episode_length_s if time_limits else None

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

    def _compile_action(self, entry: ScheduleEntry, context: CompileContext) -> None:
        cfg = entry.cfg
        target = self._resolve(entry, "target", cfg.target, context)
        if target is None:
            return
        for op in context.ops[Phase.ACTIONS]:
            if op.target == cfg.target:
                context.errors.append(f"{entry}: '{cfg.target}' is already the target of '{op.name}'.")
        if target.dtype is not wp.float32:
            context.errors.append(f"{entry}: the target '{cfg.target}' must be a float32 field.")
        columns = slice(context.action_dim, context.action_dim + math.prod(target.sizes.values()))
        context.action_dim = columns.stop
        context.mdp_fields[last_action] = ArrayLayout(wp.float32, 2, {"actions": context.action_dim})
        context.ops[Phase.ACTIONS].append(ActionOp(str(entry), cfg.target, columns, cfg.scale, cfg.offset))

    def _compile_term(self, entry: ScheduleEntry, context: CompileContext) -> None:
        """Compile a term use into an op, or record why it cannot be compiled."""
        cfg, errors = entry.cfg, context.errors
        num_errors = len(errors)
        term = cfg.term
        if term.eager is not None:
            implementation = Implementation(compute=term.eager)
            if term.torch is not None or term.warp is not None:
                errors.append(f"{entry}: an eager term cannot also have Torch or Warp implementations.")
        else:
            implementation = term.torch if context.backend is Backend.TORCH else term.warp
            if implementation is None:
                errors.append(f"{entry}: the term has no '{context.backend.value}' implementation.")
            elif not term.capturable and any(isinstance(hook, wp.Function) for hook in vars(implementation).values()):
                errors.append(
                    f"{entry}: wp.func hooks always run inside generated kernels and cannot be non-capturable."
                )
        outputs = cfg.outputs if isinstance(cfg, Event) else {}
        names = [*cfg.inputs, *outputs, *cfg.term.state, *cfg.params, *(["rng"] if cfg.term.random else [])]
        if len(set(names)) != len(names):
            errors.append(f"{entry}: inputs, outputs, state, params, and rng must have distinct names, got {names}.")
        fields = {
            arg: self._resolve(entry, arg, selector, context) for arg, selector in {**cfg.inputs, **outputs}.items()
        }
        sizes: dict[str, int] = {}
        for resolved in filter(None, fields.values()):
            for dim, size in resolved.sizes.items():
                if sizes.setdefault(dim, size) != size:
                    errors.append(f"{entry}: inputs disagree on the size of {dim!r} ({sizes[dim]} and {size}).")
        output_shape = _resolve_shape(entry, "output", cfg.term.shape, sizes, errors)
        if output_shape and isinstance(cfg, (Reward, Termination)):
            errors.append(f"{entry}: {type(cfg).__name__} outputs must have shape ('worlds',), got {output_shape}.")
        state_shapes = {
            name: _resolve_shape(entry, f"state {name!r}", spec.shape, sizes, errors)
            for name, spec in cfg.term.state.items()
        }
        if isinstance(cfg, (Command, Event)) and cfg.term.state:
            errors.append(f"{entry}: commands and events cannot keep state.")
        if isinstance(cfg, Event) and (cfg.mode is EventMode.INTERVAL) != (cfg.interval_range_s is not None):
            errors.append(f"{entry}: interval events need an interval_range_s, and other events cannot have one.")
        dtype = wp.bool if isinstance(cfg, Termination) else wp.float32
        if len(errors) == num_errors:
            # Params and rng are not arrays.
            arguments = {
                **dict.fromkeys(names),
                **fields,
                **{name: ArrayLayout(wp.float32, 1 + len(shape), {}) for name, shape in state_shapes.items()},
            }
            # Events take the mask of the worlds to write; other Warp functions take the output to write.
            if term.eager is not None:
                try:
                    inspect.signature(term.eager).bind(None, **dict.fromkeys(cfg.params))
                except TypeError as error:
                    errors.append(
                        f"{entry}: eager term {term.eager.__name__!r} is called with (ctx, **params): {error}."
                    )
            elif isinstance(cfg, Event):
                _check_hook(entry, implementation.compute, arguments, errors, extra="mask")
            else:
                extra = "out" if context.backend is Backend.WARP else None
                _check_hook(entry, implementation.compute, arguments, errors, len(output_shape), extra, dtype)
            _check_hook(entry, implementation.update, arguments, errors)
            _check_hook(entry, implementation.reset, arguments, errors, extra="mask")
        if len(errors) > num_errors:
            return

        timer = (
            cfg.resampling_time_range
            if isinstance(cfg, Command)
            else cfg.interval_range_s
            if isinstance(cfg, Event)
            else None
        )
        stream = None
        if timer is not None or cfg.term.random:
            stream = len(context.streams)
            context.streams.append(str(entry))

        group = ""
        columns = slice(None)
        if isinstance(cfg, Observation):
            group = entry.path[1]
            start = context.observation_dims.get(group, 0)
            columns = slice(start, start + math.prod(output_shape))
            context.observation_dims[group] = columns.stop
        if isinstance(cfg, Command):
            dims = zip(cfg.term.shape[1:], output_shape)
            named = {dim: size for dim, size in dims if isinstance(dim, str)}
            context.mdp_fields[entry.writes[0]] = ArrayLayout(wp.float32, 1 + len(output_shape), named)
        context.ops[cfg.phase].append(
            TermOp(
                name=str(entry),
                use=str(entry),
                phase=cfg.phase,
                index=len(context.ops[cfg.phase]),
                compute=implementation.compute,
                update=implementation.update,
                reset=implementation.reset,
                inputs=MappingProxyType({**cfg.inputs, **outputs}),
                params=MappingProxyType(dict(cfg.params)),
                state=MappingProxyType(
                    {
                        name: replace(spec, shape=(self.schema.num_worlds, *state_shapes[name]))
                        for name, spec in cfg.term.state.items()
                    }
                ),
                shape=output_shape,
                dtype=dtype,
                writes=entry.writes,
                weight=cfg.weight if isinstance(cfg, Reward) else 1.0,
                time_out=cfg.time_out if isinstance(cfg, Termination) else False,
                group=group,
                columns=columns,
                timer=timer,
                stream=stream,
                random=cfg.term.random,
                capturable=cfg.term.capturable and cfg.term.eager is None,
                eager=cfg.term.eager is not None,
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
            return ArrayLayout(schema.dtype, 1 + len(schema.dims), {dim: len(entity.names[dim]) for dim in schema.dims})
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

    A ``wp.func`` must take the world index, ``indices`` output indices, and ``arguments`` with matching array types,
    and return ``returns``. A Python function must accept ``arguments`` and ``extra`` as keywords.
    """
    if isinstance(hook, wp.Function):
        annotations = dict(list(hook.input_types.items())[1 + indices :])
        if set(annotations) != set(arguments):
            errors.append(
                f"{entry}: wp.func {hook.key!r} must take the world, {indices} output indices, and"
                f" {sorted(arguments)}, got {list(hook.input_types)}."
            )
            return
        for name, bound in arguments.items():
            annotation = annotations[name]
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
        names = sorted({*arguments, extra} - {None})
        try:
            inspect.signature(hook).bind(**dict.fromkeys(names))
        except TypeError as error:
            errors.append(f"{entry}: {hook.__name__!r} is called with {names}: {error}.")


def _resolve_shape(
    entry: ScheduleEntry, role: str, shape: tuple[str | int, ...], sizes: dict[str, int], errors: list[str]
) -> tuple[int, ...] | None:
    """Resolve a symbolic shape's dimensions after ``"worlds"`` to sizes, or report an error and return ``None``."""
    if shape[:1] != ("worlds",) or "worlds" in shape[1:]:
        errors.append(f"{entry}: the {role} shape {shape} must start with 'worlds' and contain it once.")
        return None
    unknown = [dim for dim in shape[1:] if isinstance(dim, str) and dim not in sizes]
    if unknown:
        errors.append(f"{entry}: the {role} dimensions {unknown} are not dimensions of the term's inputs.")
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
    of both. Other ops with the same name get a suffix with the index of their config.
    """
    merged: dict[Phase, list[list]] = {phase: [] for phase in Phase}  # Per phase: [op, worlds, config index]
    for index, (ops, worlds) in enumerate(configs):
        for phase in Phase:
            phase_ops, cursor, unshared = merged[phase], 0, []
            for op in ops[phase]:
                key = op if phase is Phase.ACTIONS else replace(op, index=0, stream=op.stream is not None)
                match = next((i for i in range(cursor, len(phase_ops)) if phase_ops[i][0] == key), None)
                if match is None:
                    unshared.append([key, list(worlds), index])
                    continue
                # Ops of this config that no other config shares go just before its next shared op.
                phase_ops[match:match] = unshared
                match += len(unshared)
                phase_ops[match][1] += worlds
                cursor, unshared = match + 1, []
            phase_ops += unshared
    result = {}
    for phase, phase_ops in merged.items():
        counts = collections.Counter(op.name for op, _, _ in phase_ops)
        result[phase] = tuple(
            replace(
                op,
                **({} if phase is Phase.ACTIONS else {"index": position, "stream": 0 if op.stream else None}),
                name=op.name if counts[op.name] == 1 else f"{op.name}[{index}]",
                worlds=None if len(worlds) == schema.num_worlds else tuple(sorted(worlds)),
            )
            for position, (op, worlds, index) in enumerate(phase_ops)
        )
    return result
