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
    PROGRESS_FIELDS,
    Action,
    ActionRate,
    Command,
    DirectAction,
    Entity,
    Event,
    EventMode,
    FieldSelector,
    Observation,
    Phase,
    Quantity,
    Reward,
    SubsetSelector,
    TaskCfg,
    TaskMode,
    TermCfg,
    Termination,
    TimeLimit,
    Tunable,
    commands,
    episode_length,
    group_terms,
    last_action,
    progress_field,
    quantities,
    step_dt,
    term_state,
    terminated,
    tunable,
)
from .plan import (
    ActionOp,
    ActionSlot,
    ObservationOutput,
    ObservationSlot,
    Segment,
    TaskNode,
    TaskTree,
    TermOp,
    TunedValue,
    build_plan,
)
from .scene import SceneSchema
from .terms import Implementation, hook_arguments


@dataclass(frozen=True)
class ScheduleEntry:
    """Handle to one term use in the schedule, returned by the ``add_*`` methods."""

    path: tuple[str, ...]  # Location in the task, e.g. ("observations", "policy", "joint_pos")
    cfg: DirectAction | Action | Command | Quantity | Observation | Reward | Termination | TimeLimit | Event
    concatenate: bool = True  # Observations: whether the use writes its group's buffer rather than its own
    name: str = ""  # Name in its task, without the prefix of a subtask; the last part of the path if empty
    stage: str | None = None  # The task of a sequence that the use belongs to, see TermOp.stage

    @property
    def buffer(self) -> str:
        """Observations: the buffer the use writes, its group or, in a group that is not concatenated, "group/name".

        Uses of different tasks with the same name share their columns, see :meth:`MDPBuilder.finalize`.
        """
        return self.path[1] if self.concatenate else f"{self.path[1]}/{self.name or self.path[-1]}"

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
    def when(self) -> FieldSelector | None:
        """The field that switches the use on and off, which it reads at the start of a step, if any."""
        if not isinstance(self.cfg, TermCfg) or self.cfg.when is None:
            return None
        return tunable(str(self), "when") if isinstance(self.cfg.when, Tunable) else self.cfg.when

    @property
    def writes(self) -> tuple[FieldSelector, ...]:
        """Fields the use writes: an action's target, the value of a command or quantity, or an event's outputs."""
        if isinstance(self.cfg, DirectAction):
            return (self.cfg.target,)
        if isinstance(self.cfg, Command):
            return (FieldSelector(commands.name, self.path[1], scene=False),)
        if isinstance(self.cfg, Quantity):
            return (FieldSelector(quantities.name, self.path[1], scene=False),)
        if isinstance(self.cfg, (Event, Action)):
            return tuple(self.cfg.outputs.values())
        return ()

    @property
    def provides(self) -> tuple[FieldSelector, ...]:
        """Fields of the MDP that other uses can read once this one is compiled: its value, state, and tunables."""
        if not isinstance(self.cfg, TermCfg):
            return ()
        state = [term_state(str(self), name) for name in self.cfg.term.state]
        tunables = [tunable(str(self), name) for name, value in self.cfg.params.items() if isinstance(value, Tunable)]
        tunables += [self.when] if isinstance(self.cfg.when, Tunable) else []
        return (*(selector for selector in self.writes if not selector.scene), *state, *tunables)


@dataclass(frozen=True)
class ArrayLayout:
    """Element type, number of dimensions, and dimension sizes of an array passed to a hook."""

    dtype: type
    ndim: int
    sizes: Mapping[str, int]  # Sizes of the named dimensions
    shape: tuple[int, ...] = ()  # Sizes of the dimensions after the world dimension, named or not


@dataclass
class CompileContext:
    """Ops, layouts, and errors collected while compiling the schedule."""

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
    tasks: tuple[TaskNode, ...] = ()  # The task tree, parents first, if the MDP tracks progress
    modes: dict[str, TaskMode] = field(default_factory=dict)  # How the subtasks of each task run, by path


class MDPBuilder:
    """Collects term uses, lets users edit their order, and compiles them with :meth:`finalize`.

    Uses run phase by phase. Within a phase they run in the order they were added, unless the order is edited with
    :meth:`move_before`, :meth:`move_after`, or :meth:`set_order`. Every edit is checked so that a use that reads a
    field still runs after the uses of its phase that write it. Action and observation columns follow the schedule
    order.
    """

    def __init__(self, schema: SceneSchema) -> None:
        self.schema = schema
        self._entries: dict[tuple[str, ...], ScheduleEntry] = {}
        # The tasks added with add_task by dotted path, "" for the root, and whether each is part of a sequence
        self._tasks: dict[str, tuple[TaskCfg, bool]] = {}
        # Schedules and tasks of replicated builders, and their worlds
        self._world_configs: list[tuple[tuple[ScheduleEntry, ...], dict[str, tuple[TaskCfg, bool]], range]] = []

    def replicate(self, builder: MDPBuilder, num_worlds: int) -> None:
        """Run the task of another builder in the next ``num_worlds`` worlds.

        The builder's schedule is copied, so later edits to it do not apply. Once worlds are assigned, every world
        must be, and term uses are added to the replicated builders rather than to this one.
        """
        if builder.schema != self.schema:
            raise ValueError("A replicated builder must be built against the same scene schema.")

        start = self._world_configs[-1][2].stop if self._world_configs else 0
        self._world_configs.append((builder.schedule(), dict(builder._tasks), range(start, start + num_worlds)))

    def add_task(self, task: TaskCfg) -> None:
        """Add the term uses of a task and of its subtasks in declaration order, skipping uses set to ``None``.

        The uses of a subtask get its dotted path as a prefix, as in ``rewards.grasp.lift``. A task that splits the
        worlds between subtasks replicates a builder with the rest of the task for each of them.
        """
        configs = _expand(task, self.schema.num_worlds)
        if len(configs) == 1:
            self._add_task(configs[0][0], (), (), False, {})
            return
        for config, num_worlds in configs:
            builder = MDPBuilder(self.schema)
            builder._add_task(config, (), (), False, {})
            self.replicate(builder, num_worlds)

    def _add_task(
        self,
        task: TaskCfg,
        path: tuple[str, ...],
        names: tuple[str, ...],
        staged: bool,
        scope: dict[FieldSelector, FieldSelector],
        restarts: bool = False,
    ) -> None:
        """Add a task of one world config, its uses with the prefix ``names``, then its subtasks. ``staged`` is
        whether it is part of a sequence, and ``restarts`` whether it or a parent repeats.

        The subtask of a split runs in worlds of its own, so its uses keep their names: equal uses of the world configs
        then run once, see :meth:`replicate`. A task reads its own fields by their names, as a task that runs alone
        does: its commands and quantities, the state and tunable values of its uses, and its progress and that of its
        subtasks. ``scope`` maps the names that the task and its parents define to their fields; inner names win.
        """
        task_path = ".".join(path)
        if task_path in self._tasks:
            raise ValueError(f"The builder already has a task '{task_path}'.")
        if task.done is not None and task.subtasks:
            raise ValueError(f"Task '{task_path}': a task with subtasks is done when they are; it cannot have a done.")
        prefix = ".".join(names)
        restarts = restarts or task.repeat
        stage = task_path if staged or restarts else None
        scope = {**scope, **_local_names(task, task_path, prefix)}
        self._tasks[task_path] = (replace(task, done=scope.get(task.done, task.done)), staged or restarts)

        def add(kind: str, name: str, cfg: object, group: str | None = None, concatenate: bool = True) -> None:
            # A time limit ends episodes; a task of a sequence can end itself with a termination on its started_at.
            if isinstance(cfg, TimeLimit) and staged:
                raise ValueError(f"Task '{task_path}': the tasks of a sequence cannot have a time limit.")
            # Reset and startup events set up whole episodes, so they run in every world of their task.
            setup = isinstance(cfg, Event) and cfg.mode is not EventMode.INTERVAL
            full = f"{prefix}.{name}" if prefix else name
            entry_path = (kind, full) if group is None else (kind, group, full)
            self._add(ScheduleEntry(entry_path, _scoped(cfg, scope), concatenate, name, None if setup else stage))

        for kind in ("actions", "commands", "quantities"):
            for name, use in getattr(task, kind).items():
                if use is not None:
                    add(kind, name, use)
        for group, observations in task.observations.items():
            if "/" in group:
                raise ValueError(f"Observation group names cannot contain '/': '{group}'.")
            terms, concatenate = group_terms(observations)
            uses = [entry for entry in self._entries.values() if entry.path[:2] == ("observations", group)]
            if uses and uses[0].concatenate != concatenate:
                raise ValueError(f"The observations of group '{group}' must all be concatenated or all be separate.")
            for name, observation in terms.items():
                if observation is not None:
                    add("observations", name, observation, group, concatenate)
        for kind in ("rewards", "terminations", "events"):
            for name, use in getattr(task, kind).items():
                if use is not None:
                    add(kind, name, use)

        for name, subtask in task.subtasks.items():
            subtask_names = names if task.mode is TaskMode.SPLIT else (*names, name)
            in_sequence = staged or task.mode in (TaskMode.SEQUENCE, TaskMode.CHOICE)
            self._add_task(subtask, (*path, name), subtask_names, in_sequence, scope, restarts)

    def add_action(self, name: str, action: DirectAction | Action) -> ScheduleEntry:
        return self._add(ScheduleEntry(("actions", name), action))

    def add_command(self, name: str, command: Command) -> ScheduleEntry:
        return self._add(ScheduleEntry(("commands", name), command))

    def add_quantity(self, name: str, quantity: Quantity) -> ScheduleEntry:
        return self._add(ScheduleEntry(("quantities", name), quantity))

    def add_event(self, name: str, event: Event) -> ScheduleEntry:
        return self._add(ScheduleEntry(("events", name), event))

    def add_observation(
        self, group: str, name: str, observation: Observation, concatenate: bool = True
    ) -> ScheduleEntry:
        """Add an observation to a group, concatenated with the group's other terms or in a buffer of its own."""
        if "/" in group or "/" in name:
            raise ValueError(f"Observation group and term names cannot contain '/': '{group}', '{name}'.")
        uses = [entry for entry in self._entries.values() if entry.path[:2] == ("observations", group)]
        if uses and uses[0].concatenate != concatenate:
            raise ValueError(f"The observations of group '{group}' must all be concatenated or all be in own buffers.")
        return self._add(ScheduleEntry(("observations", group, name), observation, concatenate))

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

    def finalize(
        self,
        device: str = "cpu",
        backend: Backend | str = Backend.TORCH,
        diagnostics: bool = False,
        final_observations: bool = False,
    ) -> MDP:
        """Validate the schedule and compile it into an immutable MDP.

        Raises a single ``ValueError`` that lists every error found. Editing the builder afterwards does not change the
        returned MDP. With ``diagnostics``, :func:`evaluate` also writes the reward and termination flag of each term.
        With ``final_observations``, it also writes the observations of the worlds that ended, before their reset, to
        :attr:`Transition.final_observations`, in the kernels that compute them.
        """
        mdp = self._compile(device, Backend(backend), diagnostics, final_observations)
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

    def _compile(
        self, device: str, backend: Backend, diagnostics: bool = False, final_observations: bool = False
    ) -> MDP:
        """Compile the schedule of each world config, then merge their ops, sharing the ones they have in common."""
        configs = self._world_configs or [(self.schedule(), self._tasks, range(self.schema.num_worlds))]
        world_ranges = [worlds for _, _, worlds in configs]

        errors = []
        if self._world_configs and self._entries:
            errors.append("Add term uses to the replicated builders, not to the builder that replicates them.")
        if world_ranges[-1].stop != self.schema.num_worlds:
            errors.append(f"The world configs cover {world_ranges[-1].stop} of {self.schema.num_worlds} worlds.")
        # Progress is tracked if any task has a completion or a sequence.
        tracked = any(
            task.done is not None or task.mode in (TaskMode.SEQUENCE, TaskMode.CHOICE)
            for _, tasks, _ in configs
            for task, _ in tasks.values()
        )

        # The actions of every world config compile first: they share one layout, a range of columns per name, which
        # sizes the last actions that the other uses read.
        contexts, time_limits = [], set()
        for schedule, tasks, _ in configs:
            context, time_limit = self._compile_actions(schedule, tasks if tracked else {}, backend)
            contexts.append(context)
            time_limits.add(time_limit)
        action_dim = _allocate_action_slots(contexts)
        for index, ((schedule, tasks, worlds), context) in enumerate(zip(configs, contexts)):
            context.mdp_fields[last_action] = ArrayLayout(wp.float32, 2, {"actions": action_dim}, (action_dim,))
            self._compile_schedule(schedule, tasks if tracked else {}, context)
            prefix = f"world config {index} ({worlds.start}-{worlds.stop - 1}): " if len(configs) > 1 else ""
            errors += [prefix + error for error in context.errors]

        if len(time_limits) > 1:
            errors.append(f"The world configs must share one time limit, found {sorted(map(str, time_limits))} s.")
        field_shapes = {}
        for context in contexts:
            for op in context.ops[Phase.COMMANDS] + context.ops[Phase.QUANTITIES]:
                if field_shapes.setdefault(op.writes[0], op.shape) != op.shape:
                    errors.append(f"The world configs give '{op.writes[0]}' different shapes.")
            # The ops of one use share its state, each in its own worlds.
            for phase, ops in context.ops.items():
                for op in ops:
                    for name, spec in op.state.items() if isinstance(op, TermOp) else ():
                        selector = term_state(op.use, name)
                        if field_shapes.setdefault(selector, spec.shape) != spec.shape:
                            errors.append(f"The world configs give '{selector}' different shapes.")
        if errors:
            raise ValueError("Invalid MDP:\n  " + "\n  ".join(errors))

        ops = _merge_ops([(context.ops, worlds) for context, worlds in zip(contexts, world_ranges)], self.schema)
        action_slots = _action_slots(ops[Phase.ACTIONS])
        trees = tuple(
            TaskTree(worlds, context.tasks, frozenset(_completes(context.tasks)))
            for context, worlds in zip(contexts, world_ranges)
            if tracked
        )
        modes = {node.path: node.mode for tree in trees for node in tree.nodes}
        ops[Phase.OBSERVATIONS], slots, observation_dims, layouts = _allocate_slots(
            ops[Phase.OBSERVATIONS], modes, errors
        )

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
        # A choice draws its subtask from a stream of its own.
        trees = tuple(
            replace(
                tree,
                nodes=tuple(
                    replace(node, stream=streams.setdefault(f"choice/{node.path}", len(streams)))
                    if node.mode is TaskMode.CHOICE
                    else node
                    for node in tree.nodes
                ),
            )
            for tree in trees
        )
        ops = MappingProxyType(ops)
        tuned = _tuned_values(ops, errors)

        observation_layouts = {
            buffer: layouts.get(buffer, ((dim,), wp.float32)) for buffer, dim in observation_dims.items()
        }
        mixed = {buffer.split("/")[0] for buffer in observation_layouts if "/" in buffer} & set(observation_dims)
        if mixed:
            errors.append(f"The world configs disagree on whether the groups {sorted(mixed)} are concatenated.")
        if errors:
            raise ValueError("Invalid MDP:\n  " + "\n  ".join(errors))
        subsets = {selector: ids for context in contexts for selector, ids in context.subsets.items()}
        time_limit = time_limits.pop()
        return MDP(
            schema=self.schema,
            device=device,
            backend=backend,
            ops=ops,
            plan=build_plan(ops, self.schema, trees, slots, tuned),
            action_dim=action_dim,
            observation_dims=MappingProxyType(observation_dims),
            observation_layouts=MappingProxyType(observation_layouts),
            max_episode_length=None if time_limit is None else math.ceil(time_limit / self.schema.step_dt),
            streams=tuple(streams),
            subsets=MappingProxyType(subsets),
            diagnostics=diagnostics,
            final_observations=final_observations,
            world_configs=tuple(
                WorldConfig(
                    worlds,
                    max((op.columns.stop for op in context.ops[Phase.ACTIONS]), default=0),
                    MappingProxyType(_used_columns(ops[Phase.OBSERVATIONS], worlds)),
                )
                for context, worlds in zip(contexts, world_ranges)
            ),
            action_slots=action_slots,
            tasks=trees,
            slots=slots,
            tuned=tuned,
        )

    def _compile_actions(
        self, schedule: tuple[ScheduleEntry, ...], tasks: dict[str, tuple[TaskCfg, bool]], backend: Backend
    ) -> tuple[CompileContext, float | None]:
        """Start compiling one world config with its actions, whose columns are given once those of every world
        config are known, see :func:`_allocate_action_slots`. Returns them with its episode length in seconds."""
        context = CompileContext(backend, _dependency_errors(schedule))
        context.modes = {path: task.mode for path, (task, _) in tasks.items()}
        # Terms may read the progress of every task.
        for path, (task, _) in tasks.items():
            for name in ("done", "holding", "active"):
                context.mdp_fields[progress_field(path, name)] = ArrayLayout(wp.bool, 1, {})
            staged = ["stage"] if task.mode in (TaskMode.SEQUENCE, TaskMode.CHOICE) else []
            sequence, repeat = staged, ["count"] if task.repeat else []
            for name in ("done_at", "started_at", *sequence, *repeat):
                context.mdp_fields[progress_field(path, name)] = ArrayLayout(wp.int32, 1, {})
            context.mdp_fields[progress_field(path, "score")] = ArrayLayout(wp.float32, 1, {})
        # The time limit compiles to no op, so it is known even when the other uses cannot be ordered.
        time_limits = [entry for entry in schedule if isinstance(entry.cfg, TimeLimit)]
        if len(time_limits) > 1:
            context.errors.append(f"Only one time limit is allowed, found {', '.join(map(str, time_limits))}.")
        time_limit = time_limits[0].cfg.episode_length_s if time_limits else None

        for entry in schedule:
            if isinstance(entry.cfg, DirectAction):
                self._compile_action(entry, context)
            elif isinstance(entry.cfg, Action):
                self._compile_term(entry, context)
        return context, time_limit

    def _compile_schedule(
        self, schedule: tuple[ScheduleEntry, ...], tasks: dict[str, tuple[TaskCfg, bool]], context: CompileContext
    ) -> None:
        """Compile the uses of one world config other than its actions, and its task tree if it has one."""
        # Every use compiles after the uses that provide the commands, quantities, and term state it reads.
        entries = {entry.path: entry for entry in schedule}
        providers = {selector: entry.path for entry in schedule for selector in entry.provides}
        # A use's when is read at the start of a step, which no use of its phase needs to run before.
        reads = {
            path: {providers[selector] for selector in (*entry.reads, entry.when) if selector in providers} - {path}
            for path, entry in entries.items()
        }
        try:
            order = list(graphlib.TopologicalSorter(reads).static_order())
        except graphlib.CycleError as error:
            # Without an order, every use after the cycle would also be reported for the fields it cannot find.
            context.errors.append(
                f"Uses read each other's fields in a cycle: {', '.join(map('.'.join, error.args[1]))}. A use is"
                " compiled after the uses whose commands, quantities, and state it reads."
            )
            return

        for entry in (entries[path] for path in order):
            if entry.phase is not Phase.ACTIONS and not isinstance(entry.cfg, TimeLimit):
                self._compile_term(entry, context)

        # An offset may be a field of the MDP, such as a quantity, which exists once its use is compiled.
        for op in context.ops[Phase.ACTIONS]:
            if not isinstance(op, ActionOp) or not isinstance(op.offset, FieldSelector):
                continue
            offset = self._resolve(op.name, "offset", op.offset, context)
            if offset is not None and offset.shape != self._resolve(op.name, "target", op.target, context).shape:
                context.errors.append(f"{op.name}: the offset '{op.offset}' must have the shape of '{op.target}'.")

        # Restore the schedule order within each phase, with the positions it gives. Observation columns are given
        # after the world configs merge, by name, see _allocate_slots.
        position = {str(entry): k for k, entry in enumerate(schedule)}
        for phase, ops in context.ops.items():
            if phase is Phase.ACTIONS:
                continue
            ops.sort(key=lambda op: position[op.name])
            for k, op in enumerate(ops):
                ops[k] = replace(op, index=k)

        context.tasks = tuple(self._compile_task(path, tasks, context) for path in tasks)

    def _compile_task(self, path: str, tasks: dict[str, tuple[TaskCfg, bool]], context: CompileContext) -> TaskNode:
        """Compile a task of the tree, or record why its progress cannot be tracked."""
        task, staged = tasks[path]
        name, errors = f"Task '{path}'" if path else "The task", context.errors
        children = tuple(f"{path}.{child}" if path else child for child in task.subtasks)
        done_dtype = None
        if task.done is not None:
            layout = self._resolve(name, "done", task.done, context)
            if layout is not None and (layout.ndim != 1 or layout.dtype not in (wp.bool, wp.float32, wp.int32)):
                errors.append(f"{name}: done '{task.done}' must be a field of one bool, float, or int per world.")
            done_dtype = None if layout is None else layout.dtype
        if task.hold < 1:
            errors.append(f"{name}: hold must be at least 1, got {task.hold}.")
        completes = [child for child, subtask in task.subtasks.items() if _has_completion(subtask)]
        if task.mode is TaskMode.SEQUENCE and len(completes) != len(children):
            missing = sorted(set(task.subtasks) - set(completes))
            errors.append(f"{name}: every task of a sequence needs a completion, {missing} have none.")
        weights = ()
        if task.mode is TaskMode.CHOICE:
            chances = task.weights or dict.fromkeys(task.subtasks, 1.0)
            if set(chances) != set(task.subtasks) or not all(chance >= 0 for chance in chances.values()):
                errors.append(f"{name}: a choice needs a weight of at least 0 for each of {list(task.subtasks)}.")
            elif sum(chances.values()) > 0:
                weights = tuple(chances[child] / sum(chances.values()) for child in task.subtasks)
        if task.require is not None and not 1 <= task.require <= len(completes):
            errors.append(f"{name}: require must be between 1 and {len(completes)}, the subtasks with a completion.")
        return TaskNode(
            path=path,
            mode=task.mode,
            children=children,
            done=task.done,
            done_dtype=done_dtype,
            hold=task.hold,
            score=task.score,
            final=task.final,
            require=task.require,
            staged=staged,
            repeat=task.repeat,
            weights=weights,
        )

    def _compile_action(self, entry: ScheduleEntry, context: CompileContext) -> None:
        """Compile an action use into an op, or record why it cannot be compiled."""
        cfg = entry.cfg
        target = self._resolve(entry, "target", cfg.target, context)
        if target is None:
            return
        self._check_writable(entry, cfg.target, context.errors)
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
        # Its columns of the action are given with those of every world config, by name.
        columns, slot = slice(0, len(elements)), entry.name or entry.path[-1]
        action = ActionOp(str(entry), cfg.target, columns, cfg.scale, cfg.offset, subset=cfg.subset, slot=slot)
        names = self._element_names({"target": cfg.target, **({"subset": cfg.subset} if cfg.subset else {})}, context)
        labels = names.get("subset" if cfg.subset else next(iter(target.sizes), ""))
        labels = labels if labels is not None and len(labels) == len(elements) else None
        action = replace(action, stage=entry.stage, rate=cfg.rate, labels=labels)
        # Action terms may write other elements of the same field, such as other joints, after or before it, and
        # actions of tasks of a sequence that never run together the same ones.
        for op in context.ops[Phase.ACTIONS]:
            if isinstance(op, TermOp) or _exclusive(op, action, context.modes):
                continue
            written = set(context.subsets[op.subset]) if op.subset is not None else elements
            if op.target == cfg.target and (op.subset is None or elements & written):
                context.errors.append(f"{entry}: '{cfg.target}' is already written by '{op.name}'.")
        context.ops[Phase.ACTIONS].append(action)

    def _compile_term(self, entry: ScheduleEntry, context: CompileContext) -> None:
        """Compile a term use into an op, or record why it cannot be compiled."""
        cfg, errors = entry.cfg, context.errors
        term = cfg.term
        num_errors = len(errors)

        # Check the use, and resolve the shapes of its output and state.
        implementation = _implementation(entry, context.backend, errors)
        outputs = cfg.outputs if isinstance(cfg, (Event, Action)) else {}
        # A command writes its own field, which every hook gets as ``command``.
        command = {"command": entry.writes[0]} if isinstance(cfg, Command) else {}
        # An action gets its columns of the action as ``action``.
        action = ["action"] if isinstance(cfg, Action) else []
        names = [*cfg.inputs, *outputs, *command, *action, *term.state, *cfg.params]
        names += ["rng"] if term.random else []
        if len(set(names)) != len(names):
            errors.append(f"{entry}: inputs, outputs, state, params, and rng must have distinct names, got {names}.")
        tunables, params, weight = _tunables(entry, errors)

        # Scene entities bound as inputs or outputs reach Python hooks as objects, such as articulations.
        bound = {**cfg.inputs, **outputs}
        entities = {arg: selector for arg, selector in bound.items() if isinstance(selector, Entity)}
        for arg, entity in entities.items():
            if not entity.scene or entity.name not in self.schema.entities:
                errors.append(f"{entry}: {arg!r} binds {entity.name!r}, which is not an entity of the scene.")
        hooks = vars(implementation).values() if implementation is not None else ()
        if entities and any(isinstance(hook, wp.Function) for hook in hooks):
            errors.append(
                f"{entry}: wp.func hooks cannot take scene entities ({sorted(entities)}); use a Python hook instead."
            )
        fields = {
            arg: self._resolve(entry, arg, selector, context) for arg, selector in bound.items() if arg not in entities
        }
        for name, default in tunables.items():
            fields[name] = context.mdp_fields[tunable(str(entry), name)] = _tunable_layout(entry, name, default, errors)
        for arg, selector in outputs.items():
            if arg not in entities and fields[arg] is not None:
                self._check_writable(entry, selector, errors)
        # Int params, such as a number of points, can also size dimensions.
        sizes = {name: value for name, value in params.items() if type(value) is int}
        sizes |= _dimension_sizes(fields)
        output_shape = _resolve_shape(entry, "output", term.shape, sizes, errors)
        if output_shape and isinstance(cfg, (Reward, Termination)):
            errors.append(f"{entry}: {type(cfg).__name__} outputs must have shape ('worlds',), got {output_shape}.")
        if isinstance(cfg, Action):
            _check_action(entry, output_shape, errors)
        dtype = wp.bool if isinstance(cfg, Termination) else term.dtype or wp.float32
        _check_output_type(entry, dtype, errors)
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
        if isinstance(cfg, Quantity) and (cfg.log or cfg.record) and output_shape:
            errors.append(f"{entry}: a logged or recorded quantity must have shape ('worlds',), got {output_shape}.")
        if isinstance(cfg, Event) and cfg.commit is True and cfg.mode is not EventMode.RESET:
            errors.append(f"{entry}: only reset events can ask for a commit.")
        if isinstance(cfg, Command) and implementation is not None and implementation.reset is not None:
            errors.append(f"{entry}: commands write their state when they resample, so they have no reset hook.")
        if isinstance(cfg, Observation) and cfg.noise is not None and cfg.noise.scale is not None:
            layout = self._resolve(entry, "noise scale", cfg.noise.scale, context)
            if layout is not None and (layout.ndim != 1 or layout.dtype is not wp.float32):
                errors.append(f"{entry}: the noise scale '{cfg.noise.scale}' must be a float field of one per world.")

        when, when_dtype = self._compile_when(entry, context)
        if len(errors) > num_errors:
            return

        # Check the hooks' signatures. Params and rng are not arrays.
        arguments = {
            **dict.fromkeys(names),
            **fields,
            **{name: ArrayLayout(wp.float32, len(shape), {}) for name, shape in state_shapes.items()},
            **{name: ArrayLayout(wp.float32, 1 + len(output_shape), {}) for name in command},
            **{name: ArrayLayout(wp.float32, 2, {}) for name in action},
        }
        if term.init is not None:
            try:
                inspect.signature(term.init).bind(None, **dict.fromkeys(cfg.params))  # Tunables at their default
            except TypeError as error:
                errors.append(f"{entry}: init {term.init.__name__!r} is called with (ctx, **params): {error}.")
        # Python hooks may also take ``cache``, a dict the use keeps for host objects.
        if not isinstance(implementation.compute, wp.Function):
            arguments["cache"] = None
        # Events and commands take the mask of the worlds to write; other Warp functions take the output to write.
        if isinstance(cfg, (Command, Event)):
            _check_hook(entry, implementation.compute, arguments, errors, extra="mask")
        elif isinstance(cfg, Action):
            # A Python hook may take the worlds where the action is active, as ``mask``.
            _check_hook(entry, implementation.compute, {**arguments, "mask": None}, errors)
        else:
            extra = "out" if implementation is term.warp else None
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

        observation, columns = _allocate_columns(entry, output_shape, context)
        labels = None
        if len(term.shape) == 2:
            labels = self._element_names(bound, context).get(term.shape[1])
            labels = labels if labels is not None and len(labels) == output_shape[0] else None
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
                inputs=MappingProxyType(
                    {**cfg.inputs, **outputs, **command, **{name: tunable(str(entry), name) for name in tunables}}
                ),
                params=MappingProxyType(
                    {name: self.schema.step_dt if value is step_dt else value for name, value in params.items()}
                ),
                state=MappingProxyType(
                    {
                        name: replace(spec, shape=state_shapes[name], persistent=spec.persistent or name in shared)
                        for name, spec in term.state.items()
                    }
                ),
                shape=output_shape,
                dtype=dtype,
                # An entity output writes any of the entity's writable fields.
                writes=tuple(dict.fromkeys(selector for write in entry.writes for selector in self._fields_of(write))),
                weights=(weight,) if isinstance(cfg, Reward) else (),
                tunables=MappingProxyType(tunables),
                tunable_weight=isinstance(cfg, Reward) and isinstance(cfg.weight, Tunable),
                time_out=cfg.time_out if isinstance(cfg, Termination) else False,
                observation=observation,
                timer=timer,
                random=term.random,
                capturable=term.capturable,
                torch_hooks=context.backend is Backend.WARP and implementation is term.torch,
                init=term.init,
                commit=cfg.commit if isinstance(cfg, Event) else False,
                log=isinstance(cfg, Quantity) and cfg.log,
                record=isinstance(cfg, Quantity) and cfg.record,
                columns=columns,
                slot=(entry.name or entry.path[-1]) if isinstance(cfg, Action) else "",
                rate=cfg.rate if isinstance(cfg, Action) else ActionRate.PHYSICS,
                labels=labels,
                stage=entry.stage,
                when=when,
                when_dtype=when_dtype,
                when_default=cfg.when.default if isinstance(cfg.when, Tunable) else None,
            )
        )

    def _compile_when(self, entry: ScheduleEntry, context: CompileContext) -> tuple[FieldSelector | None, type | None]:
        """Resolve the field that switches a use on and off, and its element type, or record why it cannot."""
        cfg, errors = entry.cfg, context.errors
        if cfg.when is None:
            return None, None
        if isinstance(cfg, (Quantity, Action)) or isinstance(cfg, Event) and cfg.mode is not EventMode.INTERVAL:
            errors.append(
                f"{entry}: only rewards, terminations, observations, commands, and interval events can have a when."
            )
            return None, None
        if isinstance(cfg.when, Tunable):
            if type(cfg.when.default) is not bool:
                errors.append(f"{entry}: a tunable when switches the use on or off, so its default must be a bool.")
            context.mdp_fields[entry.when] = ArrayLayout(wp.bool, 1, {})
            return entry.when, wp.bool
        if cfg.when in entry.writes or cfg.when in entry.provides:
            errors.append(f"{entry}: when '{cfg.when}' is written by the use itself, which could not turn it on again.")
            return None, None
        layout = self._resolve(entry, "when", cfg.when, context)
        if layout is None:
            return None, None
        if layout.ndim != 1 or layout.dtype not in (wp.bool, wp.float32, wp.int32):
            errors.append(f"{entry}: when '{cfg.when}' must be a field of one bool, float, or int per world.")
        return cfg.when, layout.dtype

    def _element_names(
        self, bound: Mapping[str, FieldSelector | SubsetSelector | Entity], context: CompileContext
    ) -> dict[str, tuple[str, ...]]:
        """Return the names of the elements of the dimensions that a use's inputs name, by dimension: those of each
        entity dimension of its scene fields, and those of each subset, also under its argument name. A dimension
        that inputs name differently, such as the bodies of two entities, is left out."""
        names: dict[str, tuple[str, ...]] = {}
        ambiguous = set()

        def add(dim: str, elements: tuple[str, ...]) -> None:
            if names.setdefault(dim, elements) != elements:
                ambiguous.add(dim)

        for arg, selector in bound.items():
            if isinstance(selector, SubsetSelector) and selector in context.subsets:
                entity = self.schema.entities[selector.entity]
                add(arg, tuple(entity.names[selector.dim][index] for index in context.subsets[selector]))
            elif isinstance(selector, FieldSelector) and selector.scene and selector.entity in self.schema.entities:
                entity = self.schema.entities[selector.entity]
                schema = entity.fields.get(selector.field)
                dims = () if schema is None else schema.dims
                for dim in dims:
                    add(dim, tuple(entity.names[dim]))
                if len(dims) == 1:
                    add(arg, tuple(entity.names[dims[0]]))
        return {dim: elements for dim, elements in names.items() if dim not in ambiguous}

    def _fields_of(self, selector: FieldSelector | Entity) -> tuple[FieldSelector, ...]:
        """Return the fields a write covers: a field itself, or the writable fields of a scene entity."""
        if not isinstance(selector, Entity):
            return (selector,)
        fields = self.schema.entities[selector.name].fields
        return tuple(FieldSelector(selector.name, name) for name, schema in fields.items() if schema.writable)

    def _check_writable(self, entry: ScheduleEntry, selector: FieldSelector, errors: list[str]) -> None:
        """Record an error if a use writes a scene field that writes would not reach, such as a derived one."""
        if selector.scene and not self.schema.entities[selector.entity].fields[selector.field].writable:
            errors.append(
                f"{entry}: '{selector}' is read-only: the simulation computes it, so writes would not reach it."
            )

    def _resolve(
        self, entry: ScheduleEntry | str, arg: str, selector: FieldSelector | SubsetSelector, context: CompileContext
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
    """Return the implementation of a term use for a backend, or record why it has none.

    A Warp MDP runs the Torch hooks of a term without Warp ones, between its generated kernels.
    """
    term = entry.cfg.term
    implementation = term.torch if backend is Backend.TORCH else term.warp or term.torch
    if implementation is None:
        errors.append(f"{entry}: the term has no '{backend.value}' implementation.")
    elif not term.capturable and any(isinstance(hook, wp.Function) for hook in vars(implementation).values()):
        errors.append(f"{entry}: wp.func hooks always run inside generated kernels and cannot be non-capturable.")
    return implementation


def _allocate_columns(
    entry: ScheduleEntry, output_shape: tuple[int, ...], context: CompileContext
) -> tuple[ObservationOutput | None, slice | None]:
    """Give an observation its output, with as many columns as it writes, and an action term its columns of the
    action, after those of the uses compiled before it. Returns the observation's output and the action term's
    columns."""
    cfg = entry.cfg
    if isinstance(cfg, Observation):
        # Columns are given once the world configs merge, by name, see _allocate_slots.
        columns = slice(0, math.prod(output_shape) * max(cfg.history_length, 1))
        output = ObservationOutput(entry.buffer, columns, cfg.noise, cfg.clip, cfg.scale, cfg.history_length)
        return replace(output, slot=entry.name or entry.path[-1]), None
    if isinstance(cfg, Action):
        # Its columns of the action are given with those of every world config, by name.
        return None, slice(0, output_shape[0])
    return None, None


def _check_action(entry: ScheduleEntry, output_shape: tuple[int, ...] | None, errors: list[str]) -> None:
    """Record why an action term cannot be compiled: its shape or its hooks.

    Actions write their fields in schedule order, so an action term may write elements of a field that other actions
    also write, such as other joints of a joint target.
    """
    term = entry.cfg.term
    if term.random or term.init is not None:
        errors.append(f"{entry}: action terms cannot be random or have an init hook.")
    if output_shape is not None and len(output_shape) != 1:
        errors.append(f"{entry}: an action's shape is ('worlds', columns), got {term.shape}.")


def _check_output_type(entry: ScheduleEntry, dtype: type, errors: list[str]) -> None:
    """Record why a term's output type does not fit its use: only observations of groups that are not concatenated may
    set one."""
    cfg = entry.cfg
    if cfg.term.dtype is not None and not isinstance(cfg, Observation):
        errors.append(f"{entry}: only observations can set an output type.")
    if isinstance(cfg, Observation) and dtype is not wp.float32:
        processed = cfg.noise is not None or cfg.clip is not None or cfg.scale is not None or cfg.history_length
        if processed or entry.concatenate:
            errors.append(
                f"{entry}: an observation of type {dtype.__name__} must be in a group with concatenate=False, without"
                " noise, clip, scale, or history."
            )


def _tunables(
    entry: ScheduleEntry, errors: list[str]
) -> tuple[dict[str, typing.Any], dict[str, typing.Any], float | None]:
    """Return the defaults of a use's tunable params, which are fields of the MDP of one value per world that its
    hooks read like inputs, its other params, and the default of a reward's weight."""
    cfg = entry.cfg
    tunables = {name: value.default for name, value in cfg.params.items() if isinstance(value, Tunable)}
    params = {name: value for name, value in cfg.params.items() if name not in tunables}
    weight = cfg.weight if isinstance(cfg, Reward) else None
    if isinstance(weight, Tunable):
        weight = weight.default
        if "weight" in tunables:
            errors.append(f"{entry}: a reward with a tunable weight cannot have a tunable param named 'weight'.")
    return tunables, params, weight


def _tunable_layout(entry: ScheduleEntry | str, name: str, default: object, errors: list[str]) -> ArrayLayout:
    """Return the layout of the field of a tunable value: one bool, int, float, or float vector per world."""
    if isinstance(default, tuple) and default and all(type(value) in (int, float) for value in default):
        return ArrayLayout(wp.types.vector(len(default), wp.float32), 1, {})
    dtypes = {bool: wp.bool, int: wp.int32, float: wp.float32}
    if type(default) not in dtypes:
        errors.append(
            f"{entry}: the tunable {name!r} must be a bool, int, float, or tuple of numbers, got {default!r}."
        )
    return ArrayLayout(dtypes.get(type(default), wp.float32), 1, {})


def _tuned_values(ops: Mapping[Phase, tuple], errors: list[str]) -> tuple[TunedValue, ...]:
    """Return the tunable values of the ops, once per use, with their reward weights."""
    tuned: dict[str, TunedValue] = {}
    for op in (op for phase_ops in ops.values() for op in phase_ops if isinstance(op, TermOp)):
        values = [
            TunedValue(f"{op.use}.{name}", op.inputs[name], dtype=_tunable_layout(op.name, name, default, []).dtype)
            for name, default in op.tunables.items()
        ]
        values += [TunedValue(f"{op.use}.weight", "reward_weights", op.index)] if op.tunable_weight else []
        values += [TunedValue(f"{op.use}.when", op.when, dtype=wp.bool)] if op.when_default is not None else []
        for value in values:
            if tuned.setdefault(value.path, value) != value:
                errors.append(f"The world configs tune '{value.path}' differently: tune it in all or none of them.")
    return tuple(tuned.values())


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
            if bound is None:
                continue
            if not wp.types.is_array(annotation):
                # A field of one value per world, such as a tunable param, binds to a scalar: its world's value.
                if bound.ndim != 1 or not wp.types.types_equal(wp.array(dtype=annotation).dtype, bound.dtype):
                    errors.append(
                        f"{entry}: wp.func {hook.key!r} takes {name!r} as one value, but it is a"
                        f" {bound.ndim}-dimensional array of {bound.dtype.__name__}, not one value of that type per"
                        " world."
                    )
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
    """Report each use that reads a field before a later use of the same phase writes it.

    A use also writes its own state, such as the progress of a curriculum event that another event reads.
    """
    errors = []
    for index, reader in enumerate(schedule):
        for writer in itertools.takewhile(lambda entry: entry.phase is reader.phase, schedule[index + 1 :]):
            state = (
                [term_state(str(writer), name) for name in writer.cfg.term.state]
                if isinstance(writer.cfg, TermCfg)
                else []
            )
            for selector in dict.fromkeys(reader.reads):
                written = selector in writer.writes or Entity(selector.entity) in writer.writes
                if written or selector in state:
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


def _expand(task: TaskCfg, num_worlds: int) -> list[tuple[TaskCfg, int]]:
    """Return the tasks without splits that a task runs, in order, each with its number of worlds.

    A split becomes, in each of its shares of the worlds, a split with only that subtask. Subtasks that split divide
    the same worlds: a world runs, of each, the subtask of its share.
    """
    if task.mode is TaskMode.SPLIT:
        names = list(task.subtasks)
        if not names:
            raise ValueError("A split needs at least one subtask.")
        shares = task.worlds
        if shares is None:
            shares = {name: num_worlds // len(names) + (k < num_worlds % len(names)) for k, name in enumerate(names)}
        if set(shares) != set(names) or sum(shares.values()) != num_worlds:
            raise ValueError(f"A split must give its subtasks {names} all of its {num_worlds} worlds, got {shares}.")
        return [
            (replace(task, subtasks={name: subtask}, worlds=None), count)
            for name in names
            for subtask, count in _expand(task.subtasks[name], shares[name])
        ]

    variants = {name: _expand(subtask, num_worlds) for name, subtask in task.subtasks.items()}
    # The worlds where every subtask runs one variant: between the boundaries of the shares of all of them.
    ends = sorted({end for expanded in variants.values() for end in itertools.accumulate(n for _, n in expanded)})
    expanded = []
    for start, stop in itertools.pairwise([0, *ends]):
        subtasks = {name: _variant_at(variants[name], start) for name in task.subtasks}
        expanded.append((replace(task, subtasks=subtasks), stop - start))
    return expanded or [(task, num_worlds)]


def _variant_at(variants: list[tuple[TaskCfg, int]], world: int) -> TaskCfg:
    """Return the variant of a task, of those that run in consecutive shares of the worlds, that runs in ``world``."""
    stops = list(itertools.accumulate(count for _, count in variants))
    return variants[next(k for k, stop in enumerate(stops) if world < stop)][0]


def _local_names(task: TaskCfg, path: str, prefix: str) -> dict[FieldSelector, FieldSelector]:
    """Return the fields that a task defines, under the names that a task that runs alone gives them: its commands and
    quantities, the state and tunable values of its uses, which get the prefix of its uses' names, and the progress of
    it and of its subtasks, by their paths from it."""

    def full(name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    names = {
        FieldSelector(entity.name, name, scene=False): FieldSelector(entity.name, full(name), scene=False)
        for entity, uses in ((commands, task.commands), (quantities, task.quantities))
        for name in uses
    }
    groups = [(f"observations.{group}", group_terms(uses)[0]) for group, uses in task.observations.items()]
    kinds = [(kind, getattr(task, kind)) for kind in ("actions", "commands", "quantities", "rewards", "events")]
    for kind, uses in (*kinds, ("terminations", task.terminations), *groups):
        for name, cfg in uses.items():
            if not isinstance(cfg, TermCfg) or not prefix:
                continue
            local, use = f"{kind}.{name}", f"{kind}.{full(name)}"
            names |= {term_state(local, state): term_state(use, state) for state in cfg.term.state}
            tunables = [param for param, value in cfg.params.items() if isinstance(value, Tunable)]
            tunables += ["when"] if isinstance(cfg.when, Tunable) else []
            names |= {tunable(local, param): tunable(use, param) for param in tunables}
    for relative in _task_paths(task):
        absolute = ".".join(part for part in (path, relative) if part)
        if relative != absolute:
            names |= {progress_field(relative, name): progress_field(absolute, name) for name in PROGRESS_FIELDS}
    return names


def _task_paths(task: TaskCfg) -> list[str]:
    """Return the paths from a task of it, "", and of its subtasks, their subtasks, and so on."""
    return [
        "",
        *(f"{name}.{sub}" if sub else name for name, subtask in task.subtasks.items() for sub in _task_paths(subtask)),
    ]


def _scoped(cfg: object, scope: Mapping[FieldSelector, FieldSelector]) -> object:
    """Return a use whose selectors of commands and quantities name the fields of the task that defines them."""
    if not scope:
        return cfg
    if isinstance(cfg, DirectAction):
        return replace(cfg, offset=scope.get(cfg.offset, cfg.offset)) if isinstance(cfg.offset, FieldSelector) else cfg
    if not isinstance(cfg, TermCfg):
        return cfg
    changes = {"inputs": {arg: scope.get(selector, selector) for arg, selector in cfg.inputs.items()}}
    if isinstance(cfg.when, FieldSelector):
        changes["when"] = scope.get(cfg.when, cfg.when)
    if isinstance(cfg, Observation) and cfg.noise is not None and cfg.noise.scale is not None:
        changes["noise"] = replace(cfg.noise, scale=scope.get(cfg.noise.scale, cfg.noise.scale))
    return replace(cfg, **changes)


def _has_completion(task: TaskCfg) -> bool:
    return task.done is not None or any(_has_completion(subtask) for subtask in task.subtasks.values())


def _completes(nodes: tuple[TaskNode, ...]) -> set[str]:
    """Return the paths of the tasks of a tree that have a completion."""
    completes = set()
    for node in reversed(nodes):
        if node.done is not None or any(child in completes for child in node.children):
            completes.add(node.path)
    return completes


def _allocate_action_slots(contexts: list[CompileContext]) -> int:
    """Give the action ops of every world config their columns: one slot per name, in order, as wide as its widest
    op, each op taking its first columns. Returns the number of columns.

    Ops of different world configs or of different tasks of a sequence may share a slot, since they never take it in
    the same world at the same step; other ops of the same name are errors.
    """
    widths: dict[str, int] = {}
    for context in contexts:
        named: dict[str, list] = {}
        for op in context.ops[Phase.ACTIONS]:
            named.setdefault(op.slot, []).append(op)
            widths[op.slot] = max(widths.get(op.slot, 0), op.columns.stop)
        for name, slot_ops in named.items():
            for first, second in itertools.combinations(slot_ops, 2):
                if not _exclusive(first, second, context.modes):
                    context.errors.append(
                        f"'{first.name}' and '{second.name}' share the columns of the action '{name}', so they must run"
                        " in different world configs or in different tasks of a sequence."
                    )
    starts = dict(zip(widths, itertools.accumulate(widths.values(), initial=0)))
    for context in contexts:
        actions = context.ops[Phase.ACTIONS]
        for k, op in enumerate(actions):
            start = starts[op.slot]
            actions[k] = replace(op, columns=slice(start, start + op.columns.stop))
    return sum(widths.values())


def _action_slots(ops: tuple[ActionOp | TermOp, ...]) -> tuple[ActionSlot, ...]:
    """Return the slots of the merged action ops, in column order."""
    named: dict[str, list] = {}
    for op in ops:
        named.setdefault(op.slot, []).append(op)
    slots = []
    for name, slot_ops in named.items():
        start, stop = slot_ops[0].columns.start, max(op.columns.stop for op in slot_ops)
        slots.append(ActionSlot(name, slice(start, stop), tuple(slot_ops)))
    return tuple(sorted(slots, key=lambda slot: slot.columns.start))


def _allocate_slots(
    ops: tuple[TermOp, ...], modes: Mapping[str, TaskMode], errors: list[str]
) -> tuple[tuple[TermOp, ...], tuple[ObservationSlot, ...], dict[str, int], dict[str, tuple[tuple[int, ...], type]]]:
    """Give the observation ops their columns: one slot per name in each buffer, in order, as wide as its widest op.

    Ops of different world configs or of different tasks of a sequence may share a slot, since they never write it in
    the same world at the same step. Returns the ops, the slots that some op writes only in some steps of an episode,
    whose other columns are zeroed, the columns of each buffer, and the layouts of the buffers of separate terms.
    """
    buffers: dict[str, dict[str, list[TermOp]]] = {}
    for op in ops:
        buffers.setdefault(op.observation.group, {}).setdefault(op.observation.slot, []).append(op)

    placed, slots, dims, layouts = {}, [], {}, {}
    for buffer, named in buffers.items():
        start = 0
        for name, slot_ops in named.items():
            histories = {op.observation.history_length for op in slot_ops}
            if len(histories) > 1:
                errors.append(f"The observations named '{name}' of '{buffer}' must have the same history length.")
            if "/" in buffer:
                shapes = {(op.shape, op.dtype) for op in slot_ops}
                if len(shapes) > 1:
                    errors.append(f"The observations '{buffer}' must have the same shape and type, got {shapes}.")
                layouts[buffer] = (slot_ops[0].shape, slot_ops[0].dtype)
            for first, second in itertools.combinations(slot_ops, 2):
                if not _exclusive(first, second, modes):
                    errors.append(
                        f"'{first.name}' and '{second.name}' share the columns of '{name}' in '{buffer}', so they must"
                        " run in different world configs or in different tasks of a sequence."
                    )
            size = max(math.prod(op.shape) for op in slot_ops)
            columns = slice(start, start + size * max(slot_ops[0].observation.history_length, 1))
            start = columns.stop
            for op in slot_ops:
                placed[op.name] = replace(op, observation=replace(op.observation, columns=columns, size=size))
            if any(op.entry_key is not None for op in slot_ops):
                slots.append(ObservationSlot(buffer, columns, size, tuple(placed[op.name] for op in slot_ops)))
        dims[buffer] = start
    return tuple(placed[op.name] for op in ops), tuple(slots), dims, layouts


def _exclusive(first: TermOp, second: TermOp, modes: Mapping[str, TaskMode]) -> bool:
    """Return whether two ops never run in the same world at the same step."""
    if first.worlds is not None and second.worlds is not None and not set(first.worlds) & set(second.worlds):
        return True
    if first.stage is None or second.stage is None:
        return False
    # Tasks of a sequence run one after another: the ops are exclusive below different subtasks of a sequence.
    a, b = first.stage.split("."), second.stage.split(".")
    common = list(itertools.takewhile(lambda pair: pair[0] == pair[1], zip(a, b)))
    if len(common) in (len(a), len(b)):
        return False
    return modes.get(".".join(part for part, _ in common)) in (TaskMode.SEQUENCE, TaskMode.CHOICE)


def _used_columns(ops: tuple[TermOp, ...], worlds: range) -> dict[str, int]:
    """Return, per observation buffer, the columns up to the last that the ops active in some of ``worlds`` use."""
    used: dict[str, int] = {}
    for op in ops:
        if op.worlds is None or set(op.worlds) & set(worlds):
            buffer = op.observation.group
            used[buffer] = max(used.get(buffer, 0), op.observation.columns.stop)
    return used
