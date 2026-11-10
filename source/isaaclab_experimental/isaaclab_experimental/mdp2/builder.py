# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builder that collects schedule entries and compiles them into an :class:`MDP`, like Newton's ``ModelBuilder``."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import (
    Command,
    DirectAction,
    Event,
    EventMode,
    FieldSelector,
    Observation,
    Phase,
    Reward,
    TaskCfg,
    Termination,
    TimeLimit,
)
from .mdp import MDP, ActionOp, Backend, TermOp
from .scene import SceneSchema


@dataclass(frozen=True)
class ScheduleEntry:
    """A term use placed in the schedule. Returned by the ``add_*`` methods as a handle to it."""

    path: tuple[str, ...]  # Location in the task, e.g. ("observations", "policy", "joint_pos")
    cfg: DirectAction | Command | Observation | Reward | Termination | TimeLimit | Event

    def __str__(self) -> str:
        return ".".join(self.path)


class MDPBuilder:
    """Collects schedule entries and compiles them with :meth:`finalize`; columns follow insertion order."""

    def __init__(self, schema: SceneSchema):
        self.schema = schema
        self._entries: dict[tuple[str, ...], ScheduleEntry] = {}

    def add_task(self, task: TaskCfg) -> None:
        """Add every term use of a task, in declaration order. Uses set to ``None`` are skipped."""
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
        """Entries in execution order: by phase, then in the order they were added."""
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.cfg.phase))

    def finalize(self, device: str = "cpu", backend: Backend | str = Backend.TORCH) -> MDP:
        """Validate and compile. Raises one ``ValueError`` listing every error."""
        backend = Backend(backend)
        errors: list[str] = []
        ops = {phase: [] for phase in Phase}
        observation_dims: dict[str, int] = {}
        command_dims: dict[str, dict[str, int]] = {}
        streams: list[str] = []
        action_dim = 0
        time_limits = []
        # Commands compile first, so that terms of earlier phases can read their values.
        schedule = sorted(self.schedule(), key=lambda entry: not isinstance(entry.cfg, Command))
        for entry in schedule:
            cfg = entry.cfg
            if isinstance(cfg, TimeLimit):
                time_limits.append(entry)
            elif isinstance(cfg, DirectAction):
                dims = self._field_dims(entry, "target", cfg.target, errors)
                if dims is not None:
                    columns = slice(action_dim, action_dim + math.prod(dims.values()))
                    action_dim = columns.stop
                    ops[Phase.ACTIONS].append(ActionOp(str(entry), cfg.target, columns, cfg.scale, cfg.offset))
            else:
                op = self._compile_term(entry, backend, errors, observation_dims, command_dims, streams)
                ops[cfg.phase].append(op)
                if op is not None and isinstance(cfg, Command):
                    dims = zip(cfg.term.shape[1:], op.shape)
                    command_dims[entry.path[1]] = {dim: size for dim, size in dims if isinstance(dim, str)}
        if len(time_limits) > 1:
            errors.append(f"Only one time limit is allowed, found {', '.join(map(str, time_limits))}.")
        if errors:
            raise ValueError("Invalid MDP:\n  " + "\n  ".join(errors))
        return MDP(
            schema=self.schema,
            device=device,
            backend=backend,
            action_ops=tuple(ops[Phase.ACTIONS]),
            termination_ops=tuple(ops[Phase.TERMINATIONS]),
            reward_ops=tuple(ops[Phase.REWARDS]),
            command_ops=tuple(ops[Phase.COMMANDS]),
            interval_event_ops=tuple(op for op in ops[Phase.EVENTS] if op.timer is not None),
            reset_event_ops=tuple(op for op in ops[Phase.EVENTS] if op.timer is None),
            observation_ops=tuple(ops[Phase.OBSERVATIONS]),
            action_dim=action_dim,
            observation_dims=observation_dims,
            max_episode_length=math.ceil(time_limits[0].cfg.seconds / self.schema.step_dt) if time_limits else None,
            streams=tuple(streams),
        )

    def _add(self, entry: ScheduleEntry) -> ScheduleEntry:
        if entry.path in self._entries:
            raise ValueError(f"The schedule already has an entry '{entry}'.")
        self._entries[entry.path] = entry
        return entry

    def _compile_term(
        self,
        entry: ScheduleEntry,
        backend: Backend,
        errors: list[str],
        observation_dims: dict[str, int],
        command_dims: dict[str, dict[str, int]],
        streams: list[str],
    ) -> TermOp | None:
        """Compile a term use. Appends to ``errors`` instead of raising."""
        cfg = entry.cfg
        num_errors = len(errors)
        implementation = cfg.term.torch if backend is Backend.TORCH else cfg.term.warp
        if implementation is None:
            errors.append(f"{entry}: the term has no '{backend.value}' implementation.")
        sizes: dict[str, int] = {}
        for arg, selector in cfg.inputs.items():
            if selector.entity == "commands" and selector.field in command_dims:
                dims = command_dims[selector.field]
            else:
                dims = self._field_dims(entry, f"input {arg!r}", selector, errors)
            for dim, size in (dims or {}).items():
                if sizes.setdefault(dim, size) != size:
                    errors.append(f"{entry}: inputs disagree on the size of {dim!r} ({sizes[dim]} and {size}).")
        output_shape = self._resolve_shape(entry, "output", cfg.term.shape, sizes, errors)
        if output_shape and isinstance(cfg, (Reward, Termination)):
            errors.append(f"{entry}: {type(cfg).__name__} outputs must have shape ('worlds',), got {output_shape}.")
        state_shapes = {
            name: self._resolve_shape(entry, f"state {name!r}", spec.shape, sizes, errors)
            for name, spec in cfg.term.state.items()
        }
        names = [*cfg.inputs, *cfg.term.state, *cfg.params, *(["rng"] if cfg.term.random else [])]
        if len(set(names)) != len(names):
            errors.append(f"{entry}: inputs, state, params, and rng must have distinct names, got {names}.")
        if isinstance(cfg, (Command, Event)) and cfg.term.state:
            errors.append(f"{entry}: commands and events cannot keep state.")
        if isinstance(cfg, Event) and (cfg.mode is EventMode.INTERVAL) != (cfg.interval is not None):
            errors.append(f"{entry}: interval events need an interval, and other events cannot have one.")
        if len(errors) > num_errors:
            return None

        timer = None
        if isinstance(cfg, Command):
            timer = cfg.resampling_time
        elif isinstance(cfg, Event):
            timer = cfg.interval
        stream = None
        if timer is not None or cfg.term.random:
            stream = len(streams)
            streams.append(str(entry))

        group = ""
        columns = slice(None)
        if isinstance(cfg, Observation):
            group = entry.path[1]
            start = observation_dims.get(group, 0)
            columns = slice(start, start + math.prod(output_shape))
            observation_dims[group] = columns.stop
        return TermOp(
            name=str(entry),
            compute=implementation.compute,
            update=implementation.update,
            reset=implementation.reset,
            inputs=cfg.inputs,
            params=cfg.params,
            state={
                name: replace(spec, shape=(self.schema.num_worlds, *state_shapes[name]))
                for name, spec in cfg.term.state.items()
            },
            weight=cfg.weight if isinstance(cfg, Reward) else 1.0,
            truncation=cfg.truncation if isinstance(cfg, Termination) else False,
            group=group,
            columns=columns,
            target=FieldSelector("commands", entry.path[1]) if isinstance(cfg, Command) else None,
            shape=output_shape,
            timer=timer,
            stream=stream,
            random=cfg.term.random,
        )

    def _resolve_shape(
        self, entry: ScheduleEntry, role: str, shape: tuple[str | int, ...], sizes: dict[str, int], errors: list[str]
    ) -> tuple[int, ...] | None:
        """Sizes of a symbolic shape's dimensions after ``"worlds"``, or ``None`` with an error."""
        if shape[:1] != ("worlds",) or "worlds" in shape[1:]:
            errors.append(f"{entry}: the {role} shape {shape} must start with 'worlds' and contain it once.")
            return None
        unknown = [dim for dim in shape[1:] if isinstance(dim, str) and dim not in sizes]
        if unknown:
            errors.append(f"{entry}: the {role} dimensions {unknown} are not dimensions of the term's inputs.")
            return None
        return tuple(sizes.get(dim, dim) for dim in shape[1:])

    def _field_dims(
        self, entry: ScheduleEntry, role: str, selector: FieldSelector, errors: list[str]
    ) -> dict[str, int] | None:
        """Sizes of a selected field's dimensions after the world dimension, or ``None`` with an error."""
        entity = self.schema.entities.get(selector.entity)
        if entity is None:
            errors.append(f"{entry}: {role} selects unknown entity {selector.entity!r} (in '{selector}').")
            return None
        if selector.field not in entity.fields:
            errors.append(f"{entry}: {role} selects unknown field '{selector}'. Fields: {sorted(entity.fields)}.")
            return None
        return {dim: len(entity.names[dim]) for dim in entity.fields[selector.field]}
