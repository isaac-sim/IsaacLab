# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builder that collects schedule entries and compiles them into an :class:`MDP`, like Newton's ``ModelBuilder``."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DirectAction, FieldSelector, Observation, Phase, Reward, TaskCfg, Termination, TimeLimit
from .mdp import MDP, ActionOp, Backend, TermOp
from .scene import SceneSchema


@dataclass(frozen=True)
class ScheduleEntry:
    """A term use placed in the schedule. Returned by the ``add_*`` methods as a handle to it."""

    path: tuple[str, ...]
    """Location in the task, for example ``("observations", "policy", "joint_pos")``."""
    cfg: DirectAction | Observation | Reward | Termination | TimeLimit

    def __str__(self) -> str:
        return ".".join(self.path)


class MDPBuilder:
    """Collects schedule entries against a scene schema, then compiles them with :meth:`finalize`.

    Actions and observations take columns in the order they are added.
    """

    def __init__(self, schema: SceneSchema):
        self.schema = schema
        self._entries: dict[tuple[str, ...], ScheduleEntry] = {}

    def add_config(self, cfg: TaskCfg) -> None:
        """Add every term use of a task config, in declaration order. Uses set to ``None`` are skipped."""
        for name, action in cfg.actions.items():
            if action is not None:
                self.add_action(name, action)
        for group, observations in cfg.observations.items():
            for name, observation in observations.terms.items():
                if observation is not None:
                    self.add_observation(group, name, observation)
        for name, reward in cfg.rewards.items():
            if reward is not None:
                self.add_reward(name, reward)
        for name, termination in cfg.terminations.items():
            if termination is not None:
                self.add_termination(name, termination)

    def add_action(self, name: str, action: DirectAction) -> ScheduleEntry:
        return self._add(ScheduleEntry(("actions", name), action))

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
        action_dim = 0
        time_limits = []
        for entry in self.schedule():
            cfg = entry.cfg
            if isinstance(cfg, TimeLimit):
                time_limits.append(entry)
            elif isinstance(cfg, DirectAction):
                dims = self._resolve(entry, "target", cfg.target, errors)
                if dims is not None:
                    columns = slice(action_dim, action_dim + math.prod(dims.values()))
                    action_dim = columns.stop
                    ops[Phase.ACTIONS].append(ActionOp(str(entry), cfg.target, columns, cfg.scale, cfg.offset))
            else:
                ops[entry.cfg.phase].append(self._compile_term(entry, backend, errors, observation_dims))
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
            observation_ops=tuple(ops[Phase.OBSERVATIONS]),
            action_dim=action_dim,
            observation_dims=observation_dims,
            max_episode_length=math.ceil(time_limits[0].cfg.seconds / self.schema.step_dt) if time_limits else None,
        )

    def _add(self, entry: ScheduleEntry) -> ScheduleEntry:
        if entry.path in self._entries:
            raise ValueError(f"The schedule already has an entry '{entry}'.")
        self._entries[entry.path] = entry
        return entry

    def _compile_term(
        self, entry: ScheduleEntry, backend: Backend, errors: list[str], observation_dims: dict[str, int]
    ) -> TermOp | None:
        """Compile a reward, termination, or observation. Appends to ``errors`` instead of raising."""
        cfg = entry.cfg
        num_errors = len(errors)
        implementation = cfg.term.torch if backend is Backend.TORCH else cfg.term.warp
        if implementation is None:
            errors.append(f"{entry}: the term has no '{backend.value}' implementation.")
        sizes: dict[str, int] = {}
        for arg, selector in cfg.inputs.items():
            dims = self._resolve(entry, f"input {arg!r}", selector, errors)
            for dim, size in (dims or {}).items():
                if sizes.setdefault(dim, size) != size:
                    errors.append(f"{entry}: inputs disagree on the size of {dim!r} ({sizes[dim]} and {size}).")
        shape = cfg.term.output.shape
        unknown = [dim for dim in shape if isinstance(dim, str) and dim != "worlds" and dim not in sizes]
        if shape[:1] != ("worlds",) or "worlds" in shape[1:]:
            errors.append(f"{entry}: the output shape {shape} must start with 'worlds' and contain it once.")
        elif unknown:
            errors.append(f"{entry}: the output dimensions {unknown} are not dimensions of the term's inputs.")
        elif not isinstance(cfg, Observation) and len(shape) != 1:
            errors.append(f"{entry}: {type(cfg).__name__} outputs must have shape ('worlds',), got {shape}.")
        if len(errors) > num_errors:
            return None

        output_shape = tuple(sizes.get(dim, dim) for dim in shape[1:])
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
            inputs=cfg.inputs,
            params=cfg.params,
            weight=cfg.weight if isinstance(cfg, Reward) else 1.0,
            truncation=cfg.truncation if isinstance(cfg, Termination) else False,
            group=group,
            columns=columns,
            shape=output_shape,
        )

    def _resolve(
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
