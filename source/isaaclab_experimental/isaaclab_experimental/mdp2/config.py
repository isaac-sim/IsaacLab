# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration: entities, field selectors, and term uses. Pure data, resolved by the builder."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .terms import Term


class Phase(enum.IntEnum):
    """Where a term use runs in a step. Uses run in phase order, then in the order they were added."""

    ACTIONS = enum.auto()
    TERMINATIONS = enum.auto()
    REWARDS = enum.auto()
    COMMANDS = enum.auto()
    EVENTS = enum.auto()
    OBSERVATIONS = enum.auto()


class EventMode(enum.Enum):
    """When an event runs."""

    RESET = enum.auto()
    INTERVAL = enum.auto()


@dataclass(frozen=True)
class FieldSelector:
    """A data field of a scene entity, for example ``robot.joint_pos``."""

    entity: str
    field: str

    def __str__(self) -> str:
        return f"{self.entity}.{self.field}"


@dataclass(frozen=True)
class Entity:
    """A scene entity. Attribute access selects a field: ``robot.joint_vel``."""

    name: str

    def __getattr__(self, name: str) -> FieldSelector:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldSelector(self.name, name)


commands = Entity("commands")
"""Command values, selected by command name: ``mdp2.commands.target``."""


@dataclass
class TermCfg:
    """A term bound to inputs and parameters."""

    phase: ClassVar[Phase]

    term: Term
    inputs: dict[str, FieldSelector] = field(default_factory=dict)  # Entity fields bound to input arguments
    params: dict[str, Any] = field(default_factory=dict)  # Constant keyword arguments


@dataclass
class Observation(TermCfg):
    """A term whose output, flattened per world, is written to an observation group."""

    phase: ClassVar[Phase] = Phase.OBSERVATIONS


@dataclass
class Reward(TermCfg):
    """A term whose output is added to the reward as ``weight * value * step_dt``."""

    phase: ClassVar[Phase] = Phase.REWARDS

    weight: float = 1.0


@dataclass
class Termination(TermCfg):
    """A term whose output ends the episode of the worlds where it is true."""

    phase: ClassVar[Phase] = Phase.TERMINATIONS

    time_out: bool = False  # Whether a true output is a time-out (truncation) rather than a termination


@dataclass
class Command(TermCfg):
    """A term whose output is sampled into a command value when its timer expires and on reset."""

    phase: ClassVar[Phase] = Phase.COMMANDS

    resampling_time_range: tuple[float, float] | None = None  # Seconds between resamples, None for only on reset


@dataclass
class Event(TermCfg):
    """A term that writes scene fields of the worlds selected by its ``mask`` argument."""

    phase: ClassVar[Phase] = Phase.EVENTS

    mode: EventMode = EventMode.RESET
    interval_range_s: tuple[float, float] | None = None  # Seconds between interval events


@dataclass
class TimeLimit:
    """Truncates episodes after a fixed duration."""

    phase: ClassVar[Phase] = Phase.TERMINATIONS

    episode_length_s: float  # Episode duration [s]


@dataclass
class DirectAction:
    """Writes ``scale * action + offset`` to an entity field."""

    phase: ClassVar[Phase] = Phase.ACTIONS

    target: FieldSelector
    scale: float = 1.0
    offset: float = 0.0


@dataclass(init=False)
class ObservationGroup:
    """Observations concatenated, in declaration order, into one buffer per group."""

    terms: dict[str, Observation]

    def __init__(self, **terms: Observation):
        self.terms = terms


@dataclass
class TaskCfg:
    """The decision-making part of a task. A term set to ``None`` is disabled."""

    actions: dict[str, DirectAction | None] = field(default_factory=dict)
    commands: dict[str, Command | None] = field(default_factory=dict)
    observations: dict[str, ObservationGroup] = field(default_factory=dict)
    rewards: dict[str, Reward | None] = field(default_factory=dict)
    terminations: dict[str, Termination | TimeLimit | None] = field(default_factory=dict)
    events: dict[str, Event | None] = field(default_factory=dict)
