# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration: scene entities, field selectors, and term uses. The builder resolves this plain data."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .terms import Term


class Phase(enum.IntEnum):
    """Kind of a term use, which determines where it runs in a step.

    Uses of the same phase run in schedule order. :func:`.plan.build_plan` defines where each phase runs within the
    runtime functions.
    """

    ACTIONS = enum.auto()
    TERMINATIONS = enum.auto()
    REWARDS = enum.auto()
    COMMANDS = enum.auto()
    INTERVAL_EVENTS = enum.auto()
    OBSERVATIONS = enum.auto()
    RESET_EVENTS = enum.auto()


class Constant(enum.Enum):
    """A param value that the builder resolves, such as ``params={"step_dt": mdp2.step_dt}``."""

    STEP_DT = enum.auto()  # Duration of one control step [s]


step_dt = Constant.STEP_DT
"""Duration of one control step [s], resolved by the builder."""


class EventMode(enum.Enum):
    """When an event runs: on reset, or at random intervals."""

    RESET = enum.auto()
    INTERVAL = enum.auto()


@dataclass(frozen=True)
class FieldSelector:
    """Selects a field of a scene entity (``robot.joint_pos``) or of the MDP (``mdp2.commands.target``)."""

    entity: str
    field: str
    scene: bool = True  # False for data owned by the MDP: commands, the last action, the episode length

    def __str__(self) -> str:
        return f"{self.entity}.{self.field}"


@dataclass(frozen=True)
class SubsetSelector:
    """Selects the elements of an entity dimension whose names match regular expressions, in entity order.

    Bound to an input, as in ``inputs={"joint_ids": robot.joints(".*_hip")}``, it passes the term an ``int32`` index
    array. The argument name also becomes a dimension that the term's shapes can use.
    """

    entity: str
    dim: str
    patterns: tuple[str, ...]  # Regular expressions, matched against whole names

    def __str__(self) -> str:
        return f"{self.entity}.{self.dim}({', '.join(map(repr, self.patterns))})"


@dataclass(frozen=True)
class Entity:
    """A scene entity. Accessing an attribute selects one of its fields, as in ``robot.joint_vel``."""

    name: str
    scene: bool = True

    def __getattr__(self, name: str) -> FieldSelector:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldSelector(self.name, name, self.scene)

    def joints(self, *patterns: str) -> SubsetSelector:
        """Select the joints whose names match any of the patterns."""
        return SubsetSelector(self.name, "joints", patterns)

    def bodies(self, *patterns: str) -> SubsetSelector:
        """Select the bodies whose names match any of the patterns."""
        return SubsetSelector(self.name, "bodies", patterns)


commands = Entity("commands", scene=False)
"""Command values, selected by command name: ``mdp2.commands.target``."""

last_action = FieldSelector("mdp", "last_action", scene=False)
"""The last actions, zero after a reset. Their shape is ``("worlds", "actions")``."""

episode_length = FieldSelector("mdp", "episode_length", scene=False)
"""Control steps since the last reset, ``int32`` of shape ``(num_worlds,)``."""

terminated = FieldSelector("mdp", "terminated", scene=False)
"""Whether a termination, not a time-out, ended the episode this step, ``bool`` of shape ``(num_worlds,)``.

Terminations run before rewards, so rewards can read it, like MDP1's ``is_terminated``.
"""


@dataclass(frozen=True)
class UniformNoise:
    """Noise drawn uniformly from ``[n_min, n_max]`` for each element, like MDP1's ``UniformNoiseCfg``."""

    n_min: float
    n_max: float


@dataclass
class TermCfg:
    """Base configuration of a term use: the term, the fields bound to its inputs, and constant parameters."""

    phase: ClassVar[Phase]

    term: Term
    inputs: dict[str, FieldSelector | SubsetSelector] = field(default_factory=dict)  # Passed by name, read-only
    params: dict[str, Any] = field(default_factory=dict)  # Constant keyword arguments


@dataclass
class Observation(TermCfg):
    """Configuration for an observation term, whose output is flattened per world into its group's buffer.

    As in MDP1, the noise is added first, then the value is clipped, then scaled. With a history, the group holds the
    last ``history_length`` values, oldest first, and a new episode starts with its first value repeated.
    """

    phase: ClassVar[Phase] = Phase.OBSERVATIONS

    noise: UniformNoise | None = None
    clip: tuple[float, float] | None = None  # Bounds of the noisy value
    scale: float | None = None
    history_length: int = 0  # Steps of history to observe, 0 for the latest value only


@dataclass
class Reward(TermCfg):
    """Configuration for a reward term, whose output adds ``weight * value * step_dt`` to the reward."""

    phase: ClassVar[Phase] = Phase.REWARDS

    weight: float = 1.0  # Initial weight, adjustable at runtime through MDPState.reward_weights


@dataclass
class Termination(TermCfg):
    """Configuration for a termination term, which ends the episode of every world where its output is true."""

    phase: ClassVar[Phase] = Phase.TERMINATIONS

    time_out: bool = False  # Whether a true output is a time-out (truncation) rather than a termination


@dataclass
class Command(TermCfg):
    """Configuration for a command term, which is resampled on reset and whenever its resampling timer expires."""

    phase: ClassVar[Phase] = Phase.COMMANDS

    resampling_time_range: tuple[float, float] | None = None  # Seconds between resamples, None for only on reset


@dataclass
class Event(TermCfg):
    """Configuration for an event term, which writes its ``outputs`` for the worlds selected by ``mask``."""

    outputs: dict[str, FieldSelector] = field(default_factory=dict)  # Passed by argument name; the event writes them
    mode: EventMode = EventMode.RESET
    interval_range_s: tuple[float, float] | None = None  # Seconds between interval events

    @property
    def phase(self) -> Phase:
        return Phase.INTERVAL_EVENTS if self.mode is EventMode.INTERVAL else Phase.RESET_EVENTS


@dataclass
class TimeLimit:
    """Truncates episodes after ``episode_length_s`` seconds."""

    phase: ClassVar[Phase] = Phase.TERMINATIONS

    episode_length_s: float  # Episode duration [s]


@dataclass
class DirectAction:
    """Configuration for an action that writes ``scale * action + offset`` to an entity field."""

    phase: ClassVar[Phase] = Phase.ACTIONS

    target: FieldSelector
    scale: float = 1.0
    offset: float = 0.0
    subset: SubsetSelector | None = None  # Elements of the target to write, like robot.joints("slider_to_cart")


@dataclass(init=False)
class ObservationGroup:
    """Observation terms concatenated in declaration order into one buffer, such as ``"policy"``."""

    terms: dict[str, Observation]

    def __init__(self, **terms: Observation):
        self.terms = terms


@dataclass
class TaskCfg:
    """Configuration of the MDP of a task. Setting a term use to ``None`` disables it."""

    actions: dict[str, DirectAction | None] = field(default_factory=dict)
    commands: dict[str, Command | None] = field(default_factory=dict)
    observations: dict[str, ObservationGroup] = field(default_factory=dict)
    rewards: dict[str, Reward | None] = field(default_factory=dict)
    terminations: dict[str, Termination | TimeLimit | None] = field(default_factory=dict)
    events: dict[str, Event | None] = field(default_factory=dict)
