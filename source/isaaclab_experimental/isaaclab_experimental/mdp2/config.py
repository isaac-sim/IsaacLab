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

    STARTUP_EVENTS = enum.auto()
    ACTIONS = enum.auto()
    QUANTITIES = enum.auto()
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
    """When an event runs: once at startup, on reset, or at random intervals."""

    STARTUP = enum.auto()
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

    def __str__(self) -> str:
        return self.name

    def joints(self, *patterns: str) -> SubsetSelector:
        """Select the joints whose names match any of the patterns."""
        return SubsetSelector(self.name, "joints", patterns)

    def bodies(self, *patterns: str) -> SubsetSelector:
        """Select the bodies whose names match any of the patterns."""
        return SubsetSelector(self.name, "bodies", patterns)


commands = Entity("commands", scene=False)
"""Command values, selected by command name: ``mdp2.commands.target``."""

quantities = Entity("quantities", scene=False)
"""Quantity values, selected by quantity name: ``mdp2.quantities.goal_error``."""

last_action = FieldSelector("mdp", "last_action", scene=False)
"""The last actions, zero after a reset. Their shape is ``("worlds", "actions")``."""

episode_length = FieldSelector("mdp", "episode_length", scene=False)
"""Control steps since the last reset, ``int32`` of shape ``(num_worlds,)``."""


def term_state(use: str, name: str) -> FieldSelector:
    """Select the state ``name`` of another term use, as in ``term_state("rewards.success", "succeeded")``.

    Terms read it as an input; only the use that owns it writes it. Reset events see the state of the episode that
    ended, since term state resets after them. World configs that define the use differently share its state, each
    in its own worlds, so its shapes must agree; state that the worlds share, such as a table filled by ``init``, is
    filled by each of them in turn.
    """
    return FieldSelector(use, name, scene=False)


terminated = FieldSelector("mdp", "terminated", scene=False)
"""Whether a termination, not a time-out, ended the episode this step, ``bool`` of shape ``(num_worlds,)``.

Terminations run before rewards, so rewards can read it, like MDP1's ``is_terminated``.
"""


@dataclass(frozen=True)
class UniformNoise:
    """Noise drawn uniformly from ``[n_min, n_max]`` for each element, like MDP1's ``UniformNoiseCfg``."""

    n_min: float
    n_max: float
    scale: FieldSelector | None = None  # A float field of shape (num_worlds,) that multiplies the noise, if given


@dataclass
class TermCfg:
    """Base configuration of a term use: the term, the fields bound to its inputs, and constant parameters."""

    phase: ClassVar[Phase]

    term: Term
    inputs: dict[str, FieldSelector | SubsetSelector] = field(default_factory=dict)  # Passed by name, read-only
    # Constant keyword arguments. The MDP keeps the values themselves, not copies: do not change them after finalizing.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Quantity(TermCfg):
    """Configuration for a quantity: a value per world that other terms read as ``mdp2.quantities.<name>``.

    Quantities are computed once per step, before the terminations and rewards, so that terms share a calculation
    instead of repeating it. A quantity keeps its value until the next step: reset events read what the episode that
    ended left, such as whether it succeeded. A quantity that observations or action offsets read is the exception:
    it is computed again after a reset, with the quantities that it reads.
    """

    phase: ClassVar[Phase] = Phase.QUANTITIES

    log: bool = False  # Log its average over the worlds' last finished episodes, taken at their end, as a metric
    # Record its value at the end of every episode, in the episode records of an environment with an episode budget
    record: bool = False


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
class ObservationGroup:
    """A group of observation uses, like MDP1's ``ObservationGroupCfg``. A plain dict of uses is short for a
    concatenated group.

    A concatenated group is one ``float32`` buffer of its terms in declaration order. Otherwise each term keeps its
    own buffer, shape, and type, and may output another type, such as a ``uint8`` image, when it has no noise, clip,
    scale, or history.
    """

    terms: dict[str, Observation | None] = field(default_factory=dict)
    concatenate: bool = True  # Concatenate the terms into one buffer, or keep a buffer per term


def group_terms(group: dict[str, Observation | None] | ObservationGroup) -> tuple[dict[str, Observation | None], bool]:
    """Return the uses of an observation group of :attr:`TaskCfg.observations`, and whether they are concatenated."""
    if isinstance(group, ObservationGroup):
        return group.terms, group.concatenate
    return group, True


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
    # Reset events: whether the scene commits the writes of the earlier reset events before it runs, so that it reads
    # what they changed. None infers it: before an event that reads a field derived from written ones, such as the
    # body poses after joint writes. True always commits first, and False never does.
    commit: bool | None = None

    @property
    def phase(self) -> Phase:
        phases = {
            EventMode.STARTUP: Phase.STARTUP_EVENTS,
            EventMode.RESET: Phase.RESET_EVENTS,
            EventMode.INTERVAL: Phase.INTERVAL_EVENTS,
        }
        return phases[self.mode]


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
    offset: float | FieldSelector = 0.0  # A field gives an offset per element, like robot.default_joint_pos
    subset: SubsetSelector | None = None  # Elements of the target to write, like robot.joints("slider_to_cart")


@dataclass
class Action(TermCfg):
    """Configuration for an action term, whose hooks turn the columns of the action into writes to ``outputs``.

    The term's shape is ``("worlds", n)``, with the number of columns ``n`` it takes, and its hooks get them as
    ``action``: a Python hook ``compute(action, **args)`` writes its outputs in place, and a func
    ``compute(world, action, **args)`` writes its world. Actions apply before every physics step, as in MDP1.
    """

    phase: ClassVar[Phase] = Phase.ACTIONS

    outputs: dict[str, FieldSelector] = field(default_factory=dict)  # Passed by argument name; the action writes them


@dataclass
class TaskCfg:
    """Configuration of the MDP of a task. Setting a term use to ``None`` disables it."""

    actions: dict[str, DirectAction | Action | None] = field(default_factory=dict)
    commands: dict[str, Command | None] = field(default_factory=dict)
    quantities: dict[str, Quantity | None] = field(default_factory=dict)
    # Groups, such as "policy", of observation uses: a dict of uses is a concatenated group, see ObservationGroup.
    observations: dict[str, dict[str, Observation | None] | ObservationGroup] = field(default_factory=dict)
    rewards: dict[str, Reward | None] = field(default_factory=dict)
    terminations: dict[str, Termination | TimeLimit | None] = field(default_factory=dict)
    events: dict[str, Event | None] = field(default_factory=dict)


def compose(*tasks: TaskCfg) -> TaskCfg:
    """Merge partial task configs, such as those of a scene, an embodiment, and a task, into one task config.

    Uses keep the order of the tasks, then their order within each task, and observation groups merge term by term.
    Two tasks may not both define a use, unless the later one sets it to ``None``, which removes it. Raises a single
    ``ValueError`` that lists every conflict.
    """
    composed, errors = TaskCfg(), []
    owners: dict[tuple[str, ...], int] = {}  # Task that defines each use
    concatenated: dict[str, bool] = {}  # Whether each observation group is concatenated

    def merge(path: tuple[str, ...], target: dict, uses: dict, index: int) -> None:
        for name, use in uses.items():
            if use is not None and target.get(name) is not None:
                errors.append(f"Tasks {owners[(*path, name)]} and {index} both define '{'.'.join((*path, name))}'.")
                continue
            owners[(*path, name)] = index
            target[name] = use

    for index, task in enumerate(tasks):
        merge(("actions",), composed.actions, task.actions, index)
        merge(("commands",), composed.commands, task.commands, index)
        merge(("quantities",), composed.quantities, task.quantities, index)
        merge(("rewards",), composed.rewards, task.rewards, index)
        merge(("terminations",), composed.terminations, task.terminations, index)
        merge(("events",), composed.events, task.events, index)
        for group, observations in task.observations.items():
            terms, concatenate = group_terms(observations)
            if concatenated.setdefault(group, concatenate) != concatenate:
                errors.append(f"Task {index} and an earlier task disagree on whether group '{group}' is concatenated.")
                continue
            target = composed.observations.setdefault(group, ObservationGroup(concatenate=concatenate)).terms
            merge(("observations", group), target, terms, index)

    if errors:
        raise ValueError("Cannot compose the tasks:\n  " + "\n  ".join(errors))
    return composed
