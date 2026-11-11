# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled ops, the execution plan that lists the steps each runtime function runs, and the launches that run them.

A backend compiles the plan into launches: kernels, and hooks that launch their own work. Consecutive capturable
launches form a segment that can be captured as one CUDA graph; eager terms and other non-capturable hooks run
between the graphs.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import FieldSelector, Phase, SubsetSelector, UniformNoise
from .terms import State

if TYPE_CHECKING:
    from isaaclab.utils.warp import ProxyArray

    from .compiled import MDPState, Transition


@dataclass(frozen=True)
class ActionOp:
    """Compiled action that writes a slice of the action to an entity field."""

    name: str
    target: FieldSelector
    columns: slice  # Columns of the action
    scale: float
    offset: float | FieldSelector  # A field gives an offset per element
    worlds: tuple[int, ...] | None = None  # Worlds where the op is active, None for every world
    subset: SubsetSelector | None = None  # Elements of the target it writes, None for all of them


@dataclass(frozen=True)
class TermOp:
    """Compiled term use for one backend."""

    name: str  # Unique among the ops; the use's name, with a suffix if other world configs define it differently
    use: str  # Name of the term use, such as "rewards.alive", which keys its random stream
    phase: Phase
    index: int  # Position among the ops of its phase
    compute: Callable
    update: Callable | None
    reset: Callable | None
    inputs: Mapping[str, FieldSelector | SubsetSelector]  # Hook arguments bound to fields and subsets
    params: Mapping[str, Any]
    state: Mapping[str, State]  # Shapes resolved to sizes, including the world dimension
    shape: tuple[int, ...]  # Output dimensions after the world dimension
    dtype: type  # Output element type
    writes: tuple[FieldSelector, ...] = ()  # A command's value, or the outputs of an event
    weight: float = 1.0  # Rewards: initial value of MDPState.reward_weights[index]
    time_out: bool = False  # Terminations: whether a true output truncates
    group: str = ""  # Observations: name of the observation group
    columns: slice = slice(None)  # Observations: columns of the group
    noise: UniformNoise | None = None  # Observations
    clip: tuple[float, float] | None = None  # Observations
    scale: float | None = None  # Observations
    history_length: int = 0  # Observations: steps of history in the group, 0 for the latest value only
    noise_stream: int | None = None  # Observations: row of MDPState.rng for the noise
    timer: tuple[float, float] | None = None  # Seconds between command resamples or interval events
    stream: int | None = None  # Row of MDPState.rng, for timers and random terms
    random: bool = False  # Whether the term draws from its stream
    capturable: bool = True  # Whether its hooks may run inside a CUDA graph
    eager: bool = False  # Whether ``compute`` is an eager term's ``fn(ctx, **params)``, see :mod:`.eager`
    worlds: tuple[int, ...] | None = None  # Worlds where the op is active, None for every world


class StepKind(enum.Enum):
    """Kind of work a step does with its op."""

    COMPUTE = enum.auto()  # Compute the output and write it to its destination
    COMPUTE_IF_DUE = enum.auto()  # Count down the timer, and compute for the worlds where it expired
    UPDATE_STATE = enum.auto()  # Advance the term's state after its compute, or a command's value every step
    RESET_TIMER = enum.auto()  # Draw a new time until the command resamples or the interval event runs
    RESET_STATE = enum.auto()  # Run the reset hook, or set the state to its initial value


@dataclass(frozen=True)
class Step:
    """One unit of work on an op in a runtime function."""

    kind: StepKind
    op: ActionOp | TermOp

    @property
    def capturable(self) -> bool:
        """Whether the step may run inside a CUDA graph. Only steps that call a non-capturable hook may not."""
        if isinstance(self.op, ActionOp) or self.op.capturable or self.kind is StepKind.RESET_TIMER:
            return True
        return self.kind is StepKind.RESET_STATE and self.op.reset is None


@dataclass(frozen=True)
class Segment:
    """Consecutive steps of a runtime function that run as one CUDA graph, or eagerly between graphs."""

    capturable: bool
    steps: tuple[Step, ...]


def segments(steps: Sequence[Step]) -> tuple[Segment, ...]:
    """Split the steps of a runtime function into runs of capturable and non-capturable steps."""
    groups = itertools.groupby(steps, lambda step: step.capturable)
    return tuple(Segment(capturable, tuple(group)) for capturable, group in groups)


@dataclass(frozen=True)
class ExecutionPlan:
    """Ordered steps of each runtime function. ``reset`` and ``observe`` only act on the masked worlds."""

    startup: tuple[Step, ...]
    apply_actions: tuple[Step, ...]
    evaluate: tuple[Step, ...]
    reset: tuple[Step, ...]
    observe: tuple[Step, ...]


def build_plan(ops: Mapping[Phase, tuple]) -> ExecutionPlan:
    """Order the steps of every runtime function from the ops of each phase. This defines the step contract."""
    timed = [op for op in ops[Phase.COMMANDS] + ops[Phase.INTERVAL_EVENTS] if op.timer is not None]
    observations = tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.OBSERVATIONS])
    # Commands write their state when they resample.
    stateful = [op for phase in Phase if phase not in (Phase.ACTIONS, Phase.COMMANDS) for op in ops[phase] if op.state]

    terminations_and_rewards = []
    for op in ops[Phase.TERMINATIONS] + ops[Phase.REWARDS]:
        terminations_and_rewards.append(Step(StepKind.COMPUTE, op))
        if op.update is not None:
            terminations_and_rewards.append(Step(StepKind.UPDATE_STATE, op))

    return ExecutionPlan(
        startup=tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.STARTUP_EVENTS]),
        apply_actions=tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.ACTIONS]),
        evaluate=(
            # Terminations and rewards see the commands that governed the action.
            *terminations_and_rewards,
            *(Step(StepKind.COMPUTE_IF_DUE, op) for op in timed if op.phase is Phase.COMMANDS),
            # Commands with an update hook track something every step, like a heading.
            *(Step(StepKind.UPDATE_STATE, op) for op in ops[Phase.COMMANDS] if op.update is not None),
            *(Step(StepKind.COMPUTE_IF_DUE, op) for op in timed if op.phase is Phase.INTERVAL_EVENTS),
            # Successor observations see the new commands and the events. Their state advances after all of them.
            *observations,
            *(Step(StepKind.UPDATE_STATE, op) for op in ops[Phase.OBSERVATIONS] if op.update is not None),
        ),
        reset=(
            *(Step(StepKind.RESET_TIMER, op) for op in timed),
            *(Step(StepKind.COMPUTE, op) for op in ops[Phase.RESET_EVENTS] + ops[Phase.COMMANDS]),
            # State restarts last, from the fields and commands of the new episode.
            *(Step(StepKind.RESET_STATE, op) for op in stateful),
        ),
        observe=observations,
    )


@dataclass
class Call:
    """Arguments of one runtime function call, from which launches read their arguments."""

    state: MDPState
    actions: ProxyArray | None = None
    mask: ProxyArray | None = None
    transition: Transition | None = None
    observations: dict[str, ProxyArray] | None = None


@dataclass(frozen=True)
class Launch:
    """Work that a backend runs as one unit: a kernel, or a hook that launches its own work."""

    run: Callable[[Call], None]
    capturable: bool = True

    def __call__(self, call: Call) -> None:
        self.run(call)


@dataclass(frozen=True)
class Launches:
    """Launches of each runtime function, in order, compiled from the execution plan by a backend."""

    startup: tuple[Callable[[Call], None], ...]
    apply_actions: tuple[Callable[[Call], None], ...]
    evaluate: tuple[Callable[[Call], None], ...]
    reset: tuple[Callable[[Call], None], ...]
    observe: tuple[Callable[[Call], None], ...]
