# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled ops, the execution plan that lists the steps each runtime function runs, and the launches that run them.

A backend compiles the plan into launches: kernels, and hooks that launch their own work. Consecutive capturable
launches form a segment that can be captured as one CUDA graph; hooks that cannot be captured run
between the graphs.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Entity, FieldSelector, Phase, SubsetSelector, TaskMode, UniformNoise, last_action, progress_field
from .terms import State

if TYPE_CHECKING:
    from isaaclab.utils.warp import ProxyArray

    from .compiled import MDPState, Transition
    from .scene import SceneSchema


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

    @property
    def active_key(self) -> Hashable | None:
        """Key in :attr:`MDPState.arrays` of the worlds where the op is active, None for every world."""
        return None if self.worlds is None else ("active", self.name)


@dataclass(frozen=True)
class ObservationOutput:
    """Where an observation op writes its output, and how the output is processed first."""

    group: str  # Name of the observation group
    columns: slice  # Columns of the group, for every step of the history
    noise: UniformNoise | None = None
    clip: tuple[float, float] | None = None
    scale: float | None = None
    history_length: int = 0  # Steps of history in the group, 0 for the latest value only
    noise_stream: int | None = None  # Row of MDPState.rng for the noise
    slot: str = ""  # Name of its slot of the buffer, which the ops of the same name share
    size: int = 0  # Columns of one step of history: those of its slot, which may be wider than the op's output


@dataclass(frozen=True)
class TermOp:
    """Compiled term use for one backend."""

    name: str  # Unique among the ops; the use's name, with a suffix if other world configs define it differently
    use: str  # Name of the term use, such as "rewards.alive", which keys its random stream and its state
    phase: Phase
    index: int  # Position among the ops of its phase
    compute: Callable
    update: Callable | None
    reset: Callable | None
    inputs: Mapping[str, FieldSelector | SubsetSelector | Entity]  # Hook arguments bound to fields, subsets, entities
    params: Mapping[str, Any]
    state: Mapping[str, State]  # Shapes resolved to sizes, including the world dimension
    shape: tuple[int, ...]  # Output dimensions after the world dimension
    dtype: type  # Output element type
    writes: tuple[FieldSelector, ...] = ()  # A command's value, or the outputs of an event
    weights: tuple[float, ...] = ()  # Rewards: initial weight per world config, 0 where the op is inactive
    time_out: bool = False  # Terminations: whether a true output truncates
    observation: ObservationOutput | None = None  # Observations only
    timer: tuple[float, float] | None = None  # Seconds between command resamples or interval events
    stream: int | None = None  # Row of MDPState.rng, for timers and random terms
    random: bool = False  # Whether the term draws from its stream
    capturable: bool = True  # Whether its hooks may run inside a CUDA graph
    torch_hooks: bool = False  # Whether a Warp MDP runs the term's Torch hooks, for a term without Warp ones
    init: Callable | None = None  # Eager ``fn(ctx, **params)`` that fills the state once, before the startup events
    # Reset events: whether the scene commits the earlier events' writes before it runs, None to infer it
    commit: bool | None = False
    log: bool = False  # Quantities: whether the environment logs it as a metric
    record: bool = False  # Quantities: whether the environment records it at the end of every episode
    columns: slice | None = None  # Actions: the columns of the action that the term takes
    worlds: tuple[int, ...] | None = None  # Worlds where the op is active, None for every world
    # Actions of its world config, if it reads the last actions and other world configs have more
    action_dim: int | None = None
    # The task of a sequence that the op belongs to: it runs only while that task runs, and its state, timer, and
    # command restart when the task starts. Quantities run in every step: they only restart.
    stage: str | None = None

    @property
    def active_key(self) -> Hashable | None:
        """Key in :attr:`MDPState.arrays` of the worlds where the op runs at this step, None for every world."""
        if self.stage is not None and self.phase is not Phase.QUANTITIES:
            return progress_field(self.stage, "active")
        return None if self.worlds is None else ("active", self.name)

    def array_key(self, arg: str) -> Hashable:
        """Return the key in :attr:`MDPState.arrays` of the array that the input ``arg`` reads."""
        selector = self.inputs[arg]
        if selector == last_action and self.action_dim is not None:
            return last_action, self.action_dim
        return selector


class StepKind(enum.Enum):
    """Kind of work a step does with its op."""

    COMPUTE = enum.auto()  # Compute the output and write it to its destination
    COMPUTE_IF_DUE = enum.auto()  # Count down the timer, and compute for the worlds where it expired
    UPDATE_STATE = enum.auto()  # Advance the term's state after its compute, or a command's value every step
    RESET_TIMER = enum.auto()  # Draw a new time until the command resamples or the interval event runs
    RESET_STATE = enum.auto()  # Run the reset hook, or set the state to its initial value
    COMMIT = enum.auto()  # Have the scene commit the writes of the reset events so far, before the op's event
    UPDATE_PROGRESS = enum.auto()  # Count the tasks' completions, bottom up, and advance the sequences
    START_TASKS = enum.auto()  # Start the tasks that a sequence advanced to, top down, and stop those it left
    RESET_PROGRESS = enum.auto()  # Restart the progress of every task for a new episode
    CLEAR = enum.auto()  # Zero the columns of an observation slot that no op writes at this step


@dataclass(frozen=True)
class TaskNode:
    """A task of the task tree of a world config, with what its progress depends on."""

    path: str  # Dotted path of names from the root task, "" for the root
    mode: TaskMode
    children: tuple[str, ...]  # Paths of its subtasks, in order
    done: FieldSelector | None  # Completion of a task without subtasks
    done_dtype: type | None  # Element type of ``done``
    hold: int
    score: float  # Weight in its parent's score
    final: bool | None
    require: int | None
    staged: bool  # Whether it runs only in some steps of an episode: a sequence is one of its ancestors


@dataclass(frozen=True)
class TaskTree:
    """The tasks of one world config, parents before children, and the tasks among them that have a completion."""

    worlds: range
    nodes: tuple[TaskNode, ...]
    completes: frozenset[str]  # Paths of the tasks that have a completion

    def node(self, path: str) -> TaskNode:
        return next(node for node in self.nodes if node.path == path)

    def parent(self, node: TaskNode) -> TaskNode | None:
        return None if not node.path else self.node(node.path.rpartition(".")[0])

    def runs_at_start(self, node: TaskNode) -> bool:
        """Return whether a task runs at the start of an episode: all but the later tasks of a sequence do."""
        parent = self.parent(node)
        if parent is None:
            return True
        first = parent.mode is not TaskMode.SEQUENCE or parent.children[0] == node.path
        return first and self.runs_at_start(parent)

    def counted(self, node: TaskNode) -> tuple[TaskNode, ...]:
        """Return the subtasks of a task that count towards its completion: those that have one."""
        return tuple(self.node(child) for child in node.children if child in self.completes)


@dataclass(frozen=True)
class ObservationSlot:
    """Columns of an observation buffer that several ops write, or that an op writes only in some steps."""

    buffer: str
    columns: slice  # For every step of the history
    size: int  # Columns of one step of history
    ops: tuple[TermOp, ...]  # Ops that write it, each in its own worlds and steps


@dataclass(frozen=True)
class Step:
    """One unit of work on an op, the task trees, or an observation slot, in a runtime function."""

    kind: StepKind
    op: ActionOp | TermOp | tuple[TaskTree, ...] | ObservationSlot
    entry: bool = False  # Whether it runs for the worlds whose task of the op just started, rather than for the mask

    @property
    def capturable(self) -> bool:
        """Whether the step may run inside a CUDA graph. Only steps that call a non-capturable hook may not."""
        if not isinstance(self.op, TermOp) or self.op.capturable:
            return True
        if self.kind in (StepKind.RESET_TIMER, StepKind.COMMIT):
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


def build_plan(
    ops: Mapping[Phase, tuple],
    schema: SceneSchema,
    trees: tuple[TaskTree, ...] = (),
    slots: tuple[ObservationSlot, ...] = (),
) -> ExecutionPlan:
    """Order the steps of every runtime function from the ops of each phase. This defines the step contract.

    With task trees, ``evaluate`` counts the tasks' completions after the quantities, so that terminations and
    rewards see this step's progress, and starts the tasks that sequences advanced to after the events, so that the
    successor observations see them. The terminations, rewards, commands, and events of the step run in the tasks that
    ran when it started.
    """
    timed = [op for op in ops[Phase.COMMANDS] + ops[Phase.INTERVAL_EVENTS] if op.timer is not None]
    observations = tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.OBSERVATIONS])
    clears = tuple(Step(StepKind.CLEAR, slot) for slot in slots)
    # Commands write their state when they resample, and persistent state outlives episodes.
    stateful = [
        op
        for phase in Phase
        if phase is not Phase.COMMANDS
        for op in ops[phase]
        if isinstance(op, TermOp) and (op.reset is not None or any(not spec.persistent for spec in op.state.values()))
    ]

    # Observations and action offsets read quantities between steps, also in worlds that were just reset, where
    # they are computed again, with the quantities that they read. The others keep the values of the last step.
    read = {selector for op in ops[Phase.OBSERVATIONS] for selector in op.inputs.values()}
    noises = [op.observation.noise for op in ops[Phase.OBSERVATIONS]]
    read |= {noise.scale for noise in noises if noise is not None and noise.scale is not None}
    read |= {op.offset for op in ops[Phase.ACTIONS] if isinstance(op, ActionOp)}
    observed = []
    for op in reversed(ops[Phase.QUANTITIES]):
        if op.writes[0] in read:
            observed.insert(0, Step(StepKind.COMPUTE, op))
            read |= set(op.inputs.values())

    def computed(phase_ops: tuple) -> list[Step]:
        steps = []
        for op in phase_ops:
            steps.append(Step(StepKind.COMPUTE, op))
            if op.update is not None:
                steps.append(Step(StepKind.UPDATE_STATE, op))
        return steps

    progress = [Step(StepKind.UPDATE_PROGRESS, trees)] if trees else []
    # A task that starts restarts the timers, commands, and state of its ops, as a new episode does.
    starts = []
    if any(node.staged for tree in trees for node in tree.nodes):
        staged = [op for phase_ops in ops.values() for op in phase_ops if isinstance(op, TermOp) and op.stage]
        starts = [
            Step(StepKind.START_TASKS, trees),
            *(Step(StepKind.RESET_TIMER, op, entry=True) for op in timed if op.stage),
            *(Step(StepKind.COMPUTE, op, entry=True) for op in staged if op.phase is Phase.COMMANDS),
            *(Step(StepKind.RESET_STATE, op, entry=True) for op in stateful if op.stage),
        ]

    reset_events = []
    pending: set[FieldSelector] = set()  # Scene fields written since the last commit
    for op in ops[Phase.RESET_EVENTS]:
        if op.commit or op.commit is None and pending and _reads_stale(op, pending, schema):
            reset_events.append(Step(StepKind.COMMIT, op))
            pending.clear()
        reset_events.append(Step(StepKind.COMPUTE, op))
        pending.update(selector for selector in op.writes if selector.scene)

    return ExecutionPlan(
        startup=tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.STARTUP_EVENTS]),
        apply_actions=tuple(Step(StepKind.COMPUTE, op) for op in ops[Phase.ACTIONS]),
        evaluate=(
            # Quantities, progress, terminations, and rewards see the commands that governed the action.
            *computed(ops[Phase.QUANTITIES]),
            *progress,
            *computed(ops[Phase.TERMINATIONS] + ops[Phase.REWARDS]),
            *(Step(StepKind.COMPUTE_IF_DUE, op) for op in timed if op.phase is Phase.COMMANDS),
            # Commands with an update hook track something every step, like a heading.
            *(Step(StepKind.UPDATE_STATE, op) for op in ops[Phase.COMMANDS] if op.update is not None),
            *(Step(StepKind.COMPUTE_IF_DUE, op) for op in timed if op.phase is Phase.INTERVAL_EVENTS),
            *starts,
            # Successor observations see the new commands, the events, and the tasks that started. Their state
            # advances after all of them.
            *observations,
            *clears,
            *(Step(StepKind.UPDATE_STATE, op) for op in ops[Phase.OBSERVATIONS] if op.update is not None),
        ),
        reset=(
            *([Step(StepKind.RESET_PROGRESS, trees)] if trees else []),
            *(Step(StepKind.RESET_TIMER, op) for op in timed),
            *reset_events,
            *(Step(StepKind.COMPUTE, op) for op in ops[Phase.COMMANDS]),
            # State restarts last, from the fields and commands of the new episode.
            *(Step(StepKind.RESET_STATE, op) for op in stateful),
        ),
        observe=(*observed, *observations, *clears),
    )


def _reads_stale(op: TermOp, pending: set[FieldSelector], schema: SceneSchema) -> bool:
    """Return whether a reset event reads a scene field derived from fields that earlier events wrote, which the
    scene has not recomputed yet, or takes an entity whose fields they wrote, whose data it may read unseen."""
    written = {(selector.entity, selector.field) for selector in pending}
    entities = {entity for entity, _ in written}
    for selector in op.inputs.values():
        if isinstance(selector, Entity):
            if selector.name in entities:
                return True
            continue
        if not isinstance(selector, FieldSelector) or not selector.scene or selector in op.writes:
            continue
        derived_from = schema.entities[selector.entity].fields[selector.field].derived_from
        if any((selector.entity, field) in written for field in derived_from):
            return True
    return False


@dataclass
class Call:
    """Arguments of one runtime function call, from which launches read their arguments."""

    state: MDPState
    actions: ProxyArray | None = None
    mask: ProxyArray | None = None
    transition: Transition | None = None
    observations: dict[str, ProxyArray] | None = None
    commit: Callable[[], None] | None = None  # Commits the scene writes of the reset events, see StepKind.COMMIT


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
