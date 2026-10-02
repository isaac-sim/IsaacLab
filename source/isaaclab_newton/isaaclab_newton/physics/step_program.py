# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One Newton step as a straight-line program over explicit buffers."""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import warp as wp


class StepPhase(enum.IntEnum):
    """Where a :class:`StepStage` runs inside the Newton step program."""

    COMMAND = 0
    """Once per physics step, after collision and before Newton actuators.

    Controllers and action terms write joint targets and efforts here, so actuators consume commands computed from
    the current physics step's state.
    """

    CONTROL = 1
    """Once per physics step, after Newton actuators and before the solver substeps."""

    SUBSTEP = 2
    """Before every solver substep. The callable receives the substep's input :class:`newton.State`."""

    POST_STEP = 3
    """Once after the final physics step of the program, before Newton sensors update."""


@dataclass(frozen=True, eq=False)
class StepStage:
    """A consumer operation scheduled into every Newton step program.

    Stages compose the step: controllers run in :attr:`StepPhase.COMMAND`, actuator telemetry in
    :attr:`StepPhase.CONTROL`, applied forces in :attr:`StepPhase.SUBSTEP`, and state republishing in
    :attr:`StepPhase.POST_STEP`. A graph-safe stage is recorded
    into the CUDA graph together with its neighbors; any other stage runs eagerly at the same position in the program.
    """

    fn: Callable[..., None]
    """Operation to run. :attr:`StepPhase.SUBSTEP` stages receive the input state; other stages take no arguments."""

    phase: StepPhase
    """Program position of the stage."""

    graph_safe: bool = True
    """Whether the operation has fixed buffers and shapes and no host branching on device data."""

    name: str = ""
    """Label used in errors and profiles."""


@dataclass(frozen=True)
class StepOp:
    """One operation of a compiled program, with every buffer already bound."""

    fn: Callable[[], None]
    graph_safe: bool
    name: str


def run_ops(ops: Sequence[StepOp]) -> None:
    """Run operations in order."""
    for op in ops:
        op.fn()


class StepProgram:
    """Operations of one Newton ``step()``, grouped into capturable segments.

    Consecutive graph-safe operations form a segment captured as one CUDA graph. Operations that are not graph-safe
    run eagerly between segments. Eager execution runs the same operations in the same order, so the captured and
    eager paths cannot diverge. A caller that owns an outer capture runs :meth:`run` without :meth:`capture`.
    """

    def __init__(self, ops: Sequence[StepOp], steps: int):
        """Initialize the program.

        Args:
            ops: Operations in execution order.
            steps: Physics steps the program advances.
        """
        self.ops = tuple(ops)
        self.steps = steps
        self.segments = tuple(
            (graph_safe, tuple(group))
            for graph_safe, group in itertools.groupby(self.ops, key=lambda op: op.graph_safe)
        )
        self._graphs: tuple[wp.Graph | None, ...] | None = None

    @property
    def graph_safe(self) -> bool:
        """Whether every operation can be recorded into a CUDA graph."""
        return all(op.graph_safe for op in self.ops)

    @property
    def is_captured(self) -> bool:
        """Whether graph-safe segments replay from captured graphs."""
        return self._graphs is not None

    def capture(self, capture: Callable[[Callable[[], None]], wp.Graph]) -> None:
        """Record every graph-safe segment without executing it.

        Args:
            capture: Records a callable into a graph on the simulation device.
        """
        self._graphs = tuple(
            capture(lambda ops=ops: run_ops(ops)) if graph_safe else None for graph_safe, ops in self.segments
        )

    def require_graph_safe(self) -> None:
        """Raise unless every operation can be recorded into a CUDA graph.

        Raises:
            RuntimeError: Naming the operations that cannot be recorded.
        """
        if not self.graph_safe:
            eager = sorted({op.name for op in self.ops if not op.graph_safe})
            raise RuntimeError(
                f"The Newton step program cannot be recorded into an outer CUDA graph: {eager} are not graph-safe."
            )

    def record(self) -> None:
        """Launch every operation in order, ignoring captured graphs, so a caller's capture records them.

        Raises:
            RuntimeError: If an operation cannot be recorded into a CUDA graph.
        """
        self.require_graph_safe()
        run_ops(self.ops)

    def run(self) -> None:
        """Advance physics by :attr:`steps` physics steps."""
        if self._graphs is None:
            run_ops(self.ops)
            return
        for (_, ops), graph in zip(self.segments, self._graphs):
            if graph is None:
                run_ops(ops)
            else:
                wp.capture_launch(graph)
