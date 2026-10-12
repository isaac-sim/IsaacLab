# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Straight-line step operations grouped into CUDA graphs, with eager islands for work that cannot be captured."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any


@dataclass(frozen=True)
class StepOp:
    """One operation of a :class:`StepGraph`, with every buffer bound."""

    fn: Callable[[], None]
    """Work to run. Graphable work must not synchronize, allocate outside the capture, or branch on device data."""
    graphable: bool
    """Whether the work can be recorded into a CUDA graph."""
    name: str
    """Label used in errors and profiles."""


class StepGraph:
    """A step as straight-line operations, grouped into CUDA graphs.

    Consecutive graphable operations form a segment that replays from one captured graph. Other operations run
    eagerly between segments, so the eager and captured paths run the same operations in the same order.
    """

    def __init__(self, ops: Sequence[StepOp], steps: int = 1):
        """Initialize the graph.

        Args:
            ops: Operations in execution order.
            steps: Steps one launch advances, e.g. physics steps.
        """
        self.ops = tuple(ops)
        self.steps = steps
        self.segments = tuple(
            (graphable, tuple(group)) for graphable, group in itertools.groupby(self.ops, key=lambda op: op.graphable)
        )
        self.graphable = all(op.graphable for op in self.ops)
        self.graphs: tuple[Any | None, ...] | None = None

    @property
    def captured(self) -> bool:
        """Whether graphable segments replay from captured graphs."""
        return self.graphs is not None

    def capture(self, capture: Callable[[Callable[[], None]], Any]) -> None:
        """Record every graphable segment without executing it.

        Args:
            capture: Records a function into a graph and returns an object whose ``launch()`` replays it.
        """
        self.graphs = tuple(capture(partial(_run, ops)) if graphable else None for graphable, ops in self.segments)

    def launch(self) -> None:
        """Run one step, replaying captured segments."""
        if self.graphs is None:
            _run(self.ops)
            return
        for (_, ops), graph in zip(self.segments, self.graphs):
            if graph is None:
                _run(ops)
            else:
                graph.launch()

    def record(self) -> None:
        """Launch every operation so that the caller's active capture records them.

        Raises:
            RuntimeError: If an operation cannot be recorded into a CUDA graph.
        """
        if not self.graphable:
            eager = sorted({op.name for op in self.ops if not op.graphable})
            raise RuntimeError(f"The step cannot be recorded into an outer CUDA graph: {eager} are not graphable.")
        _run(self.ops)


def _run(ops: Sequence[StepOp]) -> None:
    for op in ops:
        op.fn()
