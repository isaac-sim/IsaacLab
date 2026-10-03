# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of a baked, time-sampled recording of the live smoothie controller.

The scene has no randomization. A recorded run of
``SmoothieSequenceController`` supplies the actions for the demonstration.
Playback indexes the recorded raw actions by policy step; it performs no IK,
no bounded least-squares solve, and no live tracking/physical gating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class RecordedSequenceController:
    """Replay a baked ``(actions, stages)`` recording as raw robot actions."""

    def __init__(self, env: Any, *, recording_path: Path, trace_path: Path | None = None):
        path = Path(recording_path)
        with np.load(path) as data:
            actions = data["actions"]
            stages = data["stages"]
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"Recorded actions at {path} must have shape [N, 8].")
        if stages.shape[0] != actions.shape[0]:
            raise ValueError(f"Recorded actions and stages at {path} must have matching length.")
        self.env = env
        self.actions = actions.astype(np.float32, copy=False)
        self.stages = stages
        self.previous_step: int | None = None
        # The runner closes this owned stream through close_trace in its finally block.
        self.trace = None if trace_path is None else Path(trace_path).open("x", buffering=1)  # noqa: SIM115

    @property
    def stage(self) -> str:
        """Recorded stage label for the most recently computed step."""
        index = 0 if self.previous_step is None else self.previous_step
        return str(self.stages[min(index, len(self.stages) - 1)])

    def compute(self, step: int) -> torch.Tensor:
        """Return the recorded raw robot actions for ``step``, shape [1, 8]."""
        if type(step) is not int or step < 0 or (self.previous_step is not None and step != self.previous_step + 1):
            raise ValueError("Controller calls require consecutive nonnegative policy steps.")
        if step >= len(self.actions):
            raise IndexError(f"Recorded trajectory has {len(self.actions)} steps; step {step} was requested.")
        self.previous_step = step
        actions = torch.as_tensor(self.actions[step], device=self.env.device)[None]
        if self.trace is not None:
            self.trace.write(f"{step}\t{step * self.env.step_dt:.3f}\t{self.stage}\n")
        return actions

    def close_trace(self) -> None:
        """Close the optional controller event trace."""
        if self.trace is not None:
            self.trace.close()
