# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wall-clock phase accounting for the native NIST policy loop.

The completion callback drains the selected simulation device. A returned Python
iteration becomes GPU-completed work only after that phase's drain succeeds.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path


class ThroughputMeasurement:
    """Count complete control loops without adding a device wait on each step."""

    def __init__(
        self,
        boards: int,
        warmup_steps: int,
        measured_steps: int,
        complete_device: Callable[[], None],
        *,
        clock: Callable[[], float] = time.perf_counter,
        startup_started: float | None = None,
    ) -> None:
        if boards < 1 or warmup_steps < 0 or measured_steps < 1:
            raise ValueError("Boards and measured steps are positive; warmup steps are nonnegative.")
        self.boards = boards
        self.invocation_id = str(uuid.uuid4())
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.complete_device = complete_device
        self.clock = clock
        self.startup_started = clock() if startup_started is None else startup_started
        self.phase = "startup"
        self.phase_started: float | None = None
        self.returned_steps = {"warmup": 0, "measurement": 0}
        self.completed_steps: dict[str, int | None] = {"warmup": None, "measurement": None}
        self.elapsed: dict[str, float | None] = {"startup": None, "warmup": None, "measurement": None}
        self.episodes: list[dict] = []
        self.episode_starts = [0] * boards
        self.episode_numbers = [0] * boards
        self.configuration: dict = {}

    @property
    def finished(self) -> bool:
        """Whether all requested work has successfully drained."""
        return self.phase == "complete"

    def start(self) -> None:
        """End startup after initial reset, then begin policy warmup or measurement."""
        self.complete_device()
        now = self.clock()
        self.elapsed["startup"] = now - self.startup_started
        self.phase_started = now
        self.phase = "warmup" if self.warmup_steps else "measurement"
        if self.warmup_steps == 0:
            self.completed_steps["warmup"] = 0
            self.elapsed["warmup"] = 0.0

    def observe_episode(self, board: int, end_step: int, assembled: int, success: bool, episode_steps: int) -> None:
        """Keep native terminal values before the following env.step reuses them."""
        start = self.episode_starts[board]
        self.episodes.append(
            {
                "board_id": board,
                "episode_id": self.episode_numbers[board],
                "completion_phase": self.phase,
                "start_control_step": start,
                "end_control_step": end_step,
                "warmup_control_steps": max(0, min(end_step, self.warmup_steps) - start),
                "measured_control_steps": max(0, end_step - max(start, self.warmup_steps)),
                "assembled_parts": assembled,
                "complete_board": success,
                "native_episode_steps": episode_steps,
            }
        )
        self.episode_numbers[board] += 1
        self.episode_starts[board] = end_step

    def step_returned(self) -> None:
        """Record that env.step, including its automatic resets, returned."""
        self.returned_steps[self.phase] += 1

    def finish_step(self) -> None:
        """Finish the phase after its final native episode bookkeeping returns."""
        phase = self.phase
        target = self.warmup_steps if phase == "warmup" else self.measured_steps
        if self.returned_steps[phase] == target:
            self.complete_device()
            now = self.clock()
            self.completed_steps[phase] = self.returned_steps[phase]
            self.elapsed[phase] = now - self.phase_started
            self.phase_started = now
            self.phase = "measurement" if phase == "warmup" else "complete"

    def report(self, *, error: BaseException | None = None) -> dict:
        """Preserve a completed interval independently of a later invocation failure."""
        measurement_complete = self.finished
        successful = measurement_complete and error is None
        end_step = sum(self.returned_steps.values())
        elapsed = self.elapsed["measurement"]
        report = {
            "schema": "nist-native-throughput-v1",
            "invocation_id": self.invocation_id,
            "status": "complete" if successful else "failed" if error is not None else "incomplete",
            "measurement_status": "complete" if measurement_complete else "incomplete",
            "boards": self.boards,
            "requested_warmup_control_steps": self.warmup_steps,
            "requested_measured_control_steps": self.measured_steps,
            "measured_global_control_step_interval": {
                "start_exclusive": self.warmup_steps,
                "end_inclusive": self.warmup_steps + self.returned_steps["measurement"],
            },
            "returned_control_steps": dict(self.returned_steps),
            "gpu_completed_control_steps": dict(self.completed_steps),
            "elapsed_seconds": dict(self.elapsed),
            "configuration": self.configuration,
            "timing_scope": (
                "policy inference, finite-action check, env.step including automatic resets, "
                "native episode bookkeeping and stdout, and phase accounting"
            ),
            "startup_scope": (
                "main entry through policy loading, environment construction, initial reset, "
                "measurement metadata preparation, and device completion; Python imports excluded"
            ),
            "device_completion_scope": "Torch and Warp selected simulation device at outer phase boundaries",
            "completed_episodes": list(self.episodes),
            "active_episodes_at_cutoff": [
                {
                    "board_id": board,
                    "episode_id": self.episode_numbers[board],
                    "start_control_step": start,
                    "control_steps": end_step - start,
                    "warmup_control_steps": max(0, min(end_step, self.warmup_steps) - start),
                    "measured_control_steps": max(0, end_step - max(start, self.warmup_steps)),
                    "assembly_progress": None,
                    "progress_note": (
                        "Not sampled; terminal extras describe the previous episode "
                        "when a reset occurred on the final step."
                    ),
                }
                for board, start in enumerate(self.episode_starts)
            ],
        }
        if measurement_complete:
            if elapsed is None or elapsed <= 0:
                raise ValueError("Measured elapsed time must be positive to compute a rate.")
            report["environment_steps"] = self.boards * self.completed_steps["measurement"]
            report["environment_steps_per_second"] = report["environment_steps"] / elapsed
        if error is not None:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        return report


def write_report(path: str, report: dict) -> None:
    """Publish one complete JSON document after timing has stopped."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
