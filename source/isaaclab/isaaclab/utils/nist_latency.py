# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched env.step latency and native pre-reset board telemetry.

The caller injects Isaac Lab's benchmark.micro.measure_latency so this accounting
module can also be exercised without importing the simulation runtime.
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any


def _statistics(values: Sequence[float]) -> dict:
    """Milliseconds, sample standard deviation, and linear percentiles."""
    ordered = sorted(values)

    def percentile(q: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered) if ordered else None,
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0 if ordered else None,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _summarize(samples: Sequence[dict]) -> dict:
    return {
        "n": len(samples),
        "submission_ms": _statistics([sample["submission_ms"] for sample in samples]),
        "completed_ms": _statistics([sample["completed_ms"] for sample in samples]),
    }


class LatencyMeasurement:
    """Observe individual boards while timing their shared environment step."""

    def __init__(
        self,
        boards: int,
        warmup_steps: int,
        measured_steps: int,
        complete_device: Callable[[], None],
        *,
        measure: Callable[..., Any],
        startup_started: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if boards < 1 or warmup_steps < 0 or measured_steps < 1:
            raise ValueError("Boards and measured steps are positive; warmup steps are nonnegative.")
        self.boards = boards
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.complete_device = complete_device
        self.measure = measure
        self.clock = clock
        self.startup_started = clock() if startup_started is None else startup_started
        self.invocation_id = str(uuid.uuid4())
        self.configuration: dict = {}
        self.phase = "startup"
        self.phase_started: float | None = None
        self.elapsed: dict[str, float | None] = {"startup": None, "warmup": None, "measurement": None}
        self.attempted_steps = {"warmup": 0, "measurement": 0}
        self.returned_steps = {"warmup": 0, "measurement": 0}
        self.completed_steps = {"warmup": 0, "measurement": 0}
        self.telemetry_steps = {"warmup": 0, "measurement": 0}
        self.loop_steps = {"warmup": 0, "measurement": 0}
        self.samples: list[dict] = []
        self.episodes: list[dict] = []
        self.episode_numbers = [0] * boards
        self.episode_starts = [0] * boards
        self.last_progress: list[dict | None] = [None] * boards
        self._pending: dict | None = None

    @property
    def finished(self) -> bool:
        return self.phase == "complete"

    def start(self) -> None:
        """Drain setup before continuing the native policy through warmup."""
        if self.phase != "startup":
            raise RuntimeError("Latency measurement has already started.")
        self.complete_device()
        now = self.clock()
        self.elapsed["startup"] = now - self.startup_started
        self.phase_started = now
        self.phase = "warmup" if self.warmup_steps else "measurement"
        if not self.warmup_steps:
            self.elapsed["warmup"] = 0.0

    def measure_step(self, operation: Callable[[], Any]) -> Any:
        """Return the exact operation result; retain only successfully drained samples."""
        if self.phase not in self.returned_steps or self._pending is not None:
            raise RuntimeError("A new step needs an active phase and the previous step's bookkeeping.")
        phase = self.phase
        self.attempted_steps[phase] += 1
        pending = {
            "phase": phase,
            "global_control_step": sum(self.returned_steps.values()) + 1,
            "phase_control_step": self.returned_steps[phase] + 1,
            "operation_returned": False,
            "timing_completed": False,
            "telemetry_status": "missing",
            "board_episode_ids": list(self.episode_numbers),
        }
        self._pending = pending
        result: Any = None

        def capture_result() -> None:
            nonlocal result
            result = operation()
            pending["operation_returned"] = True
            self.returned_steps[phase] += 1

        if phase == "warmup":
            capture_result()
        else:
            timing = self.measure(capture_result, self.complete_device)
            submission_ms = float(timing.submission_s) * 1000.0
            completed_ms = float(timing.synchronized_s) * 1000.0
            if not pending["operation_returned"]:
                raise RuntimeError("The injected timer did not return from the operation.")
            if not (
                math.isfinite(submission_ms) and math.isfinite(completed_ms) and 0 <= submission_ms <= completed_ms
            ):
                raise ValueError("Latency samples describe finite, ordered host and device completion times.")
            sample = {
                "global_control_step": pending["global_control_step"],
                "measured_control_step": pending["phase_control_step"],
                "submission_ms": submission_ms,
                "completed_ms": completed_ms,
                "telemetry_status": "missing",
                "board_episode_ids": list(pending["board_episode_ids"]),
                "assembled_parts": None,
                "assembly_success": None,
                "episode_steps": None,
                "terminated": None,
                "timed_out": None,
                "reset_board_ids": None,
            }
            self.samples.append(sample)
            self.completed_steps[phase] += 1
            pending["timing_completed"] = True
            pending["sample"] = sample
        return result

    def observe_step(
        self,
        assembled_parts: list[int],
        assembly_success: list[bool],
        episode_steps: list[int],
        terminated: list[bool],
        timed_out: list[bool],
    ) -> None:
        """Attach already-computed native extras after the timed operation.

        Every vector describes the just-finished, pre-reset episode. A reset
        increments the board's episode ID for the following operation.
        """
        pending = self._pending
        if pending is None or not pending["operation_returned"] or pending["telemetry_status"] != "missing":
            raise RuntimeError("Board telemetry belongs to one returned, not-yet-observed step.")
        values = (assembled_parts, assembly_success, episode_steps, terminated, timed_out)
        if any(len(value) != self.boards for value in values):
            raise ValueError("Each native telemetry vector has one entry per board.")
        # Prepare every vector before changing episode state, so a malformed
        # snapshot cannot leave a partly advanced collection of episode IDs.
        snapshot = {
            "assembled_parts": [int(value) for value in assembled_parts],
            "assembly_success": [bool(value) for value in assembly_success],
            "episode_steps": [int(value) for value in episode_steps],
            "terminated": [bool(value) for value in terminated],
            "timed_out": [bool(value) for value in timed_out],
        }
        reset_boards = [
            board for board in range(self.boards) if snapshot["terminated"][board] or snapshot["timed_out"][board]
        ]
        end = pending["global_control_step"]
        for board in range(self.boards):
            if board in reset_boards:
                start = self.episode_starts[board]
                self.episodes.append(
                    {
                        "board_id": board,
                        "episode_id": self.episode_numbers[board],
                        "completion_phase": pending["phase"],
                        "start_control_step": start,
                        "end_control_step": end,
                        "warmup_control_steps": max(0, min(end, self.warmup_steps) - start),
                        "measured_control_steps": max(0, end - max(start, self.warmup_steps)),
                        "assembled_parts": snapshot["assembled_parts"][board],
                        "complete_board": snapshot["assembly_success"][board],
                        "native_episode_steps": snapshot["episode_steps"][board],
                        "terminated": snapshot["terminated"][board],
                        "timed_out": snapshot["timed_out"][board],
                    }
                )
                self.episode_numbers[board] += 1
                self.episode_starts[board] = end
                self.last_progress[board] = None
            else:
                self.last_progress[board] = {
                    "global_control_step": end,
                    "assembled_parts": snapshot["assembled_parts"][board],
                    "native_episode_steps": snapshot["episode_steps"][board],
                }
        snapshot["reset_board_ids"] = reset_boards
        snapshot["telemetry_status"] = "complete"
        if "sample" in pending:
            pending["sample"].update(snapshot)
        pending["telemetry_status"] = "complete"
        self.telemetry_steps[pending["phase"]] += 1

    def finish_step(self) -> None:
        """Finish native bookkeeping; only warmup needs an additional phase drain."""
        pending = self._pending
        if pending is None or pending["telemetry_status"] != "complete":
            raise RuntimeError("Finishing a step follows its native board telemetry.")
        phase = pending["phase"]
        self.loop_steps[phase] += 1
        target = self.warmup_steps if phase == "warmup" else self.measured_steps
        if self.loop_steps[phase] == target:
            if phase == "warmup":
                self.complete_device()
                self.completed_steps[phase] = self.returned_steps[phase]
            now = self.clock()
            self.elapsed[phase] = now - self.phase_started
            self.phase_started = now
            self.phase = "measurement" if phase == "warmup" else "complete"
        self._pending = None

    def report(self, *, error: BaseException | None = None) -> dict:
        """Retain valid timing and telemetry independently of invocation health."""
        timing_complete = len(self.samples) == self.measured_steps
        telemetry_complete = self.telemetry_steps["measurement"] == self.measured_steps
        measurement_complete = timing_complete and telemetry_complete
        known = [sample for sample in self.samples if sample["telemetry_status"] == "complete"]
        reset = [sample for sample in known if sample["reset_board_ids"]]
        nonreset = [sample for sample in known if not sample["reset_board_ids"]]
        unknown = [sample for sample in self.samples if sample["telemetry_status"] != "complete"]
        end = sum(self.returned_steps.values())
        observed_end = sum(self.telemetry_steps.values())
        report = {
            "schema": "nist-native-latency-v1",
            "invocation_id": self.invocation_id,
            "status": "failed" if error is not None else "complete" if self.finished else "incomplete",
            "measurement_status": "complete" if measurement_complete else "incomplete",
            "timing_status": "complete" if timing_complete else "incomplete",
            "telemetry_status": "complete" if telemetry_complete else "incomplete",
            "boards": self.boards,
            "requested_warmup_control_steps": self.warmup_steps,
            "requested_measured_control_steps": self.measured_steps,
            "measured_global_control_step_interval": {
                "start_exclusive": self.warmup_steps,
                "last_timed_step_inclusive": self.samples[-1]["global_control_step"] if self.samples else None,
            },
            "attempted_control_steps": dict(self.attempted_steps),
            "returned_control_steps": dict(self.returned_steps),
            "gpu_completed_control_steps": dict(self.completed_steps),
            "telemetry_control_steps": dict(self.telemetry_steps),
            "completed_loop_steps": dict(self.loop_steps),
            "elapsed_seconds": dict(self.elapsed),
            "configuration": self.configuration,
            "timing_scope": (
                "one batched env.step, including automatic resets and final observations, "
                "through selected-device completion, with timer/result-wrapper overhead; "
                "policy inference, finite-action check, telemetry extraction, native episode stdout "
                "and subsequent episode/statistics bookkeeping excluded"
            ),
            "device_completion_scope": (
                "injected selected-device completion before and after each measured env.step; "
                "the pre-boundary drain excludes preceding policy and other queued work"
            ),
            "startup_scope": (
                "main entry through policy loading, environment construction, initial reset, "
                "measurement metadata preparation and device completion; Python imports excluded"
            ),
            "phase_wall_time_scope": (
                "diagnostic phase wall time also includes policy, telemetry, stdout and synchronization; "
                "it is not the native throughput measurement"
            ),
            "observer_effect": (
                "Per-step synchronization and telemetry extraction change execution pacing. "
                "Telemetry is also observed during policy warmup; timing excludes its extraction cost, "
                "but that does not remove its possible effect on subsequent execution."
            ),
            "sample_scope": "One sample is the shared batch call; no individual-board GPU elapsed time is attributed.",
            "telemetry_scope": (
                "Native pre-reset extras and termination/timeout flags for each board; "
                "assembly count is instantaneous and can decrease. Post-reset progress is not inferred."
            ),
            "statistics_method": {
                "unit": "ms",
                "percentile": "linear",
                "std": "sample (ddof=1), zero for one sample",
                "empty_subset": "n=0 and null statistics",
                "sample_note": (
                    "Observed sample counts accompany every subset; small samples give limited tail information."
                ),
            },
            "statistics": {
                "all": _summarize(self.samples),
                "reset": _summarize(reset),
                "nonreset": _summarize(nonreset),
                "unknown_telemetry": _summarize(unknown),
            },
            "samples": [dict(sample) for sample in self.samples],
            "completed_episodes": list(self.episodes),
            "episode_context_cutoff_control_step": observed_end,
            "unobserved_returned_control_steps": end - observed_end,
            "active_episodes_at_cutoff": [
                {
                    "board_id": board,
                    "episode_id": self.episode_numbers[board],
                    "episode_identity_at_cutoff": "known" if end == observed_end else "unknown_after_last_observation",
                    "start_control_step": self.episode_starts[board],
                    "returned_control_steps_since_start": end - self.episode_starts[board]
                    if end == observed_end
                    else None,
                    "last_observed_progress": self.last_progress[board],
                }
                for board in range(self.boards)
            ],
            "unfinished_step": (
                {key: value for key, value in self._pending.items() if key != "sample"}
                if self._pending is not None
                else None
            ),
        }
        if error is not None:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        return report
