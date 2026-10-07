# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Timing of scripted arm travel, independent of grasp and settling time."""


def pause_speed(speed: float) -> float:
    """Return the speed-up of pauses for an arm speed multiplier: none up to the default 2, then proportional."""
    return max(1.0, speed / 2.0)


def _intervals(
    mode: str, speed: float, pauses: float, closing: float | None = None
) -> list[tuple[float, float, float]]:
    """Return the accelerated (start, end, multiplier) intervals of a scripted mode's timeline [s]."""
    intervals = [(1.0, 4.0, speed)]
    if mode in ("pick", "place"):
        intervals.append((9.0, 12.0, speed))
    if mode == "place":
        intervals.extend(((13.0, 19.0, speed), (20.0, 23.0, speed), (26.0, 29.0, speed)))
        # Settling, holds, waypoint pauses, gripper opening and the final wait.
        intervals.extend(
            (start, end, pauses)
            for start, end in ((0.0, 1.0), (8.0, 9.0), (12.0, 13.0), (19.0, 20.0), (23.0, 26.0), (29.0, 32.0))
        )
        # Closing the gripper (4 to 8 s) speeds up at most twofold, which keeps grasps gentle.
        intervals.append((4.0, 8.0, min(pauses, 2.0) if closing is None else closing))
    return sorted(intervals)


def scripted_motion_time(
    elapsed: float, mode: str, speed: float, pauses: float = 1.0, closing: float | None = None
) -> float:
    """Map elapsed simulation time [s] to script time [s], accelerating arm travel and, in place mode, pauses.

    Args:
        elapsed: Time since reset [s].
        mode: Scripted pick, place or squash mode.
        speed: Positive arm travel speed multiplier.
        pauses: Speed multiplier of the place mode's pauses; 1 keeps them.
        closing: Speed multiplier of the place mode's gripper closing; by default, the pauses' up to 2.
    """
    saved = 0.0
    for start, end, multiplier in _intervals(mode, speed, pauses, closing):
        actual_start = start - saved
        if elapsed < actual_start:
            break
        duration = (end - start) / multiplier
        if elapsed < actual_start + duration:
            return start + (elapsed - actual_start) * multiplier
        saved += end - start - duration
    return elapsed + saved


def scripted_duration(
    script_time: float, mode: str, speed: float, pauses: float = 1.0, closing: float | None = None
) -> float:
    """Return the elapsed simulation time [s] at which a scripted mode reaches ``script_time`` [s]."""
    saved = sum(
        (min(end, script_time) - start) * (1.0 - 1.0 / multiplier)
        for start, end, multiplier in _intervals(mode, speed, pauses, closing)
        if start < script_time
    )
    return script_time - saved
