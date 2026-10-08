# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Timing of the scripted pick-and-place of one berry: arm travel speeds up, the grasp keeps its gentle pace.

The script is written at the original pace, in script time [s]: approach (1-4 s), close (4-8 s), lift (9-12 s), carry
(13-19 s), lower (20-23 s), open (24-26 s) and retreat (26-29 s), with pauses between them. A speed multiplier
shortens arm travel and, above 2, the pauses; :func:`script_time` maps elapsed time to script time.
"""


def pause_speedup(speed: float) -> float:
    """Return the speed-up of pauses for an arm speed multiplier: none up to the default 2, then proportional."""
    return max(1.0, speed / 2.0)


def _intervals(speed: float, pauses: float, closing: float | None = None) -> list[tuple[float, float, float]]:
    """Return the accelerated (start, end, multiplier) intervals of the script [s]."""
    # Arm travel.
    intervals = [
        (start, end, speed) for start, end in ((1.0, 4.0), (9.0, 12.0), (13.0, 19.0), (20.0, 23.0), (26.0, 29.0))
    ]
    # Settling, holds, waypoint pauses, gripper opening and the final wait.
    intervals.extend(
        (start, end, pauses)
        for start, end in ((0.0, 1.0), (8.0, 9.0), (12.0, 13.0), (19.0, 20.0), (23.0, 26.0), (29.0, 32.0))
    )
    # Closing the gripper (4 to 8 s) speeds up at most twofold, which keeps grasps gentle.
    intervals.append((4.0, 8.0, min(pauses, 2.0) if closing is None else closing))
    return sorted(intervals)


def script_time(elapsed: float, speed: float, pauses: float = 1.0, closing: float | None = None) -> float:
    """Map elapsed simulation time [s] to script time [s], accelerating arm travel and pauses.

    Args:
        elapsed: Time since the script started [s].
        speed: Positive arm travel speed multiplier.
        pauses: Speed multiplier of the pauses; 1 keeps them.
        closing: Speed multiplier of the gripper closing; by default, the pauses' up to 2.
    """
    saved = 0.0
    for start, end, multiplier in _intervals(speed, pauses, closing):
        actual_start = start - saved
        if elapsed < actual_start:
            break
        duration = (end - start) / multiplier
        if elapsed < actual_start + duration:
            return start + (elapsed - actual_start) * multiplier
        saved += end - start - duration
    return elapsed + saved


def elapsed_time(script_time: float, speed: float, pauses: float = 1.0, closing: float | None = None) -> float:
    """Return the elapsed simulation time [s] at which the script reaches ``script_time`` [s]."""
    saved = sum(
        (min(end, script_time) - start) * (1.0 - 1.0 / multiplier)
        for start, end, multiplier in _intervals(speed, pauses, closing)
        if start < script_time
    )
    return script_time - saved
