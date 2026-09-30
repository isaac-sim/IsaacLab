# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Timing of scripted arm travel, independent of grasp and settling time."""


def scripted_motion_time(elapsed: float, mode: str, speed: float) -> float:
    """Map elapsed simulation time [s] to script time [s], accelerating only arm travel.

    Args:
        elapsed: Time since reset [s].
        mode: Scripted pick, place or squash mode.
        speed: Positive arm travel speed multiplier.
    """
    intervals = [(1.0, 4.0)]
    if mode in ("pick", "place"):
        intervals.append((9.0, 12.0))
    if mode == "place":
        intervals.extend(((13.0, 19.0), (20.0, 23.0), (26.0, 29.0)))
    saved = 0.0
    for start, end in intervals:
        actual_start = start - saved
        if elapsed < actual_start:
            break
        duration = (end - start) / speed
        if elapsed < actual_start + duration:
            return start + (elapsed - actual_start) * speed
        saved += end - start - duration
    return elapsed + saved
