# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-family identity shared by the Direct and manager-based reorientation tasks.

Geometry constants only — task tunables live inline in the workflow
configuration files, and robot identity lives in the per-robot ``*_common``
modules.
"""

GOAL_MARKER_POSITION: tuple[float, float, float] = (-0.2, -0.45, 0.68)
"""Fixed goal-marker display position [m], environment frame (state-based tasks)."""

CAMERA_GOAL_MARKER_POSITION: tuple[float, float, float] = (-0.2, 0.1, 0.6)
"""Goal-marker display position [m] for the camera tasks.

Deviates from :data:`GOAL_MARKER_POSITION` so the goal cube sits inside the
tiled camera's frustum.
"""

IN_HAND_POS_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.04)
"""Offset from the object's default position to the in-hand goal anchor [m].

Defines the Direct/manager goal-position parity: the Direct environments and
the manager command terms derive the same in-hand target point from the
object's default root position plus this offset.
"""

CAMERA_PLAY_NUM_ENVS: int = 64
"""Camera-task environment count for checkpoint playback."""
