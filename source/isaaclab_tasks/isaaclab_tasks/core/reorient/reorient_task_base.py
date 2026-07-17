# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural definitions shared by the reorientation task family.

Joint/body name lists, marker geometry, and reset-pose offsets consumed by both
the Direct and manager-based variants. Scalar task parameters are defined
per-paradigm in the respective environment configurations, following the
convention of the other core tasks.
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

SHADOW_FINGERTIP_BODY_NAMES: list[str] = [
    "robot0_ffdistal",
    "robot0_mfdistal",
    "robot0_rfdistal",
    "robot0_lfdistal",
    "robot0_thdistal",
]
"""Shadow Hand fingertip body names (identical on every backend asset)."""

ALLEGRO_ACTUATED_JOINT_NAMES: list[str] = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
"""Allegro Hand actuated joint names, in the Direct task's actuation order."""

ALLEGRO_FINGERTIP_BODY_NAMES: list[str] = [
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
]
"""Allegro Hand fingertip body names."""

SHADOW_ACTUATED_JOINT_NAMES: list[str] = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
]
"""Shadow Hand actuated joint names, in the Direct task's actuation order."""
