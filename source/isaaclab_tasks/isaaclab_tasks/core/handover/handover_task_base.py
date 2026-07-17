# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared task-defining parameters for the handover family.

The Direct environment and the manager counterpart inherit the same
shared structural definitions so the two variants cannot drift apart. Every
scalar task parameter reads as a flat ``cfg.<name>`` field on the env
configuration; non-scalar items (joint/body-name lists, the goal offset, and
marker templates) stay module-level constants.
"""

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

from isaaclab_tasks.core.reorient.reorient_task_base import (
    SHADOW_ACTUATED_JOINT_NAMES as ACTUATED_JOINT_NAMES,
)
from isaaclab_tasks.core.reorient.reorient_task_base import (
    SHADOW_FINGERTIP_BODY_NAMES as FINGERTIP_BODY_NAMES,
)
from isaaclab_tasks.utils.hydra import preset

__all__ = [
    "ACTUATED_JOINT_NAMES",
    "ACTUATED_JOINT_NAMES_NEWTON",
    "ACTUATED_JOINT_NAMES_PRESET",
    "FINGERTIP_BODY_NAMES",
    "GOAL_MARKER_CFG",
    "GOAL_POSITION_OFFSET",
    "OBJECT_RADIUS",
]


ACTUATED_JOINT_NAMES_NEWTON: list[str] = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ4",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_MFJ4",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_RFJ4",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_LFJ5",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
]
"""Actuated joint names on the production Newton Shadow Hand asset (+1 finger renumbering)."""


ACTUATED_JOINT_NAMES_PRESET = preset(
    physx=ACTUATED_JOINT_NAMES,
    newton_mjwarp=ACTUATED_JOINT_NAMES_NEWTON,
    ovphysx=ACTUATED_JOINT_NAMES,
    default=ACTUATED_JOINT_NAMES,
)
"""Per-backend actuated joint names, resolved by the physics preset key."""


OBJECT_RADIUS: float = 0.0335
"""Hand-over object sphere radius [m], also used for the goal marker."""

GOAL_POSITION_OFFSET: tuple[float, float, float] = (0.0, -0.25, 0.0)
"""Goal-position offset from the object's default position [m].

The Direct environment and the manager command derive the same fixed goal
point from the object's default root position plus this offset.
"""

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=OBJECT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
        ),
    },
)
"""Goal-marker template shared by the Direct environment and the manager command term.

Consumers relying on a different prim path use ``replace``
on this template; configclass deep-copies defaults, so sharing the instance is safe.
"""
