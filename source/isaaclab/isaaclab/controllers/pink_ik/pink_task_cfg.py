# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration objects for Pink IK."""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

from isaaclab.utils import configclass


@configclass
class PinkIKTaskCfg:
    """Task specification for deferred runtime construction."""

    class_type: str | type | None = None
    """Task builder as ``"module.path:callable"`` or callable object."""


@configclass
class FrameTaskCfg(PinkIKTaskCfg):
    """Configuration wrapper for ``pink.tasks.FrameTask``."""

    frame: Any = MISSING
    position_cost: Any = MISSING
    orientation_cost: Any = MISSING
    lm_damping: float = 0.0
    gain: float = 1.0
    class_type: str | type | None = "isaaclab.controllers.pink_ik.pink_tasks:FrameTask"


@configclass
class DampingTaskCfg(PinkIKTaskCfg):
    """Configuration wrapper for ``pink.tasks.DampingTask``."""

    cost: Any = MISSING
    class_type: str | type | None = "isaaclab.controllers.pink_ik.pink_tasks:DampingTask"


@configclass
class LocalFrameTaskCfg(PinkIKTaskCfg):
    """Configuration wrapper for ``LocalFrameTask``."""

    frame: Any = MISSING
    base_link_frame_name: Any = MISSING
    position_cost: Any = MISSING
    orientation_cost: Any = MISSING
    lm_damping: float = 0.0
    gain: float = 1.0
    class_type: str | type | None = "isaaclab.controllers.pink_ik.pink_tasks:LocalFrameTask"


@configclass
class NullSpacePostureTaskCfg(PinkIKTaskCfg):
    """Configuration wrapper for ``NullSpacePostureTask``."""

    cost: Any = MISSING
    lm_damping: float = 0.0
    gain: float = 1.0
    controlled_frames: list[str] = field(default_factory=list)
    controlled_joints: list[str] = field(default_factory=list)
    class_type: str | type | None = "isaaclab.controllers.pink_ik.null_space_posture_task:NullSpacePostureTask"
