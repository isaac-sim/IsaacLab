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
    """Base task specification for deferred runtime construction.

    All Pink IK task configs inherit from this class.  The :attr:`class_type`
    attribute is resolved at runtime to instantiate the concrete task object.
    """

    class_type: str | type | None = None
    """Task builder as ``"module.path:callable"`` or callable object."""


@configclass
class FrameTaskCfg(PinkIKTaskCfg):
    """Configuration for a :class:`~isaaclab.controllers.pink_ik.pink_tasks.FrameTask`.

    Tracks a desired end-effector pose expressed in the world frame.
    """

    frame: Any = MISSING
    """Name of the robot frame to control (e.g. end-effector link)."""

    position_cost: Any = MISSING
    """Cost weight(s) for position error — scalar or 3-element sequence."""

    orientation_cost: Any = MISSING
    """Cost weight(s) for orientation error — scalar or 3-element sequence."""

    lm_damping: float = 0.0
    """Levenberg-Marquardt damping for numerical stability."""

    gain: float = 1.0
    """Task gain that scales the overall task contribution."""

    class_type: str | type | None = "{DIR}.pink_tasks:FrameTask"
    """Default builder pointing to :class:`FrameTask`."""


@configclass
class DampingTaskCfg(PinkIKTaskCfg):
    """Configuration for a :class:`~isaaclab.controllers.pink_ik.pink_tasks.DampingTask`.

    Adds joint-velocity damping to the IK problem for numerical stability.
    """

    cost: Any = MISSING
    """Scalar cost weight penalising joint velocities."""

    class_type: str | type | None = "{DIR}.pink_tasks:DampingTask"
    """Default builder pointing to :class:`DampingTask`."""


@configclass
class LocalFrameTaskCfg(PinkIKTaskCfg):
    """Configuration for a :class:`~isaaclab.controllers.pink_ik.pink_tasks.LocalFrameTask`.

    Tracks a desired pose expressed relative to a specified base-link frame
    rather than the world frame.
    """

    frame: Any = MISSING
    """Name of the robot frame to control (e.g. end-effector link)."""

    base_link_frame_name: Any = MISSING
    """Reference frame for computing transforms and errors."""

    position_cost: Any = MISSING
    """Cost weight(s) for position error — scalar or 3-element sequence."""

    orientation_cost: Any = MISSING
    """Cost weight(s) for orientation error — scalar or 3-element sequence."""

    lm_damping: float = 0.0
    """Levenberg-Marquardt damping for numerical stability."""

    gain: float = 1.0
    """Task gain that scales the overall task contribution."""

    class_type: str | type | None = "{DIR}.pink_tasks:LocalFrameTask"
    """Default builder pointing to :class:`LocalFrameTask`."""


@configclass
class NullSpacePostureTaskCfg(PinkIKTaskCfg):
    """Configuration for a :class:`~isaaclab.controllers.pink_ik.null_space_posture_task.NullSpacePostureTask`.

    Regularises the IK solution toward a preferred joint posture in the
    null-space of the primary tasks.
    """

    cost: Any = MISSING
    """Scalar cost weight for the posture regularisation term."""

    lm_damping: float = 0.0
    """Levenberg-Marquardt damping for numerical stability."""

    gain: float = 1.0
    """Task gain that scales the overall task contribution."""

    controlled_frames: list[str] = field(default_factory=list)
    """Robot frames whose joints are included in the posture task."""

    controlled_joints: list[str] = field(default_factory=list)
    """Explicit list of joint names to include (overrides frame-based selection when non-empty)."""

    class_type: str | type | None = "{DIR}.null_space_posture_task:NullSpacePostureTask"
    """Default builder pointing to :class:`NullSpacePostureTask`."""
