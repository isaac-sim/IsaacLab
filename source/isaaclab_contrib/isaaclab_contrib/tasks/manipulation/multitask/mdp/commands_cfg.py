# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for group-aware pose command term."""

from __future__ import annotations

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import MISSING, configclass


@configclass
class PoseCommandRanges:
    """Uniform sampling ranges for a pose command [m, rad]."""

    pos_x: tuple[float, float] = (0.0, 0.0)
    """Min/max for X position [m]."""
    pos_y: tuple[float, float] = (0.0, 0.0)
    """Min/max for Y position [m]."""
    pos_z: tuple[float, float] = (0.0, 0.0)
    """Min/max for Z position [m]."""
    roll: tuple[float, float] = (0.0, 0.0)
    """Min/max for roll angle [rad]."""
    pitch: tuple[float, float] = (0.0, 0.0)
    """Min/max for pitch angle [rad]."""
    yaw: tuple[float, float] = (0.0, 0.0)
    """Min/max for yaw angle [rad]."""


@configclass
class PoseCommandCfg(CommandTermCfg):
    """Configuration for :class:`PoseCommand`.

    A group-aware pose command generator that samples targets for a
    single robot asset scoped to specific clone groups via
    :class:`SceneEntityCfg`.
    """

    class_type: type | str = "{DIR}.commands:PoseCommand"

    asset_cfg: SceneEntityCfg = MISSING
    """Robot articulation with ``body_names`` and ``groups`` for scoping."""

    ranges: PoseCommandRanges = MISSING
    """Uniform sampling ranges for the pose command target [m, rad]."""

    make_quat_unique: bool = True
    """Whether to ensure the quaternion has a positive real part."""
