# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly profile configuration classes.

An assembly profile describes the full geometric path from assembled (fraction=0)
to disassembled (fraction=1) as a sequence of contiguous segments.  Two segment
types are provided:

* :class:`EndPointsSegmentCfg` — defined by explicit start/end poses and optional
  extra revolutions.
* :class:`IncrementalSegmentCfg` — defined by start pose, travel distance, and
  a screw-pitch ratio (m/rad).

Start-sampler configs (:class:`UniformYawCfg`, :class:`DiscreteYawCfg`) define
noise applied on top of each segment's start pose.  ``None`` means no noise.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.assembly_keypoints import Offset
from isaaclab_tasks.contrib.nist.assembly_profile import (
    AssemblyProfile,
    DiscreteYaw,
    EndPointsSegment,
    IncrementalSegment,
    UniformPoseNoise,
    UniformYaw,
)

# ---------------------------------------------------------------------------
# Start-sampler Cfg classes
# ---------------------------------------------------------------------------


@configclass
class UniformYawCfg:
    """Uniformly random yaw in ``[-pi, pi]``, no position noise."""

    class_type: type = UniformYaw
    """Class of the sampler implementation."""


@configclass
class DiscreteYawCfg:
    """Randomly chosen from a discrete set of yaw angles [rad], no position noise."""

    class_type: type = DiscreteYaw
    """Class of the sampler implementation."""

    yaws: list[float] | None = None
    """Yaw angles [rad] to sample from."""


@configclass
class UniformPoseNoiseCfg:
    """Uniform noise over user-defined position [m] and euler-angle [rad] ranges."""

    class_type: type = UniformPoseNoise
    """Class of the sampler implementation."""

    x: tuple[float, float] = (0.0, 0.0)
    """Position noise range along x [m]."""

    y: tuple[float, float] = (0.0, 0.0)
    """Position noise range along y [m]."""

    z: tuple[float, float] = (0.0, 0.0)
    """Position noise range along z [m]."""

    roll: tuple[float, float] = (0.0, 0.0)
    """Roll noise range [rad]."""

    pitch: tuple[float, float] = (0.0, 0.0)
    """Pitch noise range [rad]."""

    yaw: tuple[float, float] = (0.0, 0.0)
    """Yaw noise range [rad]."""


# ---------------------------------------------------------------------------
# Segment and profile Cfg classes
# ---------------------------------------------------------------------------


@configclass
class EndPointsSegmentCfg:
    """Segment defined by explicit start and end poses.

    See :class:`EndPointsSegment` for the runtime implementation.
    """

    class_type: type = EndPointsSegment
    """Class of the segment implementation."""

    fraction: tuple[float, float] = (0.0, 1.0)
    """Fraction range ``(lo, hi)`` this segment covers. ``0`` is assembled."""

    start_sampler: UniformYawCfg | DiscreteYawCfg | UniformPoseNoiseCfg | None = None
    """Noise config applied on top of the interpolated pose. ``None`` means no noise."""

    start_pose: Offset = Offset()
    """Offset at ``fraction[0]`` (assembled end) relative to the fixed asset."""

    end_pose: Offset = Offset()
    """Offset at ``fraction[1]`` (disassembled end) relative to the fixed asset."""

    revolutions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Extra full turns ``(roll, pitch, yaw)`` between the two endpoints."""


@configclass
class IncrementalSegmentCfg:
    """Segment defined by start pose, travel distance, and rotation ratio.

    See :class:`IncrementalSegment` for the runtime implementation.
    """

    class_type: type = IncrementalSegment
    """Class of the segment implementation."""

    fraction: tuple[float, float] = (0.0, 1.0)
    """Fraction range ``(lo, hi)`` this segment covers. ``0`` is assembled."""

    start_sampler: UniformYawCfg | DiscreteYawCfg | UniformPoseNoiseCfg | None = None
    """Noise config applied on top of the interpolated pose. ``None`` means no noise."""

    start_pose: Offset = Offset()
    """Offset at ``fraction[0]`` (assembled end) relative to the fixed asset."""

    distance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear travel vector from start to end [m]."""

    ratio: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Screw pitch per axis: meters of travel per radian [m/rad]. Zero means no rotation."""


@configclass
class AssemblyProfileCfg:
    """Complete assembly path as a list of contiguous segment configs.

    ``fraction=0`` means fully assembled; increasing fraction moves toward the
    disassembled state. See :class:`AssemblyProfile` for the runtime implementation.
    """

    class_type: type = AssemblyProfile
    """Class of the profile implementation."""

    segments: list[EndPointsSegmentCfg | IncrementalSegmentCfg] | None = None
    """Ordered list of segment configs covering the full fraction range."""
