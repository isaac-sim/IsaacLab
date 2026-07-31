# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .frame_transformer import FrameTransformer


@configclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""


@configclass
class FrameTransformerCfg(SensorBaseCfg):
    """Configuration for the frame transformer sensor."""

    @configclass
    class FrameCfg:
        """Information specific to a coordinate frame."""

        prim_path: str | None = None
        """The prim path corresponding to a rigid body.

        .. deprecated::
            Use :attr:`prim_path_regex` instead. For backwards compatibility, this path
            recursively searches its descendants for rigid bodies.
        """

        prim_path_regex: str | None = None
        """The prim path regex corresponding to a rigid body.

        Only prims directly matched by the expression are selected. For example,
        ``/Robot/.*`` matches rigid-body children of ``/Robot``.

        This means that if the source :attr:`FrameTransformerCfg.prim_path_regex` is "/Robot/base",
        and the target :attr:`FrameTransformerCfg.FrameCfg.prim_path_regex` is "/Robot/.*", then
        the frame transformer will track the poses of matching rigid-body children under "/Robot",
        including "/Robot/base" (even though this will result in an identity pose w.r.t.
        the source frame).
        """

        name: str | None = None
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """

        offset: OffsetCfg = OffsetCfg()
        """The pose offset from the parent prim frame."""

    class_type: type[FrameTransformer] | str = "{DIR}.frame_transformer:FrameTransformer"

    prim_path: str | None = None
    """The prim path of the body to transform from (source frame).

    .. deprecated::
        Use :attr:`prim_path_regex` instead. For backwards compatibility, this path
        recursively searches its descendants for a rigid body.
    """

    prim_path_regex: str | None = None
    """The prim path regex of the body to transform from (source frame).

    Only a rigid body directly matched by the expression is selected.
    """

    source_frame_offset: OffsetCfg = OffsetCfg()
    """The pose offset from the source prim frame."""

    target_frames: list[FrameCfg] = MISSING
    """A list of the target frames.

    This allows a single FrameTransformer to handle multiple target prims. For example, in a quadruped,
    we can use a single FrameTransformer to track each foot's position and orientation in the body
    frame using four frame offsets.
    """

    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer")
    """The configuration object for the visualization markers. Defaults to FRAME_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
