# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .frame_transformer import FrameTransformer


@configclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""


@configclass
class FrameTransformerCfg(SensorBaseCfg):
    """Configuration for the frame transformer sensor."""

    @configclass
    class FrameCfg:
        """Information specific to a coordinate frame."""

        prim_path: str = MISSING
        """The prim path corresponding to the parent rigid body.

        This prim should be part of the same articulation as :attr:`FrameTransformerCfg.prim_path`.
        """
        name: str | None = None
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """

        offset: OffsetCfg = OffsetCfg()
        """The pose offset from the parent prim frame."""

    class_type: type = FrameTransformer

    prim_path: str = MISSING
    """The prim path of the body to transform from (source frame)."""

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

    Note:
        This attribute is only used when debug visualization is enabled.
    """
