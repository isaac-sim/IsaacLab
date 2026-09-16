# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import config_field, replace_config

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .frame_transformer import FrameTransformer


@dataclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""

    pos: tuple[float, float, float] = config_field((0.0, 0.0, 0.0))
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = config_field((0.0, 0.0, 0.0, 1.0))
    """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""


@dataclass
class FrameTransformerCfg(SensorBaseCfg):
    """Configuration for the frame transformer sensor."""

    @dataclass
    class FrameCfg:
        """Information specific to a coordinate frame."""

        prim_path: str = config_field(MISSING)
        """The prim path corresponding to a rigid body.

        This can be a regex pattern to match multiple prims. For example, "/Robot/.*"
        will match all prims under "/Robot".

        This means that if the source :attr:`FrameTransformerCfg.prim_path` is "/Robot/base",
        and the target :attr:`FrameTransformerCfg.FrameCfg.prim_path` is "/Robot/.*", then
        the frame transformer will track the poses of all the prims under "/Robot",
        including "/Robot/base" (even though this will result in an identity pose w.r.t.
        the source frame).
        """

        name: str | None = config_field(None)
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """

        offset: OffsetCfg = config_field(OffsetCfg())
        """The pose offset from the parent prim frame."""

    class_type: type[FrameTransformer] | str = config_field("{DIR}.frame_transformer:FrameTransformer")

    prim_path: str = config_field(MISSING)
    """The prim path of the body to transform from (source frame)."""

    source_frame_offset: OffsetCfg = config_field(OffsetCfg())
    """The pose offset from the source prim frame."""

    target_frames: list[FrameCfg] = config_field(MISSING)
    """A list of the target frames.

    This allows a single FrameTransformer to handle multiple target prims. For example, in a quadruped,
    we can use a single FrameTransformer to track each foot's position and orientation in the body
    frame using four frame offsets.
    """

    visualizer_cfg: VisualizationMarkersCfg = config_field(
        replace_config(FRAME_MARKER_CFG, prim_path="/Visuals/FrameTransformer")
    )
    """The configuration object for the visualization markers. Defaults to FRAME_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
