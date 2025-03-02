# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .patterns.patterns_cfg import PatternBaseCfg
from .ray_caster import RayCaster


@configclass
class RayCasterCfg(SensorBaseCfg):
    """Configuration for the ray-cast sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type = RayCaster

    mesh_prim_paths: list[str] = MISSING
    """The list of mesh primitive paths to ray cast against.

    Note:
        Currently, only a single static mesh is supported. We are working on supporting multiple
        static meshes and dynamic meshes.
    """

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    attach_yaw_only: bool = MISSING
    """Whether the rays' starting positions and directions only track the yaw orientation.

    This is useful for ray-casting height maps, where only yaw rotation is needed.
    """

    pattern_cfg: PatternBaseCfg = MISSING
    """The pattern that defines the local ray starting positions and directions."""

    max_distance: float = 1e6
    """Maximum distance (in meters) from the sensor to ray cast to. Defaults to 1e6."""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """The range of drift (in meters) to add to the ray starting positions (xyz). Defaults to (0.0, 0.0).

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
    """The configuration object for the visualization markers. Defaults to RAY_CASTER_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
