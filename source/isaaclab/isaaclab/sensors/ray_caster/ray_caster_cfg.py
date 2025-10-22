# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from dataclasses import MISSING
from typing import Literal

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
    """The list of mesh primitive paths to ray cast against."""

    dynamic_mesh_prim_paths: list[str] = []
    """The list of dynamic mesh primitive paths that move during simulation.

    These meshes will have their transforms updated before each raycast operation.
    The paths should point to meshes that are part of articulated or moving rigid bodies.
    Defaults to an empty list (all meshes are static).
    """

    dynamic_mesh_update_decimation: int = 1
    """Update dynamic meshes every N sensor updates (decimation factor).

    Setting this to values > 1 can improve performance at the cost of slightly stale mesh positions.
    For example, if set to 2, dynamic meshes are updated every other sensor update.
    Defaults to 1 (update every frame). Recommended values: 1-4.
    """

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    slice_height_range: float | None = 0.1
    """Height range (in meters) above and below the sensor to slice meshes for 2D lidar.

    Only mesh triangles within [sensor_z - slice_height_range, sensor_z + slice_height_range]
    will be kept. This reduces memory and improves performance for 2D lidar applications.

    Set to None to disable slicing and use full 3D meshes (for 3D lidar/depth sensors).
    Defaults to 0.1 meters (±10cm)."""

    enable_3d_scan: bool = False
    """Enable full 3D scanning instead of 2D planar scanning.

    When True, meshes are not sliced by height and all ray patterns are used in 3D.
    When False (default), meshes are sliced to a thin horizontal layer for 2D lidar.
    """

    attach_yaw_only: bool | None = None
    """Whether the rays' starting positions and directions only track the yaw orientation.
    Defaults to None, which doesn't raise a warning of deprecated usage.

    This is useful for ray-casting height maps, where only yaw rotation is needed.

    .. deprecated:: 2.1.1

        This attribute is deprecated and will be removed in the future. Please use
        :attr:`ray_alignment` instead.

        To get the same behavior as setting this parameter to ``True`` or ``False``, set
        :attr:`ray_alignment` to ``"yaw"`` or "base" respectively.

    """

    ray_alignment: Literal["base", "yaw", "world"] = "base"
    """Specify in what frame the rays are projected onto the ground. Default is "base".

    The options are:

    * ``base`` if the rays' starting positions and directions track the full root position and orientation.
    * ``yaw`` if the rays' starting positions and directions track root position and only yaw component of orientation.
      This is useful for ray-casting height maps.
    * ``world`` if rays' starting positions and directions are always fixed. This is useful in combination with a mapping
      package on the robot and querying ray-casts in a global frame.
    """

    pattern_cfg: PatternBaseCfg = MISSING
    """The pattern that defines the local ray starting positions and directions."""

    max_distance: float = 1e6
    """Maximum distance (in meters) from the sensor to ray cast to. Defaults to 1e6."""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """The range of drift (in meters) to add to the ray starting positions (xyz) in world frame. Defaults to (0.0, 0.0).

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """

    ray_cast_drift_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """The range of drift (in meters) to add to the projected ray points in local projection frame. Defaults to
    a dictionary with zero drift for each x, y and z axis.

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
    """The configuration object for the visualization markers. Defaults to RAY_CASTER_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
