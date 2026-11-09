# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .lidar import Lidar


@configclass
class LidarSpawnerCfg(SpawnerCfg):
    """Marks a lidar for per-environment cloning.

    The sensor authors the ``OmniLidar`` prim. This spawner puts that prim on the clone plan so
    each environment receives a copy, including in the renderer stage.
    """

    func: Callable = "{DIR}.lidar:spawn_lidar"
    """Authoring function. The sensor calls :func:`~isaaclab.sensors.lidar.spawn_lidar` itself."""


@configclass
class LidarCfg(SensorBaseCfg):
    """Configuration for a tiled OVRTX lidar.

    One :class:`~isaaclab.sensors.lidar.Lidar` prim is spawned in each environment. The renderer
    places every prim on a single lidar render product and splits the point cloud back into one
    row per environment. The prim orientation is used as authored.
    """

    @configclass
    class OffsetCfg:
        """Offset of the lidar frame from its parent prim."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation relative to the parent frame, in meters. Defaults to the parent origin."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` relative to the parent frame. Defaults to identity."""

    class_type: type[Lidar] | str = "{DIR}.lidar:Lidar"
    """Sensor implementation class."""

    spawn: LidarSpawnerCfg | None = LidarSpawnerCfg()
    """Clone-plan spawner. Defaults to one lidar prim per environment.

    Set this to None only when the prims already exist and are replicated by another prototype.
    """

    offset: OffsetCfg = OffsetCfg()
    """Pose of the lidar frame relative to its parent. Defaults to identity."""

    azimuth_deg: list[float] = MISSING
    """Solid-state azimuth of each ray, in degrees. Zero is the sensor boresight."""

    elevation_deg: list[float] = MISSING
    """Solid-state elevation of each ray, in degrees. Zero is the horizon."""

    fire_time_ns: list[int] = MISSING
    """Firing time of each ray, in nanoseconds, measured from the start of the scan."""

    channel_id: list[int] = MISSING
    """Channel id of each ray. One entry is required for every azimuth."""

    tiled_subsampling: int = 0
    """Number of rays skipped between published returns. ``0`` publishes every ray."""

    max_returns: int = 1
    """Maximum returns stored for each ray. Defaults to 1."""

    frame_of_reference: Literal["SENSOR", "WORLD", "CUSTOM"] = "WORLD"
    """Coordinate frame of the returned points. Defaults to world coordinates."""

    coords_type: Literal["CARTESIAN", "SPHERICAL"] = "CARTESIAN"
    """Coordinate convention of the returned points. Defaults to Cartesian ``x, y, z``."""

    instant_lidar: bool = True
    """Whether one render step publishes a complete scan. Defaults to True."""

    include_invalid_points: bool = True
    """Whether missed rays keep a slot in the point cloud. Defaults to True.

    Keeping those slots makes every environment return ``num_rays`` points. Misses are non-finite
    in :attr:`~isaaclab.sensors.lidar.LidarData.points`.
    """

    renderer_cfg: RendererCfg = MISSING
    """OVRTX renderer used to capture the lidar. It must be an ``ovrtx`` renderer configuration.

    Equal configurations share one renderer with the cameras in the scene, so a lidar and a camera
    that name the same configuration are rendered in one step.
    """

    def __post_init__(self):
        lengths = {
            "azimuth_deg": len(self.azimuth_deg),
            "elevation_deg": len(self.elevation_deg),
            "fire_time_ns": len(self.fire_time_ns),
            "channel_id": len(self.channel_id),
        }
        if len(set(lengths.values())) != 1 or lengths["azimuth_deg"] == 0:
            raise ValueError(f"Lidar firing-pattern arrays must have the same non-zero length, got {lengths}.")
        if self.max_returns < 1:
            raise ValueError(f"Lidar max_returns must be at least 1, got {self.max_returns}.")
        if self.tiled_subsampling < 0:
            raise ValueError(f"Lidar tiled_subsampling cannot be negative, got {self.tiled_subsampling}.")
        if self.frame_of_reference == "CUSTOM":
            raise ValueError("Lidar frame_of_reference 'CUSTOM' is not supported. Use 'WORLD' or 'SENSOR'.")
        if any(time < 0 for time in self.fire_time_ns):
            raise ValueError("Lidar fire_time_ns values cannot be negative.")
