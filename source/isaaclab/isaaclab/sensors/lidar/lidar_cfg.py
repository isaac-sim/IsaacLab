# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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

    One :class:`~isaaclab.sensors.lidar.Lidar` prim is spawned in each environment. ``model`` selects
    Lidar generic wpm or Lidar generic core, and ``scan_type`` selects a solid-state or rotary scan.
    The renderer places every prim on a single lidar render product and splits the point cloud back
    into one row per environment. The prim orientation is used as authored.
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

    model: Literal["wpm", "core"] = "core"
    """Lidar model authored on the prim. Defaults to ``core``.

    An ``OmniLidar`` prim selects Lidar generic core when ``modelName`` is absent. ``wpm`` applies
    Lidar generic wpm, which is the model selected by ``modelName`` ``LidarWpm``.
    """

    scan_type: Literal["solid_state", "rotary"] = "rotary"
    """How the firing pattern is scheduled. Defaults to ``rotary``.

    Both OmniLidar schemas default to a rotary scan. ``solid_state`` fires the pattern once per
    scan. ``rotary`` repeats it at each azimuth step.
    """

    azimuth_resolution_deg: float = 1.0
    """Azimuth step of a rotary scan, in degrees. Defaults to 1.

    A solid-state scan does not use this value. It is written as ``rotPatternFiringResDeg``.
    """

    valid_start_azimuth_deg: float = 0.0
    """Start of the rotary azimuth window, in degrees. Defaults to 0."""

    valid_end_azimuth_deg: float = 360.0
    """End of the rotary azimuth window, in degrees. Defaults to 360.

    A window whose end is not after its start wraps through 360 degrees.
    """

    rotation_direction: Literal["CW", "CCW"] = "CW"
    """Direction of a core rotary sweep. Defaults to ``CW``.

    Lidar generic wpm does not read this value.
    """

    azimuth_deg: list[float] = MISSING
    """Azimuth of each pattern ray, in degrees, relative to the tick. Zero is the tick azimuth."""

    elevation_deg: list[float] = MISSING
    """Elevation of each pattern ray, in degrees. Zero is the horizon."""

    fire_time_ns: list[int] = MISSING
    """Firing time of each pattern ray, in nanoseconds, measured from the start of its tick."""

    channel_id: list[int] = MISSING
    """Channel id of each ray. One entry is required for every azimuth."""

    tiled_subsampling: int = 0
    """Number of rays skipped between published returns. ``0`` publishes every ray.

    Lidar generic core does not read this value.
    """

    max_returns: int = 1
    """Maximum returns stored for each ray. Defaults to 1."""

    frame_of_reference: Literal["SENSOR", "WORLD", "CUSTOM"] = "WORLD"
    """Coordinate frame of the returned points. Defaults to world coordinates."""

    far_range: float | None = None
    """Maximum return distance, in meters. ``None`` leaves the sensor default of 200 m."""

    coords_type: Literal["CARTESIAN", "SPHERICAL"] = "CARTESIAN"
    """Coordinate convention of the returned points. Defaults to Cartesian ``x, y, z``.

    ``SPHERICAL`` is rejected. Published points stay Cartesian so each consumer can treat the
    three components as a position in meters.
    """

    instant_lidar: bool = True
    """Whether one render step publishes a complete scan. Defaults to True."""

    include_invalid_points: bool = True
    """Whether missed rays keep a slot in the point cloud. Defaults to True.

    Keeping those slots makes every environment return ``num_rays`` points. Misses are non-finite
    in :attr:`~isaaclab.sensors.lidar.LidarData.points`. ``False`` is rejected: a compacted cloud
    drops the authored ray index, so a return can no longer be placed on its ray.
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

        if self.coords_type != "CARTESIAN":
            raise ValueError(
                f"Lidar coords_type must be 'CARTESIAN' so each return stays a position in meters,"
                f" got {self.coords_type!r}."
            )

        if not self.include_invalid_points:
            raise ValueError(
                "Lidar include_invalid_points=False is not supported. Missed rays must keep their"
                " authored slots until returns can be placed by ray id."
            )

        if self.model not in ("wpm", "core"):
            raise ValueError(f"Lidar model must be 'wpm' or 'core', got {self.model!r}.")

        if self.scan_type not in ("solid_state", "rotary"):
            raise ValueError(f"Lidar scan_type must be 'solid_state' or 'rotary', got {self.scan_type!r}.")

        if self.azimuth_resolution_deg <= 0.0:
            raise ValueError(f"Lidar azimuth_resolution_deg must be positive, got {self.azimuth_resolution_deg}.")

        if self.rotation_direction not in ("CW", "CCW"):
            raise ValueError(f"Lidar rotation_direction must be 'CW' or 'CCW', got {self.rotation_direction!r}.")

        if self.far_range is not None and self.far_range <= 0.0:
            raise ValueError(f"Lidar far_range must be positive, got {self.far_range}.")

        if any(time < 0 for time in self.fire_time_ns):
            raise ValueError("Lidar fire_time_ns values cannot be negative.")

    def num_rays(self) -> int:
        """Return the number of point slots in one full scan.

        A solid-state scan fires the pattern once. A rotary scan repeats it once per azimuth step
        across the configured window. Each firing keeps :attr:`max_returns` slots.
        """
        ticks = 1 if self.scan_type == "solid_state" else self._rotary_tick_count()
        return ticks * len(self.azimuth_deg) * self.max_returns

    def _rotary_tick_count(self) -> int:
        """Return how many times a rotary scan applies the firing pattern."""
        span = self.valid_end_azimuth_deg - self.valid_start_azimuth_deg
        if span <= 0.0:
            span += 360.0
        return math.ceil(span / self.azimuth_resolution_deg)
