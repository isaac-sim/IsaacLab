# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .lidar import Lidar


@configclass
class LidarSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a lidar sensor."""

    func: Callable | str = "{DIR}.lidar:spawn_lidar"


@configclass
class LidarCfg(SensorBaseCfg):
    """Configuration parameters for a lidar sensor."""

    @configclass
    class OffsetCfg:
        """Offset of the lidar frame from its parent prim."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation relative to the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` relative to the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

    class_type: type[Lidar] | str = "{DIR}.lidar:Lidar"
    """Sensor implementation class."""

    offset: OffsetCfg = OffsetCfg()
    """Pose of the lidar frame relative to its parent. Defaults to identity."""

    spawn: LidarSpawnerCfg | None = LidarSpawnerCfg()
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """

    model_name: Literal["LidarCore", "LidarWpm"] = "LidarCore"
    """Lidar model. Defaults to ``LidarCore``.

    ``LidarCore`` applies ``OmniSensorGenericLidarCoreAPI``. ``LidarWpm`` applies
    ``OmniSensorGenericLidarWpmAPI``.
    """

    scan_type: Literal["rotary", "solid_state"] = "rotary"
    """How the firing pattern is scheduled. Defaults to ``rotary``.

    Both OmniLidar schemas default to a rotary scan. ``solid_state`` fires the pattern once per
    scan. ``rotary`` repeats it at each azimuth step.
    """

    rot_pattern_firing_res_deg: float = 0.1
    """Azimuth step of a rotary scan, in degrees. Defaults to 0.1.

    A solid-state scan does not use this value.
    """

    valid_start_azimuth_deg: float = 0.0
    """Start of the rotary azimuth window, in degrees. Defaults to 0."""

    valid_end_azimuth_deg: float = 360.0
    """End of the rotary azimuth window, in degrees. Defaults to 360.

    A window whose end is not after its start wraps through 360 degrees.
    """

    rotation_direction: Literal["CW", "CCW"] = "CW"
    """Direction of a core rotary sweep. Defaults to ``CW``.

    ``LidarWpm`` does not read this value.
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

    ``LidarCore`` does not read this value.
    """

    max_returns: int = 2
    """Maximum returns stored for each ray. Defaults to 2, matching ``maxReturns``."""

    output_frame_of_reference: Literal["SENSOR", "WORLD"] = "SENSOR"
    """Coordinate frame of the returned points. Defaults to ``SENSOR``."""

    far_range_m: float = 200.0
    """Maximum return distance, in meters. Defaults to 200."""

    elements_coords_type: Literal["CARTESIAN", "SPHERICAL"] = "SPHERICAL"
    """Coordinate convention of the returned points. Defaults to ``SPHERICAL``."""

    instant_lidar: bool = False
    """Whether one render step publishes a complete scan. Defaults to False."""

    include_invalid_points: bool = False
    """Whether missed rays keep a slot in the point cloud. Defaults to False.

    ``False`` drops misses. ``True`` keeps one slot per ray, with misses non-finite in
    :attr:`~isaaclab.sensors.lidar.LidarData.points`.
    """

    renderer_cfg: RendererCfg = field(default_factory=RendererCfg)
    """Renderer configuration for the lidar sensor."""

    def __post_init__(self):
        # Lidar capture is implemented only by OVRTX, so the placeholder becomes that renderer.
        if getattr(self.renderer_cfg, "renderer_type", None) == "default":
            from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg

            self.renderer_cfg = OVRTXRendererCfg()

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

        if self.output_frame_of_reference not in ("SENSOR", "WORLD"):
            raise ValueError(
                f"Lidar output_frame_of_reference must be 'SENSOR' or 'WORLD', got {self.output_frame_of_reference!r}."
            )

        if self.model_name not in ("LidarCore", "LidarWpm"):
            raise ValueError(f"Lidar model_name must be 'LidarCore' or 'LidarWpm', got {self.model_name!r}.")

        if self.scan_type not in ("solid_state", "rotary"):
            raise ValueError(f"Lidar scan_type must be 'solid_state' or 'rotary', got {self.scan_type!r}.")

        if self.rot_pattern_firing_res_deg <= 0.0:
            raise ValueError(
                f"Lidar rot_pattern_firing_res_deg must be positive, got {self.rot_pattern_firing_res_deg}."
            )

        if self.rotation_direction not in ("CW", "CCW"):
            raise ValueError(f"Lidar rotation_direction must be 'CW' or 'CCW', got {self.rotation_direction!r}.")

        if self.far_range_m <= 0.0:
            raise ValueError(f"Lidar far_range_m must be positive, got {self.far_range_m}.")

        if any(time < 0 for time in self.fire_time_ns):
            raise ValueError("Lidar fire_time_ns values cannot be negative.")

    def num_rays(self) -> int:
        """Return the number of point slots in one full scan.

        A solid-state scan fires the pattern once. A rotary scan repeats it once per azimuth step
        across the configured window. Each firing keeps :attr:`max_returns` slots.
        """
        ticks = 1
        if self.scan_type == "rotary":
            span = self.valid_end_azimuth_deg - self.valid_start_azimuth_deg
            if span <= 0.0:
                span += 360.0
            ticks = math.ceil(span / self.rot_pattern_firing_res_deg)

        return ticks * len(self.azimuth_deg) * self.max_returns
