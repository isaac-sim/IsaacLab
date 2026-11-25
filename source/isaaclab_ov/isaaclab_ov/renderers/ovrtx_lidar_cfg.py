# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.sensors import LidarCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .ovrtx_renderer_cfg import OVRTXRendererCfg

if TYPE_CHECKING:
    from .ovrtx_lidar_sensor import OVRTXLidar


@configclass
class OVRTXLidarSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a lidar sensor."""

    func: Callable | str = "{DIR}.ovrtx_lidar_sensor:spawn_lidar"


@configclass
class OVRTXLidarCfg(LidarCfg):
    """Configuration for a tiled OvRTX OmniLidar sensor."""

    class_type: type[OVRTXLidar] | str = "{DIR}.ovrtx_lidar_sensor:OVRTXLidar"
    spawn: OVRTXLidarSpawnerCfg | None = OVRTXLidarSpawnerCfg()
    renderer_cfg: OVRTXRendererCfg = OVRTXRendererCfg(motion_bvh="auto")

    scan_type: Literal["rotary", "solidState"] = "rotary"
    """USD ``scanType`` token. Defaults to ``rotary``.

    ``solidState`` fires the pattern once per scan. ``rotary`` repeats it at each azimuth step.
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

    azimuth_deg: list[float] = MISSING
    """Azimuth of each pattern ray, in degrees, relative to the tick. Zero is the tick azimuth."""

    elevation_deg: list[float] = MISSING
    """Elevation of each pattern ray, in degrees. Zero is the horizon."""

    fire_time_ns: list[int] = MISSING
    """Firing time of each pattern ray, in nanoseconds, measured from the start of its tick."""

    channel_id: list[int] = MISSING
    """Channel id of each ray. One entry is required for every azimuth."""

    tiled_subsampling: int = 0
    """Published-slot stride for ``LidarWpm``. ``0`` and ``1`` publish every authored slot.

    A larger value keeps one slot out of that many authored slots and drops the remainder.
    """

    max_returns: int = 2
    """Maximum returns stored for each ray. Defaults to 2, matching ``maxReturns``."""

    output_frame_of_reference: Literal["SENSOR", "WORLD"] = "SENSOR"
    """Coordinate frame of the returned points. Defaults to ``SENSOR``."""

    far_range_m: float = 200.0
    """Maximum return distance, in meters. Defaults to 200."""

    elements_coords_type: Literal["CARTESIAN", "SPHERICAL"] = "CARTESIAN"
    """Native coordinate encoding. Both modes publish Cartesian XYZ positions in meters. Defaults to ``CARTESIAN``."""

    instant_lidar: bool = True
    """Whether one render step publishes a complete scan. Defaults to True."""

    include_invalid_points: bool = True
    """Keep missed rays in their original slots. Tiled OvRTX lidar requires True.

    Missed rays have non-finite coordinates in
    :attr:`~isaaclab.sensors.lidar.LidarData.points`.
    """

    def __post_init__(self):
        if self.renderer_cfg.motion_bvh not in ("auto", "enable"):
            raise ValueError("OvRTX lidar requires motion_bvh='auto' or 'enable'.")

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

        if self.scan_type not in ("solidState", "rotary"):
            raise ValueError(f"Lidar scan_type must be 'solidState' or 'rotary', got {self.scan_type!r}.")

        if self.rot_pattern_firing_res_deg <= 0.0:
            raise ValueError(
                f"Lidar rot_pattern_firing_res_deg must be positive, got {self.rot_pattern_firing_res_deg}."
            )

        if self.far_range_m <= 0.0:
            raise ValueError(f"Lidar far_range_m must be positive, got {self.far_range_m}.")

        if any(time < 0 for time in self.fire_time_ns):
            raise ValueError("Lidar fire_time_ns values cannot be negative.")

    def validate_config(self) -> None:
        """Require the dense return layout used by the tiled point-cloud reader."""
        if not self.include_invalid_points:
            raise ValueError("Tiled OvRTX lidar requires include_invalid_points=True to preserve missed-ray slots.")

    def num_rays(self) -> int:
        """Return the number of point slots in one full scan.

        A solid-state scan fires the pattern once. A rotary scan repeats it once per azimuth step
        across the configured window. Each firing keeps :attr:`max_returns` slots. ``LidarWpm``
        with :attr:`tiled_subsampling` above 1 publishes one slot out of that many and drops the remainder.
        """
        ticks = 1
        if self.scan_type == "rotary":
            span = self.valid_end_azimuth_deg - self.valid_start_azimuth_deg
            if span <= 0.0:
                span += 360.0
            ticks = math.ceil(span / self.rot_pattern_firing_res_deg)

        slots = ticks * len(self.azimuth_deg) * self.max_returns
        # A stride of 2 on 7889 rays publishes 3944 slots.
        if self.tiled_subsampling > 1:
            slots //= self.tiled_subsampling
        return slots
