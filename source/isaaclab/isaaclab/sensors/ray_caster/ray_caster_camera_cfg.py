# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast camera sensor."""

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import config_field

from .patterns import PinholeCameraPatternCfg
from .ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from .ray_caster_camera import RayCasterCamera


@dataclass
class RayCasterCameraCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor."""

    @dataclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = config_field((0.0, 0.0, 0.0))
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = config_field((0.0, 0.0, 0.0, 1.0))
        """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

        convention: Literal["opengl", "ros", "world"] = config_field("ros")
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera)
          convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type["RayCasterCamera"] | str = config_field("{DIR}.ray_caster_camera:RayCasterCamera")

    offset: OffsetCfg = config_field(OffsetCfg())
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    data_types: list[str] = config_field(["distance_to_image_plane"])
    """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""

    depth_clipping_behavior: Literal["max", "zero", "none"] = config_field("none")
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none"``: No clipping is applied. Values will be returned as ``inf`` for missed rays in both
      ``distance_to_camera`` and ``distance_to_image_plane`` data types.
    """

    pattern_cfg: PinholeCameraPatternCfg = config_field(MISSING)
    """The pattern that defines the local ray starting positions and directions in a pinhole camera pattern."""

    def __post_init__(self):
        # for cameras, this quantity should be False always.
        self.ray_alignment = "base"
