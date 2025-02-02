# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast camera sensor."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .patterns import PinholeCameraPatternCfg
from .ray_caster_camera import RayCasterCamera
from .ray_caster_cfg import RayCasterCfg


@configclass
class RayCasterCameraCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type = RayCasterCamera

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    data_types: list[str] = ["distance_to_image_plane"]
    """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf`` for ``distance_to_camera`` and ``nan``
      for ``distance_to_image_plane`` data type.
    """

    pattern_cfg: PinholeCameraPatternCfg = MISSING
    """The pattern that defines the local ray starting positions and directions in a pinhole camera pattern."""

    def __post_init__(self):
        # for cameras, this quantity should be False always.
        self.attach_yaw_only = False
