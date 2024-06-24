# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import FisheyeCameraCfg, PinholeCameraCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(SensorBaseCfg):
    """Configuration for a tiled rendering-based camera sensor."""

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

    class_type: type = TiledCamera

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """

    spawn: PinholeCameraCfg | FisheyeCameraCfg | None = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"].

    Please refer to the :class:`TiledCamera` class for a list of available data types.
    """

    width: int = MISSING
    """Width of the image in pixels."""

    height: int = MISSING
    """Height of the image in pixels."""

    return_latest_camera_pose: bool = False
    """Whether to return the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is returned in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """
