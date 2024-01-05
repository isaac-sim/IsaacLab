# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing_extensions import Literal

from omni.isaac.orbit.sim import FisheyeCameraCfg, PinholeCameraCfg
from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .camera import Camera


@configclass
class CameraCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""

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

    class_type: type = Camera

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
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"]."""

    width: int = MISSING
    """Width of the image in pixels."""

    height: int = MISSING
    """Height of the image in pixels."""

    semantic_types: list[str] = ["class"]
    """List of allowed semantic types the types. Defaults to ["class"].

    For example, if semantic types is [“class”], only the bounding boxes for prims with semantics of
    type “class” will be retrieved.

    More information available at:
        https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    colorize: bool = False
    """whether to output colorized semantic information or non-colorized one. Defaults to False.

    If True, the semantic images will be a 2D array of RGBA values, where each pixel is colored according to
    the semantic type. Accordingly, the information output will contain mapping from color to semantic labels.

    If False, the semantic images will be a 2D array of integers, where each pixel is an integer representing
    the semantic ID. Accordingly, the information output will contain mapping from semantic ID to semantic labels.
    """
