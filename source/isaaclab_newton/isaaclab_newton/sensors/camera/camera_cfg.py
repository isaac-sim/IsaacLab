# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import FisheyeCameraCfg, PinholeCameraCfg
from isaaclab.utils import configclass

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

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf``.
    """

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"].

    Please refer to the :class:`Camera` class for a list of available data types.
    """

    width: int = MISSING
    """Width of the image in pixels."""

    height: int = MISSING
    """Height of the image in pixels."""

    update_latest_camera_pose: bool = False
    """Whether to update the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is updated in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """

    semantic_filter: str | list[str] = "*:*"
    """A string or a list specifying a semantic filter predicate. Defaults to ``"*:*"``.

    If a string, it should be a disjunctive normal form of (semantic type, labels). For examples:

    * ``"typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelE"``:
      All prims with semantic type "typeA" and label "labelA" but not "labelB" or with label "labelC".
      Also, all prims with semantic type "typeB" and label "labelA", or with semantic type "typeC" and label "labelE".
    * ``"typeA : * ; * : labelA"``: All prims with semantic type "typeA" or with label "labelA"

    If a list of strings, each string should be a semantic type. The segmentation for prims with
    semantics of the specified types will be retrieved. For example, if the list is ["class"], only
    the segmentation for prims with semantics of type "class" will be retrieved.

    .. seealso::

        For more information on the semantics filter, see the documentation on `Replicator Semantics Schema Editor`_.

    .. _Replicator Semantics Schema Editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize the semantic segmentation images. Defaults to True.

    If True, semantic segmentation is converted to an image where semantic IDs are mapped to colors
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    semantic_segmentation_mapping: dict = {}
    """Dictionary mapping semantics to specific colours

    Eg.

    .. code-block:: python

        {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (61, 178, 255, 255),
        }

    """
