# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass
from typing import Any

from isaaclab.utils.math import convert_camera_frame_orientation_convention


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # Camera data
    ##

    image_shape: tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: torch.Tensor = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """

    output: dict[str, torch.Tensor] = None
    """The retrieved sensor data with sensor types as key.

    The format of the data is available in the `Replicator Documentation`_. For semantic-based data,
    this corresponds to the ``"data"`` key in the output of the sensor.

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """

    info: list[dict[str, Any]] = None
    """The retrieved sensor info with sensor types as key.

    This contains extra information provided by the sensor such as semantic segmentation label mapping, prim paths.
    For semantic-based data, this corresponds to the ``"info"`` key in the output of the sensor. For other sensor
    types, the info is empty.
    """

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following
        Opengl / USD Camera convention.

        .. note::
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="opengl")
