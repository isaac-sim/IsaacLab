# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for the multi-mesh ray-cast camera sensor."""

import torch

from isaaclab.sensors.camera import CameraData


class MultiMeshRayCasterCameraData(CameraData):
    """Data container for the multi-mesh ray-cast camera sensor.

    This class extends :class:`CameraData` with additional mesh-id information.
    It does not inherit from :class:`RayCasterData` because the camera variant
    manages its own torch-based pose and hit buffers independently from the
    warp-native :class:`RayCasterData`.
    """

    image_mesh_ids: torch.Tensor = None
    """The mesh ids of the image pixels.

    Shape is (N, H, W, 1), where N is the number of sensors, H and W are the height and width of the image,
    and 1 is the number of mesh ids per pixel.
    """
