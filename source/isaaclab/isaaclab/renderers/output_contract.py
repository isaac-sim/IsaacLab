# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer→consumer output contract types.

Leaf module shared by :class:`~isaaclab.renderers.BaseRenderer` and
:class:`~isaaclab.sensors.camera.CameraData` to avoid a direct dependency
between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class RenderBufferKind(StrEnum):
    """Canonical names for the per-pixel render buffer kinds a renderer can publish.

    String values match the vocabulary used by
    :attr:`isaaclab.sensors.camera.CameraCfg.data_types`.
    """

    RGB = "rgb"
    RGBA = "rgba"
    ALBEDO = "albedo"
    DEPTH = "depth"
    DISTANCE_TO_IMAGE_PLANE = "distance_to_image_plane"
    DISTANCE_TO_CAMERA = "distance_to_camera"
    NORMALS = "normals"
    MOTION_VECTORS = "motion_vectors"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION_FAST = "instance_segmentation_fast"
    INSTANCE_ID_SEGMENTATION_FAST = "instance_id_segmentation_fast"
    SIMPLE_SHADING_CONSTANT_DIFFUSE = "simple_shading_constant_diffuse"
    SIMPLE_SHADING_DIFFUSE_MDL = "simple_shading_diffuse_mdl"
    SIMPLE_SHADING_FULL_MDL = "simple_shading_full_mdl"


@dataclass(frozen=True)
class RenderBufferSpec:
    """Per-pixel layout (channels + dtype) for one render buffer kind."""

    channels: int
    """Number of per-pixel channels (last dimension of the allocated tensor)."""

    dtype: torch.dtype
    """Torch dtype the renderer writes for this render buffer kind."""
