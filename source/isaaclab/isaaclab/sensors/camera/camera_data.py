# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Re-exported as part of the public isaaclab.sensors.camera API
from isaaclab.renderers.output_contract import CameraDataType, OutputSpec
from isaaclab.utils.math import convert_camera_frame_orientation_convention

__all__ = ["CameraData", "CameraDataType", "OutputSpec"]


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
    """Quaternion orientation `(x, y, z, w)` of the sensor origin in world frame, following the world coordinate frame

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

    info: dict[str, Any] = None
    """The retrieved sensor info with sensor types as key.

    This contains extra information provided by the sensor such as semantic segmentation label mapping, prim paths.
    For semantic-based data, this corresponds to the ``"info"`` key in the output of the sensor. For other sensor
    types, the info is empty.
    """

    @classmethod
    def allocate(
        cls,
        data_types: list[str],
        height: int,
        width: int,
        num_views: int,
        device: torch.device | str,
        supported_specs: dict[CameraDataType, OutputSpec],
    ) -> CameraData:
        """Build a :class:`CameraData` with output buffers pre-allocated.

        Allocates one ``(num_views, height, width, channels)`` tensor per type
        in the intersection of ``data_types`` and ``supported_specs``, using
        the channels and dtype from each :class:`OutputSpec`.

        Args:
            data_types: Requested output names (typically :attr:`CameraCfg.data_types`).
                Every name must be a member of :class:`CameraDataType`.
            height: Image height in pixels.
            width: Image width in pixels.
            num_views: Number of camera views (batch dimension).
            device: Torch device on which to allocate the buffers.
            supported_specs: Per-output layout the active renderer can produce,
                keyed by :class:`CameraDataType`. Names absent from this mapping
                are not allocated.

        Returns:
            A new :class:`CameraData` with :attr:`image_shape`, :attr:`output`,
            and :attr:`info` populated; all other fields at their defaults.

        Raises:
            ValueError: If ``data_types`` contains names that are not members of
                :class:`CameraDataType`.
        """
        requested: set[CameraDataType] = set()
        unknown: list[str] = []
        for name in data_types:
            try:
                requested.add(CameraDataType(name))
            except ValueError:
                unknown.append(name)
        if unknown:
            raise ValueError(
                f"Unknown CameraDataType name(s): {unknown}. Expected members of CameraDataType."
            )
        # rgb is exposed as a view into rgba when the renderer publishes both,
        # so requesting either one allocates the shared rgba buffer.
        rgb_alias = (
            CameraDataType.RGBA in supported_specs
            and CameraDataType.RGB in supported_specs
            and (CameraDataType.RGB in requested or CameraDataType.RGBA in requested)
        )
        if rgb_alias:
            requested.update({CameraDataType.RGB, CameraDataType.RGBA})

        buffers: dict[str, torch.Tensor] = {}
        for name, spec in supported_specs.items():
            if name not in requested:
                continue
            if rgb_alias and name == CameraDataType.RGB:
                continue
            buffers[str(name)] = torch.zeros(
                (num_views, height, width, spec.channels),
                dtype=spec.dtype,
                device=device,
            ).contiguous()
        if rgb_alias:
            buffers[str(CameraDataType.RGB)] = buffers[str(CameraDataType.RGBA)][..., :3]

        return cls(
            image_shape=(height, width),
            output=buffers,
            info={name: None for name in buffers},
        )

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(x, y, z, w)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(x, y, z, w)` of the sensor origin in the world frame, following
        Opengl / USD Camera convention.

        .. note::
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="opengl")
