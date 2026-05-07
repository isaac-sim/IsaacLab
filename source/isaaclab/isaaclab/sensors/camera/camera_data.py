# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

# Re-exported as part of the public isaaclab.sensors.camera API
from isaaclab.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.utils.warp.proxy_array import ProxyArray

__all__ = ["CameraData", "RenderBufferKind", "RenderBufferSpec"]


def _wrap(buf: torch.Tensor) -> ProxyArray:
    """Wrap a torch buffer as a :class:`ProxyArray` with a zero-copy ``wp.array`` view."""
    return ProxyArray(wp.from_torch(buf))


@dataclass
class CameraData:
    """Data container for the camera sensor.

    All array-valued fields are :class:`~isaaclab.utils.warp.proxy_array.ProxyArray`
    wrappers around an underlying :class:`warp.array` (Warp-primary storage with a
    cached zero-copy ``torch.Tensor`` view available via ``.torch``).
    """

    ##
    # Frame state.
    ##

    pos_w: ProxyArray = None
    """Position [m] of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: ProxyArray = None
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

    intrinsic_matrices: ProxyArray = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """

    output: dict[str, ProxyArray] = None
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
        supported_specs: dict[RenderBufferKind, RenderBufferSpec],
    ) -> CameraData:
        """Build a :class:`CameraData` with output buffers pre-allocated.

        Allocates one ``(num_views, height, width, channels)`` buffer per kind
        in the intersection of ``data_types`` and ``supported_specs``, using
        the channels and dtype from each :class:`RenderBufferSpec`. Each buffer
        is exposed as a :class:`ProxyArray` (``wp.array`` primary, zero-copy
        ``torch.Tensor`` view via ``.torch``).

        Args:
            data_types: Requested output names (typically :attr:`CameraCfg.data_types`).
                Every name must be a member of :class:`RenderBufferKind`.
            height: Image height in pixels.
            width: Image width in pixels.
            num_views: Number of camera views (batch dimension).
            device: Torch device on which to allocate the buffers.
            supported_specs: Per-buffer layout the active renderer can produce,
                keyed by :class:`RenderBufferKind`. Names absent from this mapping
                are not allocated.

        Returns:
            A new :class:`CameraData` with :attr:`image_shape`, :attr:`output`,
            and :attr:`info` populated; all other fields at their defaults.

        Raises:
            ValueError: If ``data_types`` contains names that are not members of
                :class:`RenderBufferKind`.
        """
        requested: set[RenderBufferKind] = set()
        unknown: list[str] = []
        for name in data_types:
            try:
                requested.add(RenderBufferKind(name))
            except ValueError:
                unknown.append(name)
        if unknown:
            raise ValueError(f"Unknown RenderBufferKind name(s): {unknown}. Expected members of RenderBufferKind.")
        # rgb is exposed as a view into rgba when the renderer publishes both,
        # so requesting either one allocates the shared rgba buffer.
        rgb_alias = (
            RenderBufferKind.RGBA in supported_specs
            and RenderBufferKind.RGB in supported_specs
            and (RenderBufferKind.RGB in requested or RenderBufferKind.RGBA in requested)
        )
        if rgb_alias:
            requested.update({RenderBufferKind.RGB, RenderBufferKind.RGBA})

        # Allocate the underlying torch buffers first so we can preserve the
        # rgb-as-view-of-rgba alias as a torch slice (shared memory). Each
        # buffer is then wrapped as a ProxyArray over a wp.from_torch view —
        # the wp.array and torch.Tensor share the same allocation.
        torch_buffers: dict[str, torch.Tensor] = {}
        for name, spec in supported_specs.items():
            if name not in requested:
                continue
            if rgb_alias and name == RenderBufferKind.RGB:
                continue
            torch_buffers[str(name)] = torch.zeros(
                (num_views, height, width, spec.channels),
                dtype=spec.dtype,
                device=device,
            ).contiguous()
        if rgb_alias:
            # Share storage with rgba: the slice is a non-contiguous torch view.
            # ``wp.from_torch`` accepts strided views, so the resulting ProxyArray
            # tracks the same underlying memory as the rgba buffer.
            torch_buffers[str(RenderBufferKind.RGB)] = torch_buffers[str(RenderBufferKind.RGBA)][..., :3]

        buffers: dict[str, ProxyArray] = {name: _wrap(buf) for name, buf in torch_buffers.items()}

        return cls(
            image_shape=(height, width),
            output=buffers,
            info={name: None for name in buffers},
        )

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> ProxyArray:
        """Quaternion orientation `(x, y, z, w)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        converted = convert_camera_frame_orientation_convention(self.quat_w_world.torch, origin="world", target="ros")
        return _wrap(converted.contiguous())

    @property
    def quat_w_opengl(self) -> ProxyArray:
        """Quaternion orientation `(x, y, z, w)` of the sensor origin in the world frame, following
        Opengl / USD Camera convention.

        .. note::
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        converted = convert_camera_frame_orientation_convention(
            self.quat_w_world.torch, origin="world", target="opengl"
        )
        return _wrap(converted.contiguous())
