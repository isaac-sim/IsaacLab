# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import warp as wp

# Re-exported as part of the public isaaclab.sensors.camera API
from isaaclab.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

__all__ = ["CameraData", "RenderBufferKind", "RenderBufferSpec"]


class CameraData:
    """Data container for the camera sensor.

    Public properties return :class:`~isaaclab.utils.warp.ProxyArray` wrappers.
    Use ``.torch`` for a cached zero-copy :class:`torch.Tensor` view or
    ``.warp`` for the underlying :class:`warp.array`.
    """

    def __init__(self):
        # ProxyArray wrappers — created in create_buffers()
        self._pos_w: ProxyArray | None = None
        self._quat_w_world: ProxyArray | None = None
        self._intrinsic_matrices: ProxyArray | None = None
        self._quat_w_ros: ProxyArray | None = None
        self._quat_w_opengl: ProxyArray | None = None

        # Output image buffers — allocated in allocate()
        self._output: dict[str, ProxyArray] | None = None

        self.image_shape: tuple[int, int] | None = None
        """A tuple containing (height, width) of the camera sensor."""

        self.info: dict[str, Any] | None = None
        """The retrieved sensor info with sensor types as key.

        This contains extra information provided by the sensor such as semantic segmentation label mapping, prim paths.
        For semantic-based data, this corresponds to the ``"info"`` key in the output of the sensor. For other sensor
        types, the info is empty.
        """

    ##
    # Frame state.
    ##

    @property
    def pos_w(self) -> ProxyArray:
        """Position of the sensor origin in world frame [m], following ROS convention.

        Shape is (N,), dtype ``wp.vec3f``. In torch this resolves to (N, 3),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._pos_w

    @property
    def quat_w_world(self) -> ProxyArray:
        """Quaternion orientation ``(x, y, z, w)`` of the sensor origin in world frame,
        following the world coordinate frame convention.

        .. note::
            World frame convention follows the camera aligned with forward axis +X and up axis +Z.

        Shape is (N,), dtype ``wp.quatf``. In torch this resolves to (N, 4),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._quat_w_world

    ##
    # Camera data
    ##

    @property
    def intrinsic_matrices(self) -> ProxyArray:
        """The intrinsic matrices for the camera.

        Shape is (N,), dtype ``wp.mat33f``. In torch this resolves to (N, 3, 3),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._intrinsic_matrices

    @property
    def output(self) -> dict[str, ProxyArray] | None:
        """The retrieved sensor data with sensor types as key.

        Each value is a :class:`~isaaclab.utils.warp.ProxyArray` of shape
        ``(N, H, W, C)`` where N is the number of views, H/W are image dimensions,
        and C is the number of channels. Use ``.torch`` for a ``torch.Tensor`` view
        or ``.warp`` for the underlying ``wp.array``.

        The format of the data is available in the `Replicator Documentation`_. For semantic-based data,
        this corresponds to the ``"data"`` key in the output of the sensor.

        .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
        """
        return self._output

    def create_buffers(self, num_views: int, device: str) -> None:
        """Allocate warp arrays for pose and intrinsics and create their :class:`ProxyArray` wrappers.

        Called by :class:`~isaaclab.sensors.camera.Camera` after :meth:`allocate` to
        populate the pose and intrinsics buffers.

        Args:
            num_views: Number of camera views (batch dimension).
            device: Device for tensor storage (e.g. ``"cuda:0"``).
        """
        self._pos_w = ProxyArray(wp.zeros(num_views, dtype=wp.vec3f, device=device))
        self._quat_w_world = ProxyArray(wp.zeros(num_views, dtype=wp.quatf, device=device))
        self._intrinsic_matrices = ProxyArray(wp.zeros(num_views, dtype=wp.mat33f, device=device))
        self._quat_w_ros = ProxyArray(wp.zeros(num_views, dtype=wp.quatf, device=device))
        self._quat_w_opengl = ProxyArray(wp.zeros(num_views, dtype=wp.quatf, device=device))

    @classmethod
    def allocate(
        cls,
        data_types: list[str],
        height: int,
        width: int,
        num_views: int,
        device: str,
        supported_specs: dict[RenderBufferKind, RenderBufferSpec],
    ) -> CameraData:
        """Build a :class:`CameraData` with output buffers pre-allocated as warp arrays.

        Allocates one ``(num_views, height, width, channels)`` warp array per kind
        in the intersection of ``data_types`` and ``supported_specs``, using
        the channels and dtype from each :class:`RenderBufferSpec`. Each buffer is
        wrapped in a :class:`~isaaclab.utils.warp.ProxyArray`; call ``.torch`` on
        the result to obtain a zero-copy :class:`torch.Tensor` view.

        Args:
            data_types: Requested output names (typically :attr:`CameraCfg.data_types`).
                Every name must be a member of :class:`RenderBufferKind`.
            height: Image height in pixels.
            width: Image width in pixels.
            num_views: Number of camera views (batch dimension).
            device: Device on which to allocate the buffers.
            supported_specs: Per-buffer layout the active renderer can produce,
                keyed by :class:`RenderBufferKind`. Names absent from this mapping
                are not allocated.

        Returns:
            A new :class:`CameraData` with :attr:`image_shape`, :attr:`output`,
            and :attr:`info` populated; pose/intrinsic buffers must be created
            separately via :meth:`create_buffers`.

        Raises:
            ValueError: If ``data_types`` contains names that are not members of
                :class:`RenderBufferKind`.
        """
        valid_names = {kind.value for kind in RenderBufferKind}
        unknown = [name for name in data_types if name not in valid_names]
        if unknown:
            raise ValueError(f"Unknown RenderBufferKind name(s): {unknown}. Expected members of RenderBufferKind.")
        requested = {RenderBufferKind(name) for name in data_types}

        rgb_kinds = {RenderBufferKind.RGB, RenderBufferKind.RGBA}
        rgb_alias = rgb_kinds <= supported_specs.keys() and not requested.isdisjoint(rgb_kinds)
        if rgb_alias:
            requested.update(rgb_kinds)

        allocated = requested.intersection(supported_specs)
        if rgb_alias:
            allocated.remove(RenderBufferKind.RGB)

        buffers: dict[str, ProxyArray] = {}
        for name, spec in supported_specs.items():
            if name not in allocated:
                continue
            shape = (num_views, height, width, spec.channels)
            buffers[str(name)] = ProxyArray(wp.zeros(shape, dtype=spec.dtype, device=device))

        if rgb_alias:
            # Zero-copy strided view into rgba: shape (N, H, W, 3), skipping the alpha channel.
            # Byte strides for a contiguous (N, H, W, 4) uint8 array are (H*W*4, W*4, 4, 1).
            # Using the same outer strides but limiting the last dim to 3 channels gives a
            # non-contiguous view where each pixel reads RGB without the alpha byte.
            rgba_wp = buffers[str(RenderBufferKind.RGBA)].warp
            rgb_wp = wp.array(
                ptr=rgba_wp.ptr,
                shape=(num_views, height, width, 3),
                strides=(height * width * 4, width * 4, 4, 1),
                dtype=wp.uint8,
                device=rgba_wp.device,
                copy=False,
            )
            buffers[str(RenderBufferKind.RGB)] = ProxyArray(rgb_wp)

        obj = cls()
        obj.image_shape = (height, width)
        obj._output = buffers
        obj.info = {name: None for name in buffers}
        return obj

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> ProxyArray:
        """Quaternion orientation ``(x, y, z, w)`` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.

        Shape is (N,), dtype ``wp.quatf``. In torch this resolves to (N, 4),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        convert_camera_frame_orientation_convention_wp(self._quat_w_world.warp, self._quat_w_ros.warp, "world", "ros")
        return self._quat_w_ros

    @property
    def quat_w_opengl(self) -> ProxyArray:
        """Quaternion orientation ``(x, y, z, w)`` of the sensor origin in the world frame, following
        Opengl / USD Camera convention.

        .. note::
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.

        Shape is (N,), dtype ``wp.quatf``. In torch this resolves to (N, 4),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        convert_camera_frame_orientation_convention_wp(
            self._quat_w_world.warp, self._quat_w_opengl.warp, "world", "opengl"
        )
        return self._quat_w_opengl
