# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split a tiled OVRTX lidar point cloud into per-environment buffers."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import warp as wp

from isaaclab_ov.renderers.ovrtx_renderer_kernels import copy_dense_lidar_points_kernel

# Bit 6 of the PointCloud Flags channel. See omni::sensors::ElementFlags::VALID.
_VALID_FLAG = 0x40
# rtx::rtxsensor::SensorFrameData::CompactionType::eCompacted
_COMPACTED = 2
# PointCloud coordsType. Spherical Coordinates contain azimuth [deg], elevation [deg], and range [m].
_SPHERICAL = 1


def _as_torch(value, *, clone: bool = True) -> torch.Tensor:
    """Return a Torch view of a DLPack tensor.

    A cloned tensor outlives the mapped render variable. A CUDA view is only valid until the caller
    releases the map on the stream that consumed it.
    """
    try:
        tensor = torch.from_dlpack(value)
    except (TypeError, BufferError, RuntimeError):
        tensor = torch.as_tensor(np.from_dlpack(value))
    if clone:
        return tensor.detach().clone()
    return tensor


def _scalar(mapped, name: str) -> int | None:
    """Read one integer point-cloud parameter. Parameters are always on the CPU."""
    params = mapped.params
    if name not in params:
        return None
    values = np.from_dlpack(params[name]).reshape(-1)
    if values.size == 0:
        return None
    return int(values[0])


def _consuming_torch_stream(device: str):
    """Run Torch ops on the Warp stream that will consume the mapped views.

    A CUDA stream handle of 0 is the legacy default stream and must be preserved. ``None`` means
    Warp has no CUDA stream on this device.
    """
    cuda_stream = wp.get_stream(device).cuda_stream
    if cuda_stream is None:
        return contextlib.nullcontext()
    return torch.cuda.stream(torch.cuda.ExternalStream(cuda_stream, device=device))


def _channel(mapped, name: str, *, clone: bool = True) -> torch.Tensor | None:
    """Return one channel tensor, or None when the point cloud did not publish it."""
    if name not in mapped:
        return None
    return _as_torch(mapped[name], clone=clone)


def copy_tiled_point_cloud(mapped, points: torch.Tensor, intensity: torch.Tensor, valid: torch.Tensor) -> None:
    """Copy one mapped point cloud into per-environment buffers.

    ``points`` has shape ``(num_envs, num_rays, 3)``. Tile ``i`` is environment ``i``, which is the
    render-product relationship order. Misses stay non-finite. A dense cloud must contain one slot
    per authored ray. A compacted cloud is rejected because it does not identify the authored ray.

    Args:
        mapped: Mapped PointCloud render variable. CUDA clouds must stay mapped until the
            consuming stream completes.
        points: Destination positions.
        intensity: Destination intensity, shape ``(num_envs, num_rays)``.
        valid: Destination hit mask, shape ``(num_envs, num_rays)``.

    Raises:
        RuntimeError: If the tile count, layout, or ray count does not match the destination.
    """
    num_envs, num_rays, _ = points.shape
    tile_count = _scalar(mapped, "tileCount")
    if tile_count is None:
        tile_count = num_envs
    if tile_count != num_envs:
        raise RuntimeError(f"OVRTX lidar returned {tile_count} tiles for {num_envs} environments.")

    coordinates = _channel(mapped, "Coordinates", clone=False)
    flags = _channel(mapped, "Flags", clone=False)
    signal = _channel(mapped, "Intensity", clone=False)

    on_device = (
        coordinates is not None and flags is not None and signal is not None and coordinates.device.type == "cuda"
    )
    if not on_device:
        coordinates = None if coordinates is None else coordinates.detach().clone()
        flags = None if flags is None else flags.detach().clone()
        signal = None if signal is None else signal.detach().clone()

    if coordinates is None or flags is None or signal is None:
        available = list(mapped.keys())
        raise RuntimeError(
            f"OVRTX lidar point cloud is missing Coordinates, Intensity, or Flags. Available: {available}."
        )

    # CUDA layout fixes and the copy kernel must share Warp's stream so unmap(stream=) can
    # wait for every read of the mapped views.
    stream_ctx = _consuming_torch_stream(str(points.device)) if on_device else contextlib.nullcontext()
    with stream_ctx:
        if coordinates.shape[-1] != 3:
            if coordinates.shape[0] == 3:
                coordinates = coordinates.transpose(0, 1).contiguous()
            else:
                raise RuntimeError(f"OVRTX lidar Coordinates have shape {tuple(coordinates.shape)}, expected (*, 3).")
        coordinates = coordinates.reshape(-1, 3)
        flags = flags.reshape(-1)
        signal = signal.reshape(-1)
        if coordinates.shape[0] != flags.shape[0] or coordinates.shape[0] != signal.shape[0]:
            raise RuntimeError(
                "OVRTX lidar point-cloud channels disagree:"
                f" Coordinates {coordinates.shape[0]}, Flags {flags.shape[0]}, Intensity {signal.shape[0]}."
            )

        device = points.device
        stride = _scalar(mapped, "tileStridePoints") or 0

        if _scalar(mapped, "compactionType") == _COMPACTED:
            raise RuntimeError(
                "OVRTX returned a compacted lidar cloud. Missed rays must keep their authored slots"
                " until each return can be placed by ray id."
            )

        if stride != num_rays:
            raise RuntimeError(
                f"OVRTX lidar tile has {stride} point slots, but the sensor authored {num_rays}. "
                "include_invalid_points keeps one slot per ray."
            )

        if coordinates.shape[0] != tile_count * stride:
            raise RuntimeError(
                f"OVRTX lidar returned {coordinates.shape[0]} points for {tile_count} tiles of {stride} rays."
            )

        coords_type = _scalar(mapped, "coordsType")
        if coords_type not in (0, _SPHERICAL):
            raise RuntimeError(f"Unsupported OvRTX lidar coordsType: {coords_type!r}.")
        spherical = coords_type == _SPHERICAL
        if on_device:
            _copy_dense_on_device(coordinates, signal, flags, points, intensity, valid, tile_count, stride, spherical)
        else:
            coordinates = coordinates.to(device=device, dtype=points.dtype)
            if spherical:
                azimuth, elevation = torch.deg2rad(coordinates[:, :2]).unbind(dim=1)
                distance = coordinates[:, 2]
                horizontal = distance * elevation.cos()
                coordinates = torch.stack(
                    (horizontal * azimuth.cos(), horizontal * azimuth.sin(), distance * elevation.sin()), dim=1
                )
            signal = signal.to(device=device, dtype=intensity.dtype)
            hit_source = (flags.to(device=device, dtype=torch.int64) & _VALID_FLAG) != 0

            _copy_dense(coordinates, signal, hit_source, points, intensity, valid, tile_count, stride)


def _copy_dense(coordinates, signal, hits, points, intensity, valid, tile_count: int, stride: int) -> None:
    """Copy a dense ``(tile, ray)`` cloud. ``hits`` selects the finite returns."""
    coordinates = coordinates.reshape(tile_count, stride, 3)
    signal = signal.reshape(tile_count, stride)
    hits = hits.reshape(tile_count, stride)
    filled = torch.where(hits.unsqueeze(-1), coordinates, torch.full_like(coordinates, float("inf")))
    signal = torch.where(hits, signal, torch.zeros_like(signal))
    points.copy_(filled)
    intensity.copy_(signal)
    valid.copy_(hits)


def _copy_dense_on_device(
    coordinates, signal, flags, points, intensity, valid, tile_count: int, stride: int, spherical: bool
) -> None:
    """Copy a dense CUDA ``(tile, ray)`` cloud into the destination buffers.

    Torch prep and the Warp launch share Warp's current stream. The caller releases the map on that
    stream; this does not wait for the launch.
    """
    count = tile_count * stride
    launch_device = str(points.device)
    with _consuming_torch_stream(launch_device):
        coordinates = coordinates.reshape(count, 3).contiguous()
        signal = signal.reshape(count).contiguous().to(dtype=torch.float32)
        flags = flags.reshape(count).contiguous().to(dtype=torch.int32)
        # Warp resolves string device names. A torch.device object is not one of those names.
        wp.launch(
            copy_dense_lidar_points_kernel,
            dim=count,
            inputs=[
                wp.from_torch(coordinates, dtype=wp.float32),
                wp.from_torch(signal, dtype=wp.float32),
                wp.from_torch(flags, dtype=wp.int32),
                wp.from_torch(points.reshape(count, 3), dtype=wp.vec3f),
                wp.from_torch(intensity.reshape(count), dtype=wp.float32),
                wp.from_torch(valid.reshape(count), dtype=wp.bool),
                spherical,
            ],
            device=launch_device,
        )
