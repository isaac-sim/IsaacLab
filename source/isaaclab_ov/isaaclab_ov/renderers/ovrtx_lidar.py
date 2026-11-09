# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split a tiled OVRTX lidar point cloud into per-environment buffers."""

from __future__ import annotations

import numpy as np
import torch

# Bit 6 of the PointCloud Flags channel. See omni::sensors::ElementFlags::VALID.
_VALID_FLAG = 0x40
# rtx::rtxsensor::SensorFrameData::CompactionType::eCompacted
_COMPACTED = 2


def _as_torch(value) -> torch.Tensor:
    """Copy a DLPack tensor into a Torch tensor that outlives the mapped render variable."""
    try:
        tensor = torch.from_dlpack(value)
    except (TypeError, BufferError, RuntimeError):
        tensor = torch.as_tensor(np.from_dlpack(value))
    return tensor.detach().clone()


def _scalar(mapped, name: str) -> int | None:
    """Read one integer point-cloud parameter. Parameters are always on the CPU."""
    params = mapped.params
    if name not in params:
        return None
    values = np.from_dlpack(params[name]).reshape(-1)
    if values.size == 0:
        return None
    return int(values[0])


def _channel(mapped, name: str) -> torch.Tensor | None:
    """Return a cloned channel tensor, or None when the point cloud did not publish it."""
    if name not in mapped:
        return None
    return _as_torch(mapped[name])


def copy_tiled_point_cloud(mapped, points: torch.Tensor, intensity: torch.Tensor, valid: torch.Tensor) -> None:
    """Copy one mapped WPM point cloud into per-environment buffers.

    ``points`` has shape ``(num_envs, num_rays, 3)``. Tile ``i`` is environment ``i``, which is the
    render-product relationship order. Misses stay non-finite. A dense cloud must contain one slot
    per authored ray; a compacted cloud is scattered into the leading slots and the tail is a miss.

    Args:
        mapped: Mapped PointCloud render variable. It must stay mapped for this call.
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

    coordinates = _channel(mapped, "Coordinates")
    flags = _channel(mapped, "Flags")
    signal = _channel(mapped, "Intensity")
    if coordinates is None or flags is None or signal is None:
        available = list(mapped.keys())
        raise RuntimeError(
            f"OVRTX lidar point cloud is missing Coordinates, Intensity, or Flags. Available: {available}."
        )
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
    coordinates = coordinates.to(device=device, dtype=points.dtype)
    signal = signal.to(device=device, dtype=intensity.dtype)
    hit_source = (flags.to(device=device, dtype=torch.int64) & _VALID_FLAG) != 0
    stride = _scalar(mapped, "tileStridePoints") or 0
    compacted = _scalar(mapped, "compactionType") == _COMPACTED
    if not compacted and stride == num_rays:
        _copy_dense(coordinates, signal, hit_source, points, intensity, valid, tile_count, stride)
        return
    if not compacted and stride > 0:
        raise RuntimeError(
            f"OVRTX lidar tile has {stride} point slots, but the sensor authored {num_rays}. "
            "include_invalid_points keeps one slot per ray."
        )
    offsets = _channel(mapped, "TilePointOffsets")
    if offsets is None:
        raise RuntimeError("OVRTX lidar returned a compacted point cloud without TilePointOffsets.")
    _copy_compacted(coordinates, signal, hit_source, offsets.to(device=device), points, intensity, valid, num_rays)


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


def _copy_compacted(coordinates, signal, hits, offsets, points, intensity, valid, num_rays: int) -> None:
    """Scatter packed returns into the leading ray slots. Unused slots stay misses."""
    offsets = offsets.reshape(-1).to(dtype=torch.int64)
    filled = torch.full_like(points, float("inf"))
    signal_out = torch.zeros_like(intensity)
    valid_out = torch.zeros_like(valid)
    for index in range(points.shape[0]):
        start, end = int(offsets[index]), int(offsets[index + 1])
        count = end - start
        if count > num_rays:
            raise RuntimeError(f"OVRTX lidar tile {index} has {count} points, more than the {num_rays} ray slots.")
        if count <= 0:
            continue
        tile_hits = hits[start:end]
        filled[index, :count] = torch.where(
            tile_hits.unsqueeze(-1), coordinates[start:end], torch.full((count, 3), float("inf"), device=points.device)
        )
        signal_out[index, :count] = torch.where(tile_hits, signal[start:end], torch.zeros(count, device=points.device))
        valid_out[index, :count] = tile_hits
    points.copy_(filled)
    intensity.copy_(signal_out)
    valid.copy_(valid_out)
