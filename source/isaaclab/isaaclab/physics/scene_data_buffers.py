# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-allocated GPU buffer pool for transform format conversion.

Provides :class:`TransformBufferPool`, which manages pre-allocated GPU
buffers for format conversion results and implements the three fast paths:
format-match passthrough, same-frame cache, and cross-frame buffer reuse.
"""

from __future__ import annotations

import warp as wp

from .scene_data_conversion import ConversionDispatcher
from .scene_data_types import (
    Mat44Transforms,
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformData,
    TransformFormat,
    Vec3Mat33Transforms,
    Vec3QuatTransforms,
)


def _allocate_transform_data(
    fmt: TransformFormat,
    count: int,
    device: str,
    *,
    quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
    matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
    double_precision: bool = False,
) -> TransformData:
    """Allocate a :class:`TransformData` with zeroed GPU buffers."""
    if fmt == TransformFormat.VEC3_QUAT:
        return Vec3QuatTransforms(
            count=count,
            device=device,
            positions=wp.zeros(count, dtype=wp.vec3, device=device),
            orientations=wp.zeros(count, dtype=wp.quatf, device=device),
            quat_convention=quat_convention,
        )
    if fmt == TransformFormat.VEC3_MAT33:
        return Vec3Mat33Transforms(
            count=count,
            device=device,
            positions=wp.zeros(count, dtype=wp.vec3, device=device),
            rotations=wp.zeros(count, dtype=wp.mat33f, device=device),
            layout=matrix_layout,
        )
    if fmt == TransformFormat.TRANSFORM:
        return TransformArrayData(
            count=count,
            device=device,
            transforms=wp.zeros(count, dtype=wp.transformf, device=device),
        )
    if fmt == TransformFormat.MAT44:
        mat_dtype = wp.mat44d if double_precision else wp.mat44f
        return Mat44Transforms(
            count=count,
            device=device,
            matrices=wp.zeros(count, dtype=mat_dtype, device=device),
            layout=matrix_layout,
            double_precision=double_precision,
        )
    raise ValueError(f"Unknown format: {fmt}")


class TransformBufferPool:
    """Pre-allocated GPU buffer pool for transform format conversions.

    Manages output buffers keyed by conversion parameters and implements
    generation-based caching to avoid redundant conversions within a frame.

    Three fast paths are supported:

    1. **Format-match passthrough** -- when the requested format matches the
       source format and ``allow_passthrough=True``, the source data is
       returned directly (zero work).
    2. **Same-frame cache** -- if a conversion was already performed this
       frame (same generation), the cached result is returned.
    3. **Cross-frame reuse** -- output buffers persist across frames;
       only the conversion kernel is re-launched (no allocation).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, TransformData] = {}
        self._cache_generation: dict[tuple, int] = {}

    def _cache_key(
        self,
        target_format: TransformFormat,
        quat_convention: QuaternionConvention,
        matrix_layout: MatrixLayout,
        double_precision: bool,
        index_map_id: int | None,
    ) -> tuple:
        return (target_format, quat_convention, matrix_layout, double_precision, index_map_id)

    def get_or_convert(
        self,
        source: TransformData,
        target_format: TransformFormat,
        *,
        generation: int,
        quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
        matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
        double_precision: bool = False,
        stream: wp.Stream | None = None,
        allow_passthrough: bool = True,
        index_map: wp.array | None = None,
    ) -> TransformData:
        """Return transform data in the requested format.

        Uses cached results when possible; allocates and converts when not.

        Args:
            source: Source transform data from the simulator.
            target_format: Desired output format.
            generation: Frame generation counter. When this matches the
                cached generation, the cached result is returned.
            quat_convention: Desired quaternion convention.
            matrix_layout: Desired matrix layout.
            double_precision: Whether MAT44 output should use float64.
            stream: Optional CUDA stream for deferred kernel execution.
            allow_passthrough: If ``True`` and formats match, return source
                directly (zero-copy). Consumer must not mutate the result.
            index_map: Optional index remapping for subset scatter writes.

        Returns:
            Transform data in the requested format.
        """
        # Fast path 1: format-match passthrough
        if (
            allow_passthrough
            and index_map is None
            and ConversionDispatcher.can_passthrough(
                source,
                target_format,
                quat_convention=quat_convention,
                matrix_layout=matrix_layout,
                double_precision=double_precision,
            )
        ):
            return source

        index_map_id = id(index_map) if index_map is not None else None
        key = self._cache_key(target_format, quat_convention, matrix_layout, double_precision, index_map_id)

        # Fast path 2: same-frame cache hit
        if key in self._cache and self._cache_generation.get(key) == generation:
            cached = self._cache[key]
            if cached.count == source.count:
                return cached

        # Fast path 3: cross-frame buffer reuse (or allocate if size changed)
        target_count = source.count
        if index_map is not None:
            target_count = int(index_map.shape[0])
        existing = self._cache.get(key)
        if existing is not None and existing.count == target_count:
            target = existing
        else:
            target = _allocate_transform_data(
                target_format,
                target_count,
                source.device,
                quat_convention=quat_convention,
                matrix_layout=matrix_layout,
                double_precision=double_precision,
            )

        ConversionDispatcher.convert(source, target, stream=stream, index_map=index_map)

        self._cache[key] = target
        self._cache_generation[key] = generation
        return target

    def clear(self) -> None:
        """Release all cached buffers."""
        self._cache.clear()
        self._cache_generation.clear()
