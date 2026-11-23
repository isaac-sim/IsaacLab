# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused geometry interpolation and native destination writes for SDP."""

from __future__ import annotations

from typing import Any

import warp as wp

from .scene_data_backend import SceneDataFormat


@wp.func
def geometry_point(source: SceneDataFormat.Points, index: int) -> wp.vec3f:
    return source.points[index]


@wp.func
def geometry_point(source: SceneDataFormat.WeightedPoints, index: int) -> wp.vec3f:  # noqa: F811
    value = wp.vec3f()
    for corner in range(4):
        value += source.points[source.indices[index, corner]] * source.weights[index, corner]
    return value


@wp.func
def geometry_point(source: SceneDataFormat.CapsuleEndpoints, index: int) -> wp.vec3f:  # noqa: F811
    ends = source.endpoints[index]
    left = wp.transform_multiply(source.transforms[source.shape_body[ends[0]]], source.shape_transform[ends[0]])
    right = wp.transform_multiply(source.transforms[source.shape_body[ends[2]]], source.shape_transform[ends[2]])
    a = wp.transform_point(left, wp.vec3f(0.0, 0.0, float(ends[1]) * source.shape_scale[ends[0]][1]))
    b = wp.transform_point(right, wp.vec3f(0.0, 0.0, float(ends[3]) * source.shape_scale[ends[2]][1]))
    return (a + b) * 0.5


@wp.kernel(enable_backward=False)
def convert_geometry_points_kernel(
    source: Any,
    source_indices: wp.array(dtype=wp.int32),
    destination_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.vec3f),
):
    index = wp.tid()
    source_index = source_indices[index] if source_indices.shape[0] else index
    destination_index = destination_indices[index] if destination_indices.shape[0] else index
    output[destination_index] = geometry_point(source, source_index)


@wp.kernel(enable_backward=False)
def convert_geometry_fabric_kernel(
    source: Any,
    source_indices: wp.array(dtype=wp.int32),
    destination_indices: wp.array(dtype=wp.vec2i),
    output: wp.fabricarrayarray(dtype=wp.vec3f),
):
    index = wp.tid()
    destination = destination_indices[index]
    source_index = source_indices[index] if source_indices.shape[0] else index
    output[destination[0]][destination[1]] = geometry_point(source, source_index)
