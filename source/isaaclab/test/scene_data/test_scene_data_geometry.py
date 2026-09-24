# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native geometry publication, interpolation, and direct destination writes."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.scene_data_backend import SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider


def test_geometry_views_alias_native_ranges_and_follow_pointer_swaps(monkeypatch):
    """Native geometry skips copies, including padded ranges and published pointer swaps."""
    source = SceneDataFormat.Points()
    source.points = wp.array(np.arange(18).reshape(6, 3), dtype=wp.vec3f, device="cpu")
    replacement = wp.array(np.arange(18, 36).reshape(6, 3), dtype=wp.vec3f, device="cpu")
    ranges = {"/Cloth/visual": (1, 2), "/Particles/points": (4, 2)}
    backend = SimpleNamespace(geometry_version=0, get_geometry_batches=lambda _: [(source, ranges)])
    provider = SceneDataProvider(backend)
    launch, copy = Mock(wraps=wp.launch), Mock(wraps=wp.copy)
    monkeypatch.setattr(wp, "launch", launch)
    monkeypatch.setattr(wp, "copy", copy)
    views = provider.get_geometry_points()
    assert provider.get_geometry_points() is views
    for path, (offset, count) in ranges.items():
        assert views[path].ptr == source.points.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f)
        np.testing.assert_array_equal(views[path].numpy(), source.points.numpy()[offset : offset + count])
    previous = views["/Cloth/visual"]
    source.points = replacement
    backend.geometry_version += 1
    updated = provider.get_geometry_points()
    assert updated["/Cloth/visual"].ptr != previous.ptr
    np.testing.assert_array_equal(updated["/Cloth/visual"].numpy(), [[21, 22, 23], [24, 25, 26]])
    launch.assert_not_called()
    copy.assert_not_called()
    output = wp.zeros(4, dtype=wp.vec3f, device="cpu")
    offsets = {"/Particles/points": 0, "/Cloth/visual": 2}
    reordered = provider.get_geometry_points(output=output, offsets=offsets)
    assert reordered is output
    assert provider.get_geometry_points(output=output, offsets=offsets) is reordered
    assert launch.call_count == 1
    np.testing.assert_array_equal(output.numpy(), replacement.numpy()[[4, 5, 1, 2]])
    released = weakref.ref(output)
    launch.reset_mock()
    del reordered, output
    gc.collect()
    assert released() is None


@pytest.mark.parametrize("fabric", [False, True])
@pytest.mark.parametrize(
    "source_device,destination_device", [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")]
)
def test_geometry_interpolation_and_reordering_write_once_per_version(
    monkeypatch, fabric, source_device, destination_device
):
    """One conversion writes barycentric vertices directly into the consumer's reordered layout."""
    if "cuda:0" in (source_device, destination_device) and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    source = SceneDataFormat.WeightedPoints()
    nodes = np.arange(18, dtype=np.float32).reshape(6, 3)
    indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)
    weights = np.array([[0.25] * 4, [1, 0, 0, 0], [0.5, 0.25, 0.25, 0]], dtype=np.float32)
    source.points = wp.array(nodes, dtype=wp.vec3f, device=source_device)
    source.indices = wp.array(indices, dtype=wp.int32, device=source_device)
    source.weights = wp.array(weights, dtype=wp.float32, device=source_device)
    ranges = {"/First/visual": (0, 2), "/Second/visual": (2, 1)}
    backend = SimpleNamespace(geometry_version=0, get_geometry_batches=lambda _: [(source, ranges)])
    provider = SceneDataProvider(backend)
    output = wp.full(6, wp.vec3f(-7), device=destination_device)
    offsets = {"/Second/visual": 0, "/First/visual": 3}
    destination = output
    if fabric:
        pointers = wp.array([output.ptr, output[3:].ptr], dtype=wp.uint64, device=destination_device)
        lengths = wp.array([1, 2], dtype=wp.uint64, device=destination_device)
        storage = SimpleNamespace(
            __fabric_arrays_interface__={
                "version": 1,
                "device": destination_device,
                "attribs": {
                    "points": {
                        "type": (True, "f4", 3, 1, "vector"),
                        "access": 2,
                        "pointers": [pointers.ptr],
                        "counts": [2],
                        "array_lengths": [lengths.ptr],
                    }
                },
            }
        )
        destination = SceneDataFormat.FabricPoints()
        destination.points = wp.fabricarrayarray(data=storage, attrib="points", dtype=wp.vec3f)
        offsets = {"/Second/visual": 0, "/First/visual": 1}
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
    assert provider.get_geometry_points(output=destination, offsets=offsets) is destination
    assert provider.get_geometry_points(output=destination, offsets=offsets) is destination
    destination_launches = 1 + (source_device != destination_device)
    assert launch.call_count == destination_launches
    expected = np.full((6, 3), -7, dtype=np.float32)
    interpolated = (nodes[indices] * weights[..., None]).sum(axis=1)
    expected[0], expected[3:5] = interpolated[2], interpolated[:2]
    np.testing.assert_allclose(output.numpy(), expected)
    shared = provider.get_geometry_points()
    assert provider.get_geometry_points() is shared
    assert launch.call_count == destination_launches + 1
    np.testing.assert_allclose(shared["/First/visual"].numpy(), interpolated[:2])
    source.points.assign(nodes + 10)
    backend.geometry_version += 1
    assert provider.get_geometry_points(output=destination, offsets=offsets) is destination
    assert provider.get_geometry_points() is shared
    assert launch.call_count == 2 * (destination_launches + 1)
    expected[[0, 3, 4]] += 10
    np.testing.assert_allclose(output.numpy(), expected)
    np.testing.assert_allclose(shared["/Second/visual"].numpy(), interpolated[2:] + 10)
    if fabric:
        provider = SceneDataProvider(
            SimpleNamespace(geometry_version=0, get_geometry_batches=lambda _: [(destination, {})])
        )
        native = SceneDataFormat.FabricPoints()
        launch.reset_mock()
        assert provider.get_geometry_points(output=native) is native
        assert native.points is destination.points
        launch.assert_not_called()


def test_cable_endpoint_conversion_composes_shape_poses_and_averages_joints():
    """Two differently oriented capsules produce the two ends and averaged connecting vertex."""
    source = SceneDataFormat.CapsuleEndpoints()
    sine = np.sqrt(0.5)
    source.transforms = wp.array(
        [[2, 3, 3, sine, 0, 0, sine], [1, 2, 3, 0, sine, 0, sine]], dtype=wp.transformf, device="cpu"
    )
    source.shape_body = wp.array([1, 0], dtype=wp.int32, device="cpu")
    source.shape_transform = wp.array(
        [[0, 0, 1, 0, 0, 0, 1], [0, 0, 0.5, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu"
    )
    source.shape_scale = wp.array([[0.1, 1, 0.1], [0.1, 2, 0.1]], dtype=wp.vec3f, device="cpu")
    source.endpoints = wp.array([[0, -1, 0, -1], [0, 1, 1, -1], [1, 1, 1, 1]], dtype=wp.vec4i, device="cpu")
    backend = SimpleNamespace(geometry_version=0, get_geometry_batches=lambda _: [(source, {"/Cable/curve": (0, 3)})])
    points = SceneDataProvider(backend).get_geometry_points()["/Cable/curve"]
    np.testing.assert_allclose(points.numpy(), [[1, 2, 3], [2.5, 3.25, 3], [2, 0.5, 3]], atol=1e-6)
