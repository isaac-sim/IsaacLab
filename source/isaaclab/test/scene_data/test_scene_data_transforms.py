# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneDataProvider transform conversion and index mapping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.cloner.usd import UsdReplicateContext
from isaaclab.scene_data.scene_data_backend import SceneDataFormat, SceneDataPublication
from isaaclab.scene_data.scene_data_provider import SceneDataProvider


@pytest.mark.skipif(
    wp.get_cuda_device_count() == 0, reason="requires a CUDA device to reproduce the default-device mismatch"
)
def test_get_transforms_matches_backend_device_when_warp_default_is_cuda():
    """Transform conversion must follow its CPU publication, not Warp's CUDA default."""
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array(
        [[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for x in range(3)], dtype=wp.transformf, device="cpu"
    )
    provider = SceneDataProvider(
        SimpleNamespace(
            transforms=transforms,
            transform_count=3,
            transform_paths=["/World/a", "/World/b", "/World/c"],
        )
    )

    with wp.ScopedDevice("cuda:0"):
        mapping = provider.create_mapping(["/World/c", "/World/a", "/World/b"])
        assert mapping is not None
        assert str(mapping.device) == "cpu"

        output = SceneDataFormat.Vec3_Quat()
        assert provider.get_transforms(output, mapping=mapping, allow_passthrough=False)

    assert str(output.positions.device) == "cpu"
    assert str(output.orientations.device) == "cpu"
    assert np.allclose(output.positions.numpy()[:, 0], [2.0, 0.0, 1.0])


def test_publication_aliases_native_pointer_and_converts_once_per_write(monkeypatch):
    """Clean requests share one conversion; writes and native buffer swaps invalidate it."""
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    publication = SceneDataPublication(data)
    provider = SceneDataProvider(SimpleNamespace(transform_publication=publication, transform_count=1))
    with pytest.raises(ValueError, match="destination count"):
        provider.request_transforms(SceneDataFormat.Transform, count=2)
    launch = wp.launch
    calls = []

    def record_launch(*args, **kwargs):
        calls.append(kwargs.get("kernel", args[0] if args else None))
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_launch)
    assert provider.request_transforms(SceneDataFormat.Transform).transforms is data.transforms
    assert calls == []
    converted = provider.request_transforms(SceneDataFormat.Vec3_Quat)
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert len(calls) == 1
    np.testing.assert_array_equal(converted.positions.numpy(), [[1, 2, 3]])

    data.transforms.assign([[4, 5, 6, 0, 0, 0, 1]])
    publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert len(calls) == 2
    np.testing.assert_array_equal(converted.positions.numpy(), [[4, 5, 6]])

    data.transforms = wp.array([[7, 8, 9, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.Transform).transforms is data.transforms
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert len(calls) == 3
    np.testing.assert_array_equal(converted.positions.numpy(), [[7, 8, 9]])


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat", "Vec3_Matrix33", "Matrix44"])
@pytest.mark.parametrize("scaled", [False, True])
def test_transposed_matrices_fuse_format_mapping_and_scale(format_name, scaled):
    """All native formats produce the same row-vector matrices, with output-indexed scale."""
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]], dtype=np.float32)
    rotations = np.array([np.eye(3), np.diag([-1, -1, 1])], dtype=np.float32)
    matrices = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = poses[:, :3]
    data = getattr(SceneDataFormat, format_name)()
    if format_name == "Transform":
        data.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    elif format_name == "Matrix44":
        data.matrices = wp.array(matrices, dtype=wp.mat44f, device="cpu")
    else:
        data.positions = wp.array(poses[:, :3], dtype=wp.vec3f, device="cpu")
        data.orientations = wp.array(
            poses[:, 3:] if format_name == "Vec3_Quat" else rotations,
            dtype=wp.quatf if format_name == "Vec3_Quat" else wp.mat33f,
            device="cpu",
        )
    provider = SceneDataProvider(SimpleNamespace(transform_publication=SceneDataPublication(data), transform_count=2))
    mapping = wp.array([1, 0], dtype=wp.int32, device="cpu")
    scales = wp.array([[2, 3, 4], [5, 6, 7]], dtype=wp.vec3f, device="cpu") if scaled else None
    output = provider.request_transforms(SceneDataFormat.TransposedMatrix44d, mapping, scales=scales)
    expected = matrices[::-1].transpose(0, 2, 1).copy()
    if scaled:
        expected[:, :3, :3] *= scales.numpy()[:, :, None]
    np.testing.assert_allclose(output.matrices.numpy(), expected)
    assert provider.request_transforms(SceneDataFormat.TransposedMatrix44d, mapping, scales=scales) is output


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat", "Vec3_Matrix33", "Matrix44"])
@pytest.mark.parametrize("solver_only_body", [False, True])
def test_fabric_conversion_preserves_scale_and_refreshes_reallocated_destinations(
    format_name, solver_only_body, monkeypatch
):
    """Rigid destinations preserve scale and refresh while solver-only cable bodies are excluded."""
    poses = [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]
    paths = ["/World/a", "/World/b"]
    if solver_only_body:
        poses.insert(1, [7, 8, 9, 0, 0, 0, 1])
        paths.insert(1, "/World/cable_edge_body_0")
    data = SceneDataFormat.Transform()
    data.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    native = SceneDataProvider(
        SimpleNamespace(transform_publication=SceneDataPublication(data), transform_count=len(poses))
    )
    publication = SceneDataPublication(native.request_transforms(getattr(SceneDataFormat, format_name)))
    provider = SceneDataProvider(
        SimpleNamespace(
            transform_publication=publication,
            transforms=publication.data,
            transform_count=len(poses),
            transform_paths=paths,
        )
    )
    context = UsdReplicateContext(None)
    context._fabric_provider, context._fabric_device, context._fabric_paths = provider, "cpu", paths
    context._fabric_output = SceneDataFormat.FabricMatrix44()
    expected = np.array([np.diag([-2, -3, 4, 1]), np.diag([5, 6, 7, 1])], dtype=np.float64)
    expected[:, 3, :3] = [[4, 5, 6], [1, 2, 3]]
    launch = wp.launch
    calls = []

    def record_launch(*args, **kwargs):
        calls.append(args[0])
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_launch)
    provider._prepare_fabric_output = context._prepare_fabric_output
    for allocation in range(2):
        matrices = wp.array([np.diag([2, 3, 4, 1]), np.diag([5, 6, 7, 1])], dtype=wp.mat44d, device="cpu")
        interface = {
            "version": 1,
            "device": "cpu",
            "attribs": {
                "omni:fabric:worldMatrix": {
                    "type": (True, "f8", 16, 0, "matrix"),
                    "access": 2,
                    "pointers": [matrices.ptr],
                    "counts": [2],
                }
            },
        }
        changes = [True]
        context._fabric_selection = SimpleNamespace(
            __fabric_arrays_interface__=interface,
            PrepareForReuse=lambda: changes.pop() if changes else False,
            GetPaths=lambda: ["/World/b", "/World/a"],
        )
        output = provider.request_transforms(SceneDataFormat.FabricMatrix44)
        assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
        assert provider.transform_generation == 1
        assert len(calls) == allocation + 1
        np.testing.assert_allclose(matrices.numpy(), expected)


@pytest.mark.parametrize("gpu_options", [None, 3])
def test_fabric_hierarchy_uses_available_sdk_path(gpu_options):
    """Older SDKs retain change tracking; GPU hierarchy updates pause and restore it."""
    calls = []
    context = UsdReplicateContext(None)
    context._fabric_device, context._fabric_generation = "cpu", -1
    context._fabric_update_options = gpu_options
    context._fabric_provider = SimpleNamespace(
        transform_generation=1, request_transforms=lambda _format: calls.append("write")
    )
    context._fabric_hierarchy = SimpleNamespace(update_world_xforms=lambda: calls.append("cpu"))
    if gpu_options is not None:
        context._fabric_hierarchy = SimpleNamespace(
            update_world_xforms_gpu_with_options=lambda options: calls.append(("gpu", options)),
            track_world_xform_changes=lambda active: calls.append(("world", active)),
            track_local_xform_changes=lambda active: calls.append(("local", active)),
        )
    context._update_fabric()
    expected = (
        ["write", "cpu"]
        if gpu_options is None
        else [("world", False), ("local", False), "write", ("gpu", gpu_options), ("world", True), ("local", True)]
    )
    assert calls == expected
    calls.clear()
    context._update_fabric()
    assert calls == [call for call in expected if call not in ("cpu", ("gpu", gpu_options))]
