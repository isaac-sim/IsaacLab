# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneDataProvider transform conversion and index mapping."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp

from pxr import UsdUtils

import isaaclab.scene_data as scene_data
from isaaclab.cloner.usd import UsdReplicateContext
from isaaclab.scene_data.scene_data_backend import SceneDataFormat
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
    assert not hasattr(scene_data, "SceneDataPublication")
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend = SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=1)
    provider = SceneDataProvider(backend)
    with pytest.raises(ValueError, match="destination count"):
        provider.request_transforms(SceneDataFormat.Transform, count=2)
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
    assert provider.request_transforms(SceneDataFormat.Transform).transforms is data.transforms
    launch.assert_not_called()
    converted = provider.request_transforms(SceneDataFormat.Vec3_Quat)
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert launch.call_count == 1
    np.testing.assert_array_equal(converted.positions.numpy(), [[1, 2, 3]])

    data.transforms.assign([[4, 5, 6, 0, 0, 0, 1]])
    backend.transforms_dirty = True
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert launch.call_count == 2
    np.testing.assert_array_equal(converted.positions.numpy(), [[4, 5, 6]])

    data.transforms = wp.array([[7, 8, 9, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend.transforms_dirty = True
    assert provider.request_transforms(SceneDataFormat.Transform).transforms is data.transforms
    assert provider.request_transforms(SceneDataFormat.Vec3_Quat) is converted
    assert launch.call_count == 3
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
    provider = SceneDataProvider(SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=2))
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
@pytest.mark.parametrize("gpu_options", [None, 3])
def test_fabric_conversion_preserves_scale_and_refreshes_reallocated_destinations(
    format_name, solver_only_body, gpu_options, monkeypatch
):
    """Rigid destinations preserve scale and refresh while solver-only cable bodies are excluded."""
    poses = [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]
    paths = ["/World/a", "/World/b"]
    if solver_only_body:
        poses.insert(1, [7, 8, 9, 0, 0, 0, 1])
        paths.insert(1, "/World/cable_edge_body_0")
    data = SceneDataFormat.Transform()
    data.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    native = SceneDataProvider(SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=len(poses)))
    provider = SceneDataProvider(
        SimpleNamespace(
            transforms=native.request_transforms(getattr(SceneDataFormat, format_name)),
            transforms_dirty=True,
            transform_count=len(poses),
            transform_paths=paths,
            fabric=None,
        )
    )
    provider._fabric_device = "cpu"
    provider._fabric_output = SceneDataFormat.FabricMatrix44()
    provider._fabric_update_options = gpu_options
    provider._fabric_hierarchy = Mock()
    expected = np.array([np.diag([-2, -3, 4, 1]), np.diag([5, 6, 7, 1])], dtype=np.float64)
    expected[:, 3, :3] = [[4, 5, 6], [1, 2, 3]]
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
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

        def prepare_for_reuse():
            if gpu_options is not None:
                provider._fabric_hierarchy.track_world_xform_changes.assert_called_with(False)
                provider._fabric_hierarchy.track_local_xform_changes.assert_called_with(False)
            return changes.pop() if changes else False

        provider._fabric_selection = SimpleNamespace(
            __fabric_arrays_interface__=interface,
            PrepareForReuse=prepare_for_reuse,
            GetPaths=lambda: ["/World/b", "/World/a"],
        )
        output = provider.request_transforms(SceneDataFormat.FabricMatrix44)
        if gpu_options is None:
            provider._fabric_hierarchy.update_world_xforms.assert_called_once_with()
        else:
            provider._fabric_hierarchy.update_world_xforms_gpu_with_options.assert_called_once_with(gpu_options)
        provider._fabric_hierarchy.reset_mock()
        assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
        provider._fabric_hierarchy.update_world_xforms.assert_not_called()
        provider._fabric_hierarchy.update_world_xforms_gpu_with_options.assert_not_called()
        assert provider.transform_generation == 1
        assert launch.call_count == 2 * (allocation + 1)
        np.testing.assert_allclose(matrices.numpy(), expected)

    if format_name == "Transform" and not solver_only_body:
        rotations = np.random.default_rng(42).normal(size=(2000, len(poses), 4)).astype(np.float32)
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        poses = np.asarray(poses, dtype=np.float32)
        for rotation in rotations:
            poses[:, 3:] = rotation
            data.transforms.assign(poses)
            provider.backend.transforms_dirty = True
            provider.request_transforms(SceneDataFormat.FabricMatrix44)
        np.testing.assert_allclose(
            np.linalg.norm(matrices.numpy()[:, :3, :3], axis=-1),
            np.linalg.norm(expected[:, :3, :3], axis=-1),
            rtol=1.0e-6,
        )


@pytest.mark.parametrize("gpu_options", [None, 3])
@pytest.mark.parametrize("native", [False, True])
def test_fabric_hierarchy_uses_available_sdk_path(gpu_options, native, monkeypatch):
    """Consumers share one SDP binding and hierarchy update; cloning owns neither."""
    context = UsdReplicateContext(None)
    assert not any(hasattr(context, name) for name in ("_prepare_fabric", "_update_fabric"))
    assert not hasattr(SceneDataProvider, "_update_fabric")
    calls = []
    hierarchy = SimpleNamespace(update_world_xforms=lambda: calls.append("cpu"))
    if gpu_options is not None:
        hierarchy.update_world_xforms_gpu_with_options = lambda options: calls.append(("gpu", options))
        hierarchy.track_world_xform_changes = lambda active: calls.append(("world", active))
        hierarchy.track_local_xform_changes = lambda active: calls.append(("local", active))
    fabric_stage = Mock()
    attach = Mock(return_value=fabric_stage)
    fabric_hierarchy = SimpleNamespace(
        IFabricHierarchy=lambda: SimpleNamespace(get_fabric_hierarchy=lambda *args: hierarchy)
    )
    if gpu_options is not None:
        fabric_hierarchy.FabricHierarchyGpuUpdateOptions = SimpleNamespace(RIGID_BODY=1, FORCE_UPDATE=2)
    usdrt = SimpleNamespace(
        Usd=SimpleNamespace(
            Stage=SimpleNamespace(Attach=attach), Access=SimpleNamespace(Read=object(), ReadWrite=object())
        ),
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Matrix4d=object())),
        hierarchy=fabric_hierarchy,
    )
    monkeypatch.setitem(sys.modules, "usdrt", usdrt)
    monkeypatch.setitem(sys.modules, "usdrt.hierarchy", fabric_hierarchy)
    monkeypatch.setattr(UsdUtils, "StageCache", SimpleNamespace(Get=lambda: Mock()))
    backend = SimpleNamespace(fabric=Mock() if native else None, fabric_dirty=True)
    provider = SceneDataProvider(backend)
    stage = object()
    provider._prepare_fabric(stage, "cpu")
    provider._prepare_fabric(stage, "cpu")
    attach.assert_called_once()
    fabric_stage.SelectPrims.assert_called_once()
    assert fabric_stage.SelectPrims.call_args.kwargs["require_attrs"][0][2] is (
        usdrt.Usd.Access.Read if native else usdrt.Usd.Access.ReadWrite
    )
    if native:
        fabric_stage.SynchronizeToFabric.assert_not_called()
        assert calls == []
        output = provider._fabric_output
        output.matrices = object()
        provider._fabric_selection.PrepareForReuse.return_value = False
        assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
        assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
        backend.fabric.force_update.assert_called_once_with(0.0, 0.0)
        backend.fabric_dirty = True
        provider.request_transforms(SceneDataFormat.FabricMatrix44)
        assert backend.fabric.force_update.call_count == 2
    else:
        fabric_stage.SynchronizeToFabric.assert_called_once()
        assert calls == ["cpu"]
