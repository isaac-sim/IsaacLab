# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneDataProvider transform conversion and index mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.scene_data_backend import SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider
from isaaclab.test.utils import test_devices


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
            transforms_dirty=True,
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
    backend = SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=1)
    provider = SceneDataProvider(backend)
    native = SceneDataFormat.Transform()
    converted = SceneDataFormat.Vec3_Quat()
    other = SceneDataFormat.Vec3_Quat()
    with pytest.raises(ValueError, match="destination count"):
        provider.get_transforms(native, count=2)
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
    assert provider.get_transforms(native)
    assert native.transforms is data.transforms
    launch.assert_not_called()
    assert provider.get_transforms(converted)
    assert provider.get_transforms(other)
    assert other.positions is converted.positions
    assert launch.call_count == 1
    np.testing.assert_array_equal(converted.positions.numpy(), [[1, 2, 3]])
    # A consumer can rebind its wrapper without changing another consumer's arrays.
    converted.positions = None
    assert provider.get_transforms(converted)
    assert converted.positions is other.positions

    data.transforms.assign([[4, 5, 6, 0, 0, 0, 1]])
    backend.transforms_dirty = True
    assert provider.get_transforms(converted)
    assert converted.positions is other.positions
    assert launch.call_count == 2
    np.testing.assert_array_equal(converted.positions.numpy(), [[4, 5, 6]])

    data.transforms = wp.array([[7, 8, 9, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend.transforms_dirty = True
    assert provider.get_transforms(native)
    assert native.transforms is data.transforms
    assert provider.get_transforms(converted)
    assert converted.positions is other.positions
    assert launch.call_count == 3
    np.testing.assert_array_equal(converted.positions.numpy(), [[7, 8, 9]])


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat"])
def test_owned_transform_buffers_are_written_directly_and_do_not_alias_cache(format_name, monkeypatch):
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend = SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=1)
    provider = SceneDataProvider(backend)
    shared, owned = (getattr(SceneDataFormat, format_name)() for _ in range(2))
    assert provider.get_transforms(shared)
    provider.init_output(owned)
    arrays = [getattr(owned, name) for name in owned._cls.vars]
    launch, copy = Mock(wraps=wp.launch), Mock(wraps=wp.copy)
    monkeypatch.setattr(wp, "launch", launch)
    monkeypatch.setattr(wp, "copy", copy)
    for x in (4, 7):
        data.transforms.assign([[x, 5, 6, 0, 0, 0, 1]])
        backend.transforms_dirty = True
        launch.reset_mock()
        copy.reset_mock()
        assert provider.get_transforms(owned, allow_passthrough=False)
        assert launch.call_count == int(format_name != "Transform")
        assert copy.call_count == int(format_name == "Transform")
        assert provider.get_transforms(shared)
        for name, array in zip(owned._cls.vars, arrays):
            assert getattr(owned, name) is array
            assert array is not getattr(shared, name)
            np.testing.assert_array_equal(array.numpy(), getattr(shared, name).numpy())


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
    output = SceneDataFormat.TransposedMatrix44d()
    assert provider.get_transforms(output, mapping, scales=scales)
    expected = matrices[::-1].transpose(0, 2, 1).copy()
    if scaled:
        expected[:, :3, :3] *= scales.numpy()[:, :, None]
    np.testing.assert_allclose(output.matrices.numpy(), expected)
    matrices = output.matrices
    assert provider.get_transforms(output, mapping, scales=scales)
    assert output.matrices is matrices


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat", "Vec3_Matrix33", "Matrix44"])
@pytest.mark.parametrize("device", test_devices())
def test_fabric_conversion_preserves_scale_and_refreshes_reallocated_destinations(format_name, device, monkeypatch):
    """Fabric conversion skips solver-only bodies and preserves scales across buffer reallocations."""
    poses = [[1, 2, 3, 0, 0, 0, 1], [7, 8, 9, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]
    data = SceneDataFormat.Transform()
    data.transforms = wp.array(poses, dtype=wp.transformf, device=device)
    native = SceneDataProvider(SimpleNamespace(transforms=data, transforms_dirty=True, transform_count=len(poses)))
    source = getattr(SceneDataFormat, format_name)()
    assert native.get_transforms(source)
    provider = SceneDataProvider(
        SimpleNamespace(
            transforms=source,
            transforms_dirty=True,
            transform_count=len(poses),
            fabric=None,
        )
    )
    provider._fabric_output = SceneDataFormat.FabricMatrix44()
    assert provider._fabric_output._cls is SceneDataFormat.FabricMatrix44
    provider._fabric_output.scales = wp.empty(len(poses), dtype=wp.vec3f, device=device)
    expected = np.array([np.diag([-2, -3, 4, 1]), np.diag([5, 6, 7, 1])], dtype=np.float64)
    expected[:, 3, :3] = [[4, 5, 6], [1, 2, 3]]
    indices = wp.array([len(poses) - 1, 0], dtype=wp.int32, device=device)
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
    scales = provider._fabric_output.scales
    for allocation in range(2):
        authored = np.array([np.diag([2, 3, 4, 1]), np.diag([5, 6, 7, 1])], dtype=np.float64)
        if allocation:
            authored[:, :3, :3] *= 1.001  # Rebinding must not recapture scale from a rounded runtime cache.
        matrices = wp.array(authored, dtype=wp.mat44d, device=device)
        local_matrices = wp.empty(2, dtype=wp.mat44d, device=device)

        def update_world_xforms_gpu(_no_structural_changes):
            matrices.assign(local_matrices)
            return True

        provider._fabric_hierarchy = Mock()
        provider._fabric_hierarchy.update_world_xforms_gpu.side_effect = update_world_xforms_gpu
        interface = {
            "version": 1,
            "device": device,
            "attribs": {
                "isaaclab:transformIndex": {
                    "type": (True, "i4", 1, 0, ""),
                    "access": 1,
                    "pointers": [indices.ptr],
                    "counts": [2],
                },
                "omni:fabric:worldMatrix": {
                    "type": (True, "f8", 16, 0, "matrix"),
                    "access": 1,
                    "pointers": [matrices.ptr],
                    "counts": [2],
                },
                "omni:fabric:localMatrix": {
                    "type": (True, "f8", 16, 0, "matrix"),
                    "access": 2,
                    "pointers": [local_matrices.ptr],
                    "counts": [2],
                },
            },
        }
        changes = [True]
        provider._fabric_write_selection = SimpleNamespace(
            __fabric_arrays_interface__=interface,
            PrepareForReuse=Mock(return_value=False),
        )
        provider._fabric_selection = SimpleNamespace(
            __fabric_arrays_interface__={
                **interface,
                "attribs": {name: {**attr, "access": 1} for name, attr in interface["attribs"].items()},
            },
            PrepareForReuse=lambda: changes.pop() if changes else False,
        )
        output = SceneDataFormat.FabricMatrix44()
        assert provider.get_transforms(output)
        assert output.scales is scales
        provider._fabric_hierarchy.update_world_xforms_gpu.assert_called_once_with(False)
        provider._fabric_hierarchy.reset_mock()
        provider._fabric_write_selection.PrepareForReuse.reset_mock()
        previous_matrices = output.matrices
        assert provider.get_transforms(output)
        assert output.matrices is previous_matrices
        assert provider._fabric_hierarchy.mock_calls == []
        provider._fabric_write_selection.PrepareForReuse.assert_not_called()
        assert launch.call_count == allocation + 2
        np.testing.assert_allclose(matrices.numpy(), expected)

    if format_name == "Transform":
        rotations = np.random.default_rng(42).normal(size=(2000, len(poses), 4)).astype(np.float32)
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        poses = np.asarray(poses, dtype=np.float32)
        for rotation in rotations:
            poses[:, 3:] = rotation
            data.transforms.assign(poses)
            provider.backend.transforms_dirty = True
            provider.get_transforms(output)
        np.testing.assert_allclose(
            np.linalg.norm(matrices.numpy()[:, :3, :3], axis=-1),
            np.linalg.norm(expected[:, :3, :3], axis=-1),
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(matrices.numpy()[:, 3, :3], poses[[2, 0], :3])
        assert provider._fabric_hierarchy.update_world_xforms_gpu.call_count == len(rotations)
        provider._fabric_hierarchy.update_world_xforms_gpu.assert_called_with(True)
        assert provider._fabric_write_selection.PrepareForReuse.call_count == len(rotations)
