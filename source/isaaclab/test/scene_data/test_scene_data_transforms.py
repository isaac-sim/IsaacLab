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

from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider
from isaaclab.test.utils import test_devices


class _Backend(SimpleNamespace, SceneDataBackend):
    transforms = None
    transform_count = 0
    transform_paths = ()


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
        _Backend(
            transforms=transforms,
            transforms_version=0,
            transform_count=3,
            transform_paths=["/World/a", "/World/b", "/World/c"],
        )
    )

    with wp.ScopedDevice("cuda:0"):
        mapping = provider.create_mapping(["/World/c", "/World/a", "/World/b"])
        assert mapping is not None
        assert str(mapping.device) == "cpu"
        assert provider.create_mapping(["/World/c", "/World/a", "/World/b"]) is mapping

        output = SceneDataFormat.Vec3_Quat()
        assert provider.get_transforms(output, mapping=mapping, allow_passthrough=False)

    assert str(output.positions.device) == "cpu"
    assert str(output.orientations.device) == "cpu"
    assert np.allclose(output.positions.numpy()[:, 0], [2.0, 0.0, 1.0])


def test_publication_aliases_native_pointer_and_converts_once_per_write(monkeypatch):
    """Clean reads share conversions; no provider can hide a publication from another."""
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend = _Backend(transforms=data, transforms_version=0, transform_count=1)
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
    backend.transforms_version += 1
    assert provider.get_transforms(converted)
    assert converted.positions is other.positions
    assert launch.call_count == 2
    np.testing.assert_array_equal(converted.positions.numpy(), [[4, 5, 6]])

    data.transforms = wp.array([[7, 8, 9, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend.transforms_version += 1
    assert provider.get_transforms(native)
    assert native.transforms is data.transforms
    assert provider.get_transforms(converted)
    assert converted.positions is other.positions
    assert launch.call_count == 3
    np.testing.assert_array_equal(converted.positions.numpy(), [[7, 8, 9]])

    peer = SceneDataProvider(backend)
    peer_output = SceneDataFormat.Vec3_Quat()
    assert peer.get_transforms(peer_output)
    for position in ([10, 11, 12], [13, 14, 15]):
        data.transforms.assign([position + [0, 0, 0, 1]])
        backend.transforms_version += 1
        assert provider.get_transforms(converted)
        assert peer.get_transforms(peer_output)
        np.testing.assert_array_equal(converted.positions.numpy(), [position])
        np.testing.assert_array_equal(peer_output.positions.numpy(), [position])


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat"])
def test_owned_transform_buffers_are_written_directly_and_do_not_alias_cache(format_name, monkeypatch):
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    backend = _Backend(transforms=data, transforms_version=0, transform_count=1)
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
        backend.transforms_version += 1
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


def test_mapping_preserves_unmapped_destination_slots():
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    provider = SceneDataProvider(
        _Backend(transforms=data, transforms_version=0, transform_count=2, transform_paths=["/a", "/b"])
    )
    mapping = provider.create_mapping(["/a", "/b", None])
    output = SceneDataFormat.Transform()
    output.transforms = wp.zeros(3, dtype=wp.transformf, device="cpu")
    assert provider.get_transforms(output, mapping, allow_passthrough=False, count=3)
    np.testing.assert_array_equal(output.transforms.numpy()[:2], data.transforms.numpy())
    np.testing.assert_array_equal(output.transforms.numpy()[2], np.zeros(7))
    with pytest.raises(KeyError, match="/missing"):
        provider.create_mapping(["/a", "/missing"])


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat", "Vec3_Matrix33", "Matrix44"])
@pytest.mark.parametrize("scaled", [False, True])
def test_transposed_matrices_fuse_format_mapping_and_scale(format_name, scaled):
    """All native formats produce the same row-vector matrices, with output-indexed scale."""
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 1, 2, 2, 2]], dtype=np.float32)
    poses[1, 3:] /= np.sqrt(13)
    # Independent quaternion-to-matrix reference; a non-axis rotation exposes transpose/scale ordering errors.
    rotations = np.array([np.eye(3), np.array([[-3, -4, 12], [12, 3, 4], [-4, 12, 3]]) / 13], dtype=np.float32)
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
    provider = SceneDataProvider(_Backend(transforms=data, transforms_version=0, transform_count=2))
    mapping = wp.array([1, 0], dtype=wp.int32, device="cpu")
    scales = wp.array([[2, 3, 4], [5, 6, 7]], dtype=wp.vec3f, device="cpu") if scaled else None
    output = SceneDataFormat.TransposedMatrix44d()
    assert provider.get_transforms(output, mapping, scales=scales)
    expected = matrices[::-1].transpose(0, 2, 1).copy()
    if scaled:
        expected[:, :3, :3] *= scales.numpy()[:, :, None]
    np.testing.assert_allclose(output.matrices.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)
    matrices = output.matrices
    assert provider.get_transforms(output, mapping, scales=scales)
    assert output.matrices is matrices


@pytest.mark.parametrize("format_name", ["Transform", "Vec3_Quat", "Vec3_Matrix33", "Matrix44"])
@pytest.mark.parametrize("device", test_devices())
def test_fabric_conversion_preserves_scale_and_refreshes_reallocated_destinations(format_name, device, monkeypatch):
    """Fabric conversion skips solver-only bodies and preserves scales across buffer reallocations."""
    assert set(SceneDataFormat.FabricMatrix44.vars) == {"matrices"}
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [7, 8, 9, 0, 0, 0, 1], [4, 5, 6, 1, 2, 2, 2]], dtype=np.float32)
    poses[2, 3:] /= np.sqrt(13)
    data = SceneDataFormat.Transform()
    data.transforms = wp.array(poses, dtype=wp.transformf, device=device)
    native = SceneDataProvider(_Backend(transforms=data, transforms_version=0, transform_count=len(poses)))
    source = getattr(SceneDataFormat, format_name)()
    assert native.get_transforms(source)
    provider = SceneDataProvider(_Backend(transforms=source, transforms_version=0, transform_count=len(poses)))
    authored_scales = np.array([[5, 6, 7], [1, 1, 1], [2, 3, 4]], dtype=np.float32)
    scales = wp.array(authored_scales, dtype=wp.vec3f, device=device)
    expected = np.array([np.eye(4), np.diag([5, 6, 7, 1])], dtype=np.float64)
    rotation = np.array([[-3, -4, 12], [12, 3, 4], [-4, 12, 3]]) / 13
    expected[0, :3, :3] = np.diag([2, 3, 4]) @ rotation.T
    expected[:, 3, :3] = [[4, 5, 6], [1, 2, 3]]
    indices = wp.array([len(poses) - 1, 0], dtype=wp.int32, device=device)
    launch = Mock(wraps=wp.launch)
    monkeypatch.setattr(wp, "launch", launch)
    for _ in range(2):
        matrices = wp.empty(2, dtype=wp.mat44d, device=device)
        interface = {
            "version": 1,
            "device": device,
            "attribs": {
                "mapping": {
                    "type": (True, "i4", 1, 0, ""),
                    "access": 1,
                    "pointers": [indices.ptr],
                    "counts": [2],
                },
                "matrices": {
                    "type": (True, "f8", 16, 0, "matrix"),
                    "access": 2,
                    "pointers": [matrices.ptr],
                    "counts": [2],
                },
            },
        }
        storage = SimpleNamespace(__fabric_arrays_interface__=interface)
        output = SceneDataFormat.FabricMatrix44()
        output.matrices = wp.fabricarray(storage, "matrices")
        mapping = wp.fabricarray(storage, "mapping")
        destination = output.matrices
        launch.reset_mock()
        assert provider.get_transforms(output, mapping, scales=scales)
        assert provider.get_transforms(output, mapping, scales=scales)
        assert output.matrices is destination
        launch.assert_called_once()
        assert len(provider._transform_cache) == 1
        np.testing.assert_allclose(matrices.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_array_equal(scales.numpy(), authored_scales)
