# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneDataProvider transform conversion, index mapping, and geometry (points) copies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="requires CUDA")


class _PointsBackend(SceneDataBackend):
    """Backend publishing two geometry entities: env_0 (2 points) then env_1 (3 points)."""

    def __init__(self, points: np.ndarray | None = None, geometry_counts: list[int] | None = None, device=None):
        if points is None:
            points = np.array([[i, 0.0, 0.0] for i in (0.0, 1.0, 2.0, 0.0, 1.0)], dtype=np.float32)
            points[3:, 1] = 1.0
        self._geometry_counts = geometry_counts or [2, 3]
        self._scene_data = SceneDataFormat.Points()
        self._scene_data.points = wp.array(points, dtype=wp.vec3f, device=device)

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return SceneDataFormat.Transform()

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []

    @property
    def points(self) -> SceneDataFormat.Points:
        return self._scene_data

    @property
    def point_count(self) -> int:
        return int(self._scene_data.points.shape[0])

    @property
    def geometry_paths(self) -> list[str]:
        return ["/World/envs/env_0/A", "/World/envs/env_1/A"]

    @property
    def geometry_counts(self) -> list[int]:
        return self._geometry_counts


_SWAPPED = ["/World/envs/env_1/A", "/World/envs/env_0/A"]


@requires_cuda
def test_get_transforms_follows_publication_device_not_warp_default():
    """Transform conversion and mapping follow the CPU publication even when Warp's default device is CUDA."""
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array(
        [[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for x in range(3)], dtype=wp.transformf, device="cpu"
    )
    provider = SceneDataProvider(
        SimpleNamespace(transforms=transforms, transform_count=3, transform_paths=["/World/a", "/World/b", "/World/c"])
    )

    with wp.ScopedDevice("cuda:0"):
        assert provider.create_mapping(["/World/a", "/World/b", "/World/c"]) is None
        mapping = provider.create_mapping(["/World/c", "/World/a", "/World/b"])
        assert str(mapping.device) == "cpu"
        output = SceneDataFormat.Vec3_Quat()
        assert provider.get_transforms(output, mapping=mapping, allow_passthrough=False)

    assert str(output.positions.device) == str(output.orientations.device) == "cpu"
    assert output.positions.numpy()[:, 0].tolist() == [2.0, 0.0, 1.0]
    assert output.orientations.numpy()[:, 3].tolist() == [1.0, 1.0, 1.0]


def test_get_transforms_passthrough_and_copy_for_matching_formats():
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transformf, device="cpu")
    provider = SceneDataProvider(SimpleNamespace(transforms=transforms, transform_count=1, transform_paths=[]))

    aliased = SceneDataFormat.Transform()
    assert provider.get_transforms(aliased)
    assert aliased.transforms is transforms.transforms

    copied = SceneDataFormat.Transform()
    assert provider.get_transforms(copied, allow_passthrough=False)
    assert copied.transforms is not transforms.transforms
    assert copied.transforms.numpy().tolist() == transforms.transforms.numpy().tolist()


def test_create_geometry_mapping():
    provider = SceneDataProvider(_PointsBackend())
    assert provider.create_geometry_mapping(["/World/envs/env_0/A", "/World/envs/env_1/A"], [0, 2]) is None
    assert provider.create_geometry_mapping(_SWAPPED, [0, 3]).numpy().tolist() == [3, 0]


@requires_cuda
def test_create_geometry_mapping_uses_points_device_with_empty_transforms():
    """Geometry mapping follows its point publication, independently of transforms."""
    with wp.ScopedDevice("cpu"):
        mapping = SceneDataProvider(_PointsBackend(device="cuda:0")).create_geometry_mapping(_SWAPPED, [0, 3])
    assert mapping.device == wp.get_device("cuda:0")


def test_create_geometry_mapping_rejects_empty_points_publication():
    backend = _PointsBackend()
    backend._scene_data.points = None
    with pytest.raises(ValueError, match="Points contains no published arrays"):
        SceneDataProvider(backend).create_geometry_mapping(_SWAPPED, [0, 3])


def test_get_points_copies_unpadded_entity_slices():
    provider = SceneDataProvider(_PointsBackend())
    output = SceneDataFormat.Points()
    output.points = wp.empty(5, dtype=wp.vec3f)
    mapping = provider.create_geometry_mapping(_SWAPPED, [0, 3])
    assert provider.get_points(output, mapping=mapping, allow_passthrough=False)
    # env_1 (3 points) lands at offset 0 and env_0 (2 points) at offset 3
    assert output.points.numpy()[:, 0].tolist() == [2.0, 0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("backend", "dst_size", "mapping", "expected_x"),
    [
        # second entity has 3 points; destination only has room for 2
        pytest.param(_PointsBackend(), 2, [-1, 0], [2.0, 0.0], id="destination_capacity"),
        # backend stride 3 but shadow slots of size 2 at offsets 0 and 2: no bleed into the next slot
        pytest.param(
            _PointsBackend(points=np.array([[float(i), 0.0, 0.0] for i in range(6)]), geometry_counts=[3, 3]),
            4,
            [0, 2],
            [0.0, 1.0, 3.0, 4.0],
            id="next_slot",
        ),
    ],
)
def test_get_points_clamps_oversized_entity_copies(caplog, backend, dst_size, mapping, expected_x):
    provider = SceneDataProvider(backend)
    output = SceneDataFormat.Points()
    output.points = wp.zeros(dst_size, dtype=wp.vec3f)

    with caplog.at_level("WARNING"):
        assert provider.get_points(output, mapping=wp.array(mapping, dtype=wp.int32), allow_passthrough=False)

    assert output.points.numpy()[:, 0].tolist() == expected_x
    assert any("Clamping geometry point copy" in record.message for record in caplog.records)
