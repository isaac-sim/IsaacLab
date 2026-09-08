# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneData geometry (points) mapping and copy."""

from __future__ import annotations

import numpy as np
import warp as wp

from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider


class _PointsBackend(SceneDataBackend):
    def __init__(
        self,
        points: np.ndarray | None = None,
        geometry_paths: list[str] | None = None,
        geometry_counts: list[int] | None = None,
    ):
        if points is None:
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        self._points = wp.array(points, dtype=wp.vec3f)
        self._geometry_paths = geometry_paths or ["/World/envs/env_0/A", "/World/envs/env_1/A"]
        self._geometry_counts = geometry_counts or [2, 3]
        self._scene_data = SceneDataFormat.Points()
        self._scene_data.points = self._points

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
        return int(self._points.shape[0])

    @property
    def geometry_paths(self) -> list[str]:
        return self._geometry_paths

    @property
    def geometry_counts(self) -> list[int]:
        return self._geometry_counts


def test_create_geometry_mapping_returns_none_for_identity_order():
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)
    mapping = provider.create_geometry_mapping(
        ["/World/envs/env_0/A", "/World/envs/env_1/A"],
        [0, 2],
    )
    assert mapping is None


def test_create_geometry_mapping_remaps_out_of_order_entities():
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)
    mapping = provider.create_geometry_mapping(
        ["/World/envs/env_1/A", "/World/envs/env_0/A"],
        [0, 3],
    )
    assert mapping is not None
    assert mapping.numpy().tolist() == [3, 0]


def test_get_points_copies_unpadded_entity_slices():
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)
    output = SceneDataFormat.Points()
    output.points = wp.empty(5, dtype=wp.vec3f)
    mapping = provider.create_geometry_mapping(
        ["/World/envs/env_1/A", "/World/envs/env_0/A"],
        [0, 3],
    )
    assert provider.get_points(output, mapping=mapping, allow_passthrough=False)

    copied = output.points.numpy()
    # Backend order is env_0 (2 pts) then env_1 (3 pts). Mapping writes env_1 to
    # offset 0 and env_0 to offset 3.
    assert np.allclose(copied[0:3, 0], [2.0, 0.0, 1.0])
    assert np.allclose(copied[3:5, 0], [0.0, 1.0])


def test_get_points_clamps_copy_to_destination_capacity(caplog):
    """Oversized backend entity counts must not overflow the consumer buffer."""
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)
    output = SceneDataFormat.Points()
    # Second backend entity has 3 points; destination only has room for 2.
    output.points = wp.zeros(2, dtype=wp.vec3f)
    mapping = wp.array([-1, 0], dtype=wp.int32)

    with caplog.at_level("WARNING"):
        assert provider.get_points(output, mapping=mapping, allow_passthrough=False)

    copied = output.points.numpy()
    assert np.allclose(copied[:, 0], [2.0, 0.0])
    assert any("Clamping geometry point copy" in record.message for record in caplog.records)


def test_get_points_clamps_copy_to_next_destination_slot(caplog):
    """Backend strides larger than shadow slots must not bleed into the next entity."""
    # Backend: two entities with 3 points each (flat src stride 3).
    # Shadow mapping: dest slots of size 2 at offsets 0 and 2 (flat dest size 4).
    backend = _PointsBackend(
        points=np.array([[float(i), 0.0, 0.0] for i in range(6)], dtype=np.float32),
        geometry_counts=[3, 3],
    )
    provider = SceneDataProvider(backend)
    output = SceneDataFormat.Points()
    output.points = wp.zeros(4, dtype=wp.vec3f)
    mapping = wp.array([0, 2], dtype=wp.int32)

    with caplog.at_level("WARNING"):
        assert provider.get_points(output, mapping=mapping, allow_passthrough=False)

    copied = output.points.numpy()
    # Each shadow slot receives only the first 2 points of its backend entity.
    assert np.allclose(copied[:, 0], [0.0, 1.0, 3.0, 4.0])
    assert any("Clamping geometry point copy" in record.message for record in caplog.records)
