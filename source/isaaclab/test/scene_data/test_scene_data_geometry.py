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
    def __init__(self):
        self._points = wp.array(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            dtype=wp.vec3f,
        )
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
        return ["/World/envs/env_0/A", "/World/envs/env_1/A"]

    @property
    def geometry_counts(self) -> list[int]:
        return [2, 3]


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
