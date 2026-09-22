# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for volume deformable sim-to-visual remapping."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.deformable_vis_remap import build_volume_vis_barycentric_remap, launch_volume_vis_remap

pytestmark = pytest.mark.unit

_UNIT_TET = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_TET_INDICES = np.array([0, 1, 2, 3], dtype=np.int32)


def _vis(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[x, y, z]], dtype=np.float32)


def test_remap_embeds_and_interpolates_vis_vertex():
    """A visual vertex inside the tet gets its barycentric weights and follows the sim nodes on remap."""
    remap = build_volume_vis_barycentric_remap(_UNIT_TET, _TET_INDICES, _vis(0.25, 0.25, 0.25))
    assert remap is not None
    assert remap.tet_vertex_indices.numpy().tolist() == [[0, 1, 2, 3]]
    np.testing.assert_allclose(remap.bary_weights.numpy()[0], [0.25] * 4, atol=1e-4)

    render_q = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    for shift in (0.0, 1.0):
        sim_q = wp.array(_UNIT_TET + np.array([shift, 0.0, 0.0], dtype=np.float32), dtype=wp.vec3f, device="cpu")
        launch_volume_vis_remap(sim_q, render_q, 0, 0, remap)
        np.testing.assert_allclose(render_q.numpy()[0], [0.25 + shift, 0.25, 0.25], atol=1e-4)


def test_remap_clamps_outside_hull_vertex(caplog):
    with caplog.at_level("WARNING", logger="isaaclab.scene_data.deformable_vis_remap"):
        remap = build_volume_vis_barycentric_remap(_UNIT_TET, _TET_INDICES, _vis(2.0, 2.0, 2.0))
    assert remap is not None
    assert any("clamped 1/1" in record.message for record in caplog.records)


def test_remap_returns_none_for_empty_inputs():
    empty = np.empty((0, 3), dtype=np.float32)
    assert build_volume_vis_barycentric_remap(_UNIT_TET, _TET_INDICES, empty) is None
    assert build_volume_vis_barycentric_remap(empty, _TET_INDICES, _UNIT_TET[:1]) is None
