# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for volume deformable sim-to-visual remapping."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.deformable_vis_remap import (
    build_volume_vis_barycentric_remap,
    launch_volume_vis_remap,
)

pytestmark = pytest.mark.usefixtures("wp_init")


@pytest.fixture
def wp_init():
    wp.init()


def test_build_volume_vis_barycentric_remap_embeds_vis_vertex_in_tet():
    sim_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    vis_vertices = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)

    remap = build_volume_vis_barycentric_remap(sim_vertices, tet_indices, vis_vertices)
    assert remap is not None
    assert remap.tet_vertex_indices.shape == (1, 4)
    assert remap.bary_weights.shape == (1, 4)
    np.testing.assert_allclose(remap.bary_weights.numpy()[0], [0.25, 0.25, 0.25, 0.25], atol=1e-4)


def test_build_volume_vis_barycentric_remap_clamps_outside_hull_vertex(caplog):
    sim_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    vis_vertices = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    with caplog.at_level("WARNING", logger="isaaclab.scene_data.deformable_vis_remap"):
        remap = build_volume_vis_barycentric_remap(sim_vertices, tet_indices, vis_vertices)

    assert remap is not None
    assert any("clamped" in record.message for record in caplog.records)


def test_launch_volume_vis_remap_interpolates_sim_into_render_buffer():
    remap = build_volume_vis_barycentric_remap(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
    )
    assert remap is not None

    sim_q = wp.array(
        [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    render_q = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    launch_volume_vis_remap(sim_q, render_q, 0, 0, remap)
    result = render_q.numpy()[0]
    np.testing.assert_allclose(result, [0.25, 0.25, 0.25], atol=1e-4)


def test_launch_volume_vis_remap_reuses_device_arrays():
    remap = build_volume_vis_barycentric_remap(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
    )
    assert remap is not None

    tet_ptr = remap.tet_vertex_indices.ptr
    bary_ptr = remap.bary_weights.ptr

    sim_q = wp.array(
        [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    render_q = wp.zeros(1, dtype=wp.vec3f, device="cpu")

    launch_volume_vis_remap(sim_q, render_q, 0, 0, remap)

    sim_q = wp.array(
        [
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),
            wp.vec3(1.0, 1.0, 0.0),
            wp.vec3(1.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    launch_volume_vis_remap(sim_q, render_q, 0, 0, remap)

    assert remap.tet_vertex_indices.ptr == tet_ptr
    assert remap.bary_weights.ptr == bary_ptr
    np.testing.assert_allclose(render_q.numpy()[0], [1.25, 0.25, 0.25], atol=1e-4)
