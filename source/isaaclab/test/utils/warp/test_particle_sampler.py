# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp mesh particle samplers."""

import re

import numpy as np
import pytest

from isaaclab.utils.warp import sample_particles_in_cavity, sample_particles_in_mesh


def _box_mesh(lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an outward-facing triangulated cube."""
    vertices = np.asarray(
        [
            [lower, lower, lower],
            [upper, lower, lower],
            [upper, upper, lower],
            [lower, upper, lower],
            [lower, lower, upper],
            [upper, lower, upper],
            [upper, upper, upper],
            [lower, upper, upper],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_sample_particles_in_mesh_fills_solid_volume_on_cpu():
    vertices, faces = _box_mesh(-1.0, 1.0)

    points = sample_particles_in_mesh(vertices, faces, spacing=1.0, device="cpu")

    assert points.shape == (8, 3)
    assert np.all(np.abs(points) == pytest.approx(0.5))


def test_sample_particles_in_cavity_fills_empty_shell_and_honors_water_level_on_cpu():
    outer_vertices, outer_faces = _box_mesh(-1.0, 1.0)
    inner_vertices, inner_faces = _box_mesh(-0.6, 0.6)
    vertices = np.concatenate((outer_vertices, inner_vertices))
    # Reverse the inner surface so its winding cancels the outer volume inside the cavity.
    faces = np.concatenate((outer_faces, inner_faces[:, ::-1] + outer_vertices.shape[0]))

    points = sample_particles_in_cavity(
        vertices,
        faces,
        spacing=0.5,
        device="cpu",
        min_ray_hits=6,
        water_level=0.0,
    )

    assert points.shape == (4, 3)
    assert np.all(points[:, 2] == pytest.approx(-0.25))
    assert np.all(np.abs(points[:, :2]) == pytest.approx(0.25))


def test_sample_particles_in_cavity_honors_surface_margin_on_cpu():
    outer_vertices, outer_faces = _box_mesh(-1.0, 1.0)
    inner_vertices, inner_faces = _box_mesh(-0.6, 0.6)
    vertices = np.concatenate((outer_vertices, inner_vertices))
    faces = np.concatenate((outer_faces, inner_faces[:, ::-1] + outer_vertices.shape[0]))

    points = sample_particles_in_cavity(
        vertices,
        faces,
        spacing=0.2,
        device="cpu",
        surface_margin=0.21,
        min_ray_hits=6,
    )

    assert points.shape == (64, 3)
    assert np.all(np.abs(points) <= 0.3 + 1.0e-6)


def test_sample_particles_in_mesh_rejects_non_integer_face_indices():
    vertices, faces = _box_mesh(-1.0, 1.0)
    invalid_faces = faces.astype(np.float64)
    invalid_faces[0, 0] = 0.5

    with pytest.raises(ValueError, match="`faces` must contain integer indices"):
        sample_particles_in_mesh(vertices, invalid_faces, spacing=0.5, device="cpu")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spacing": 0.5, "device": "cpu", "jitter": -0.1}, "`jitter` must be in [0, 0.5]"),
        ({"spacing": 0.5, "device": "cpu", "jitter": 0.6}, "`jitter` must be in [0, 0.5]"),
        ({"spacing": 0.5, "device": "cpu", "inset": -0.1}, "`inset` must be non-negative"),
        ({"spacing": 0.5, "device": "cpu", "surface_margin": -0.1}, "`surface_margin` must be non-negative"),
        ({"spacing": 0.5, "device": "cpu", "max_query_dist": 0.0}, "`max_query_dist` must be positive"),
    ],
)
def test_sample_particles_in_mesh_rejects_invalid_settings(kwargs, message):
    vertices, faces = _box_mesh(-1.0, 1.0)

    with pytest.raises(ValueError, match=re.escape(message)):
        sample_particles_in_mesh(vertices, faces, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spacing": 0.0, "device": "cpu"}, "`spacing` must be positive"),
        ({"spacing": 0.5, "device": "cpu", "jitter": 0.6}, "`jitter` must be in [0, 0.5]"),
        ({"spacing": 0.5, "device": "cpu", "inset": -0.1}, "`inset` must be non-negative"),
        ({"spacing": 0.5, "device": "cpu", "surface_margin": -0.1}, "`surface_margin` must be non-negative"),
        ({"spacing": 0.5, "device": "cpu", "min_ray_hits": 0}, "`min_ray_hits` must be in [1, 6]"),
        ({"spacing": 0.5, "device": "cpu", "min_ray_hits": 1.9}, "`min_ray_hits` must be an integer"),
        ({"spacing": 0.5, "device": "cpu", "max_ray_dist": 0.0}, "`max_ray_dist` must be positive"),
        ({"spacing": 0.5, "device": "cpu", "max_query_dist": 0.0}, "`max_query_dist` must be positive"),
    ],
)
def test_sample_particles_in_cavity_rejects_invalid_settings(kwargs, message):
    vertices, faces = _box_mesh(-1.0, 1.0)

    with pytest.raises(ValueError, match=re.escape(message)):
        sample_particles_in_cavity(vertices, faces, **kwargs)
