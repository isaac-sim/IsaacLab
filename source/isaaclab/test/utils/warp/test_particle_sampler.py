# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp mesh particle samplers."""

import numpy as np
import pytest

from isaaclab.utils.warp import make_box_region_mesh, sample_particles_in_cavity, sample_particles_in_mesh

pytestmark = pytest.mark.unit


def _shell_mesh(outer: float, inner: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed double-walled cube whose inner surface faces into the cavity."""
    outer_vertices, outer_faces = make_box_region_mesh((outer,) * 3)
    inner_vertices, inner_faces = make_box_region_mesh((inner,) * 3)
    vertices = np.concatenate((outer_vertices, inner_vertices))
    # Reverse the inner surface so its winding cancels the outer volume inside the cavity.
    faces = np.concatenate((outer_faces, inner_faces[:, ::-1] + outer_vertices.shape[0]))
    return vertices, faces


def test_sample_particles_in_mesh_fills_solid_volume():
    vertices, faces = make_box_region_mesh((1.0, 1.0, 1.0))
    points = sample_particles_in_mesh(vertices, faces, spacing=1.0, device="cpu")
    assert points.shape == (8, 3)
    assert np.all(np.abs(points) == pytest.approx(0.5))


def test_sample_particles_in_cavity_honors_water_level_and_surface_margin():
    vertices, faces = _shell_mesh(1.0, 0.6)

    below_water = sample_particles_in_cavity(
        vertices, faces, spacing=0.5, device="cpu", min_ray_hits=6, water_level=0.0
    )
    assert below_water.shape == (4, 3)
    assert np.all(below_water[:, 2] == pytest.approx(-0.25))
    assert np.all(np.abs(below_water[:, :2]) == pytest.approx(0.25))

    eroded = sample_particles_in_cavity(vertices, faces, spacing=0.2, device="cpu", surface_margin=0.21, min_ray_hits=6)
    assert eroded.shape == (64, 3)
    assert np.all(np.abs(eroded) <= 0.3 + 1.0e-6)


def test_samplers_reject_non_integer_face_indices():
    vertices, faces = make_box_region_mesh((1.0, 1.0, 1.0))
    invalid_faces = faces.astype(np.float64)
    invalid_faces[0, 0] = 0.5
    with pytest.raises(ValueError, match="`faces` must contain integer indices"):
        sample_particles_in_mesh(vertices, invalid_faces, spacing=0.5, device="cpu")
