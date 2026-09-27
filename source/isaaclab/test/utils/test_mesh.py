# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD primitive conversion without launching a simulator."""

import numpy as np
import pytest

from pxr import Usd, UsdGeom

from isaaclab.utils.mesh import create_trimesh_from_geom_shape

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "shape_type, axis",
    [(UsdGeom.Cone, "X"), (UsdGeom.Cone, "Y"), (UsdGeom.Cone, "Z"), (UsdGeom.Cylinder, "X"), (UsdGeom.Capsule, "Y")],
)
def test_primitive_mesh_axis(shape_type, axis):
    """Meshes retain USD bounds, and cone apices point along the positive authored axis."""
    stage = Usd.Stage.CreateInMemory()
    shape = shape_type.Define(stage, "/Shape")
    radius, height = 0.5, 4.0
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(height)
    shape.CreateAxisAttr(axis)

    mesh = create_trimesh_from_geom_shape(shape.GetPrim())

    axis_index = "XYZ".index(axis)
    half_extents = np.full(3, radius)
    half_extents[axis_index] = height / 2 + (radius if shape_type is UsdGeom.Capsule else 0.0)
    # Capsule hemispheres are tessellated, so their radial bounds only approximate the radius.
    np.testing.assert_allclose(mesh.bounds, [-half_extents, half_extents], rtol=2e-3, atol=1e-6)
    if shape_type is UsdGeom.Cone:
        apex = mesh.vertices[np.isclose(mesh.vertices[:, axis_index], height / 2)]
        expected_apex = np.zeros((1, 3))
        expected_apex[0, axis_index] = height / 2
        np.testing.assert_allclose(apex, expected_apex, atol=1e-6)
