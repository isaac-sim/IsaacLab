# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from isaaclab.utils.mesh import create_trimesh_from_geom_shape

pytestmark = pytest.mark.unit


class _Attribute:
    def __init__(self, value):
        self._value = value

    def Get(self):
        return self._value


class _ConePrim:
    def __init__(self, axis: str, radius: float = 1.0, height: float = 4.0):
        self._attributes = {"axis": axis, "radius": radius, "height": height}

    def GetTypeName(self):
        return "Cone"

    def GetAttribute(self, name):
        return _Attribute(self._attributes[name])

    def GetPath(self):
        return "/World/Cone"


@pytest.mark.parametrize(("axis", "axis_index"), [("X", 0), ("Y", 1), ("Z", 2)])
def test_cone_primitive_respects_axis_and_apex_direction(axis, axis_index):
    height = 4.0
    mesh = create_trimesh_from_geom_shape(_ConePrim(axis=axis, height=height))

    vertices = np.asarray(mesh.vertices)
    apex = vertices[np.argmax(vertices[:, axis_index])]
    transverse_indices = [index for index in range(3) if index != axis_index]

    assert np.isclose(apex[axis_index], height / 2.0)
    np.testing.assert_allclose(apex[transverse_indices], 0.0, atol=1e-7)
    assert np.isclose(vertices[:, axis_index].min(), -height / 2.0)
