# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-face orientation for the contributed conveyor task."""

import pytest

from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import MeshSpec, belt_collision_geometry_specs

pytestmark = pytest.mark.kitless


def test_belt_collision_top_faces_point_upward():
    """One-sided conveyor collision meshes support parcels from above."""
    for side in ("Left", "Right"):
        for spec in belt_collision_geometry_specs(side):
            if not isinstance(spec, MeshSpec):
                continue
            top_z = max(vertex[2] for vertex in spec.vertices)
            top_faces = [face for face in spec.faces if all(spec.vertices[index][2] == top_z for index in face)]
            assert top_faces
            for face in top_faces:
                a, b, c = (spec.vertices[index] for index in face)
                cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                assert cross_z > 0.0
