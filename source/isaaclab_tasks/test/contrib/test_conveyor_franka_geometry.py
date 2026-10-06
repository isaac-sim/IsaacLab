# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Durable geometry checks for the contributed conveyor Franka task."""

from collections import Counter

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import _collision_properties, _cube
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    MeshSpec,
    belt_mesh_spec,
    guard_mesh_specs,
)


def _edge_use_counts(spec: MeshSpec) -> Counter[tuple[int, int]]:
    """Count triangle uses of every undirected mesh edge."""
    edges: Counter[tuple[int, int]] = Counter()
    for triangle in spec.faces:
        for start, end in zip(triangle, triangle[1:] + triangle[:1], strict=True):
            edges[tuple(sorted((start, end)))] += 1
    return edges


def test_belt_top_faces_point_upward():
    """One-sided triangle-mesh surfaces support parcels from above."""
    for side in ("Left", "Right"):
        specs = [belt_mesh_spec(side), *guard_mesh_specs(side)]
        for spec in specs:
            assert set(_edge_use_counts(spec).values()) == {2}
            _assert_top_faces_point_upward(spec)


def _assert_top_faces_point_upward(spec):
    top_z = max(vertex[2] for vertex in spec.vertices)
    for face in spec.faces:
        vertices = tuple(spec.vertices[index] for index in face)
        if all(vertex[2] == top_z for vertex in vertices):
            a, b, c = vertices
            cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            assert cross_z > 0.0


def test_contact_configuration_uses_one_mujoco_parameterization():
    """Raw MuJoCo solref must not be combined with shadowed Newton force-space gains."""
    mujoco_cfg = _collision_properties()[-1]
    cube_material = _cube("TestCube", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)).spawn.physics_material

    assert mujoco_cfg.solref is not None
    assert cube_material.contact_stiffness is None
    assert cube_material.contact_damping is None
    assert cube_material.torsional_friction is None
    assert cube_material.rolling_friction is None
