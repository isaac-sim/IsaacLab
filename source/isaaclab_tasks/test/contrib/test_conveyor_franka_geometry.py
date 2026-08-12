# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Durable geometry checks for the contributed conveyor Franka task."""

from collections import Counter

from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_TOP_Z,
    TURN_SEGMENT_COUNT,
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


def test_racetrack_visual_meshes_are_named_watertight_loops():
    """Belts and rails remain uniquely named, closed racetrack meshes."""
    specs = tuple(spec for side in ("Left", "Right") for spec in (belt_mesh_spec(side), *guard_mesh_specs(side)))
    expected_loop_vertices = 2 * TURN_SEGMENT_COUNT + 2

    assert len(specs) == 6
    assert len({spec.name for spec in specs}) == len(specs)
    for spec in specs:
        assert len(spec.vertices) == 4 * expected_loop_vertices
        assert len(spec.faces) == 8 * expected_loop_vertices
        assert set(_edge_use_counts(spec).values()) == {2}


def test_belt_top_faces_point_upward():
    """One-sided triangle-mesh surfaces support parcels from above."""
    for side in ("Left", "Right"):
        spec = belt_mesh_spec(side)
        for face in spec.faces:
            vertices = tuple(spec.vertices[index] for index in face)
            if all(vertex[2] == BELT_TOP_Z for vertex in vertices):
                a, b, c = vertices
                cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                assert cross_z > 0.0
