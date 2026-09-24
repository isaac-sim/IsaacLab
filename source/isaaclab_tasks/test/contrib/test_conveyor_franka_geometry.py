# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Durable geometry checks for the contributed conveyor Franka task."""

from collections import Counter

import pytest

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import _collision_properties, _cube
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_TURN_RADIUS,
    TURN_SEGMENT_COUNT,
    MeshSpec,
    belt_collision_section_specs,
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


@pytest.mark.parametrize("warehouse", [False, True])
def test_belt_top_faces_point_upward(warehouse):
    """One-sided triangle-mesh surfaces support parcels from above."""
    for side in ("Left", "Right"):
        if warehouse:
            from isaaclab_tasks.contrib.conveyor_franka.conveyor_warehouse_geometry import (
                warehouse_belt_sections,
                warehouse_guard_meshes,
            )

            specs = [
                section.geometry for section in warehouse_belt_sections(side) if isinstance(section.geometry, MeshSpec)
            ]
            specs.extend(warehouse_guard_meshes(side))
        else:
            specs = [belt_mesh_spec(side)]
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


def test_collision_sections_carry_schema_aligned_belt_intent():
    """Task geometry and runtime descriptions share paths, units, and curve semantics."""
    sections = belt_collision_section_specs("Left", velocity=0.35, friction_coefficient=0.5, contact_threshold=0.997)

    assert len(sections) == 4
    assert tuple(section.belt.prim_path for section in sections) == tuple(
        f"{{ENV_REGEX_NS}}/{section.geometry.name}" for section in sections
    )
    assert tuple(section.belt.velocity for section in sections) == (0.35,) * 4
    assert tuple(section.belt.friction_coefficient for section in sections) == (0.5,) * 4
    assert tuple(section.belt.contact_threshold for section in sections) == (0.997,) * 4
    assert tuple(section.belt.curved for section in sections) == (False, False, True, True)
    assert tuple(section.belt.radius for section in sections) == (None, None, BELT_TURN_RADIUS, BELT_TURN_RADIUS)


def test_elevated_belt_normals_accept_contacts_on_every_ramp_panel():
    """Inclined parcels receive traction instead of sliding into the bottom transition."""
    import numpy as np

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_warehouse_geometry import warehouse_belt_sections

    for side in ("Left", "Right"):
        for section in warehouse_belt_sections(side):
            if section.belt.curved or abs(section.belt.direction[2]) < 1e-6:
                continue
            vertices = np.asarray(section.geometry.vertices)
            triangles = vertices[np.asarray(section.geometry.faces)]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            top = normals[normals[:, 2] > 1e-8]
            top /= np.linalg.norm(top, axis=1, keepdims=True)
            assert len(top) > 0
            assert np.all(top @ section.belt.surface_normal >= section.belt.contact_threshold)
            assert abs(np.dot(section.belt.direction, section.belt.surface_normal)) < 1e-6
