# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Durable geometry checks for the contributed conveyor Franka task."""

from collections import Counter

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import _collision_properties, _cube
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_TOP_Z,
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
