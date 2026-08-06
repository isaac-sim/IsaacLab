# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the contributed conveyor Franka racetrack geometry."""

from collections import Counter

import pytest

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorForceCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_TOP_Z,
    TURN_SEGMENT_COUNT,
    MeshSpec,
    belt_direction,
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


def test_racetrack_mesh_counts_and_names_are_consistent():
    """Verify each lane has one belt and two uniquely named rail meshes."""
    specs = []
    for side in ("Left", "Right"):
        specs.append(belt_mesh_spec(side))
        specs.extend(guard_mesh_specs(side))

    assert len(specs) == 6
    assert len({spec.name for spec in specs}) == len(specs)
    expected_loop_vertices = 2 * TURN_SEGMENT_COUNT + 2
    for spec in specs:
        assert len(spec.vertices) == 4 * expected_loop_vertices
        assert len(spec.faces) == 8 * expected_loop_vertices


def test_racetrack_meshes_are_watertight():
    """Verify belts and rails have no open or multiply connected triangle edges."""
    for side in ("Left", "Right"):
        for spec in (belt_mesh_spec(side), *guard_mesh_specs(side)):
            assert set(_edge_use_counts(spec).values()) == {2}


def test_belt_top_faces_point_upward():
    """The one-sided triangle-mesh collision surface must support parcels from above."""
    for side in ("Left", "Right"):
        spec = belt_mesh_spec(side)
        for face in spec.faces:
            a, b, c = (spec.vertices[index] for index in face)
            if a[2] == b[2] == c[2] == BELT_TOP_Z:
                cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                assert cross_z > 0.0


def test_racetrack_lanes_counter_rotate():
    """Verify the two analytic conveyor velocity fields use opposite directions."""
    assert belt_direction("Left") == -belt_direction("Right")


@pytest.mark.parametrize(
    ("parameter", "value"),
    (("speed", -0.1), ("friction", -0.1), ("normal_threshold", 1.1)),
)
def test_conveyor_force_config_rejects_invalid_values(parameter: str, value: float):
    """Verify force configuration rejects values outside its physical domain."""
    with pytest.raises(ValueError):
        ConveyorForceCfg(**{parameter: value})
