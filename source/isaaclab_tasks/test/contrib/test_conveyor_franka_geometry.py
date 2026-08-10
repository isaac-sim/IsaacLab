# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the contributed conveyor Franka racetrack geometry."""

import sys
from collections import Counter

import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.contrib.conveyor_franka.conveyor_force_driver import (
    BeltContact,
    _integrate_encoders,
    _prepare_contact_patches,
    _update_effective_velocities,
)
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorForceCfg, ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_TOP_Z,
    BELT_TURN_RADIUS,
    TURN_SEGMENT_COUNT,
    CuboidSpec,
    MeshSpec,
    belt_collision_section_specs,
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


def test_collision_sections_and_velocity_fields_share_one_description():
    """Straight and curved force fields stay aligned with robust collision geometry."""
    for side in ("Left", "Right"):
        sections = belt_collision_section_specs(side)

        assert len(sections) == 4
        assert [section.velocity_field_type for section in sections] == ["constant", "constant", "pivot", "pivot"]
        assert all(isinstance(section.geometry, CuboidSpec) for section in sections[:2])
        for section in sections[2:]:
            assert isinstance(section.geometry, MeshSpec)
            assert set(_edge_use_counts(section.geometry).values()) == {2}
        assert all(section.radius == BELT_TURN_RADIUS for section in sections[2:])
        assert sections[0].direction == tuple(-value for value in sections[1].direction)
        assert sections[2].direction == sections[3].direction


def _make_belt_contact(conveyor: int, force: float, next_contact: int) -> BeltContact:
    """Build one horizontal contact for the patch-normalization kernel."""
    contact = BeltContact()
    contact.valid = 1
    contact.body = 0
    contact.conveyor = conveyor
    contact.point = wp.vec3()
    contact.normal = wp.vec3(0.0, 0.0, 1.0)
    contact.normal_force = force
    contact.next_body_contact = next_contact
    return contact


@pytest.mark.parametrize(
    ("conveyors", "expected_forces"),
    (((0, 0), (4.0, 6.0)), ((0, 1), (5.0, 5.0))),
)
def test_contact_patch_normalizes_only_across_overlapping_sections(conveyors, expected_forces):
    """A seam preserves total load without perturbing contacts on one section."""
    contacts = wp.array(
        [
            _make_belt_contact(conveyors[0], 4.0, 1),
            _make_belt_contact(conveyors[1], 6.0, -1),
        ],
        dtype=BeltContact,
        device="cpu",
    )
    body_contact_head = wp.array([0], dtype=wp.int32, device="cpu")
    body_q = wp.array([wp.transform()], dtype=wp.transform, device="cpu")
    body_com = wp.array([wp.vec3()], dtype=wp.vec3, device="cpu")
    patch_head = wp.full(2, -1, dtype=wp.int32, device="cpu")
    adjusted_force = wp.zeros(2, dtype=wp.float32, device="cpu")
    splitting_scale = wp.zeros(2, dtype=wp.float32, device="cpu")

    wp.launch(
        _prepare_contact_patches,
        dim=1,
        inputs=[contacts, body_contact_head, body_q, body_com],
        outputs=[patch_head, adjusted_force, splitting_scale],
        device="cpu",
    )

    np.testing.assert_allclose(adjusted_force.numpy(), expected_forces)
    np.testing.assert_allclose(splitting_scale.numpy(), (0.5, 0.5))
    np.testing.assert_allclose(adjusted_force.numpy().sum(), 10.0)


def test_disabled_surface_remembers_command_and_stops_encoder():
    """One effective-speed seam drives both traction and encoder state."""
    commanded = wp.array([2.0], dtype=wp.float32, device="cpu")
    enabled = wp.array([0], dtype=wp.int32, device="cpu")
    effective = wp.zeros(1, dtype=wp.float32, device="cpu")
    encoder = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        _update_effective_velocities,
        dim=1,
        inputs=[commanded, enabled],
        outputs=[effective],
        device="cpu",
    )
    wp.launch(_integrate_encoders, dim=1, inputs=[0.5, effective], outputs=[encoder], device="cpu")
    np.testing.assert_allclose(effective.numpy(), (0.0,))
    np.testing.assert_allclose(encoder.numpy(), (0.0,))

    commanded.assign(np.array([3.0], dtype=np.float32))
    enabled.fill_(1)
    wp.launch(
        _update_effective_velocities,
        dim=1,
        inputs=[commanded, enabled],
        outputs=[effective],
        device="cpu",
    )
    wp.launch(_integrate_encoders, dim=1, inputs=[0.5, effective], outputs=[encoder], device="cpu")
    np.testing.assert_allclose(effective.numpy(), (3.0,))
    np.testing.assert_allclose(encoder.numpy(), (1.5,))


def test_environment_config_without_optional_visualizers(monkeypatch):
    """The task configuration remains usable without the visualizer package."""
    monkeypatch.setitem(sys.modules, "isaaclab_visualizers", None)

    cfg = ConveyorFrankaEnvCfg()

    assert cfg.sim.default_visualizer_cfg is None


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("speed", -0.1),
        ("friction", -0.1),
        ("normal_threshold", 1.1),
        ("startup_duration_s", 0.0),
        ("transported_body_count_per_env", 0),
        ("transported_body_pattern", "["),
    ),
)
def test_conveyor_force_config_rejects_invalid_values(parameter: str, value: object):
    """Verify force configuration rejects values outside its physical domain."""
    with pytest.raises(ValueError):
        ConveyorForceCfg(**{parameter: value})
