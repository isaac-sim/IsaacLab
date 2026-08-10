# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared racetrack geometry and velocity-field descriptions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

BELT_COLOR = (0.09, 0.09, 0.09)
"""Dark-rubber color used by Newton's conveyor example."""

GUARD_COLOR = (0.66, 0.69, 0.74)
"""Brushed-metal color used by Newton's conveyor example."""

PARCEL_COLOR = (0.72, 0.55, 0.35)
"""Cardboard color used by Newton's conveyor example."""

BELT_CENTER_X = 0.58
BELT_CENTER_Y = 0.51
BELT_TURN_RADIUS = 0.24
BELT_HALF_STRAIGHT = 0.44
BELT_WIDTH = 0.15
BELT_THICKNESS = 0.04
BELT_TOP_Z = 0.04
TURN_SEGMENT_COUNT = 96

# Collision surfaces extend underneath the rails and overlap at section seams.
# This keeps the dynamic parcels on a continuous +Z-facing surface without
# exposing the belt prism's vertical side faces to the contact solver.
BELT_COLLISION_OVERHANG = 0.02
BELT_COLLISION_SEAM_OVERLAP = 0.004

GUARD_THICKNESS = 0.018
# Keep the rails below the parcel tops so both lanes remain easy to read from
# the default oblique camera.
GUARD_HEIGHT = 0.02
GUARD_BASE_OVERLAP = 0.005


@dataclass(frozen=True)
class MeshSpec:
    """Triangle mesh and semantic name for one static racetrack component."""

    name: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ConveyorSectionSpec:
    """Collision mesh and velocity field for one conveyor section.

    The direction and pivot are expressed in the collision prim's local frame.
    Constant sections interpret ``direction`` as the unit travel direction;
    pivot sections interpret it as the unit rotation axis and use ``radius``
    to convert commanded linear speed to angular speed. ``surface_normal`` is
    also local, so rotated or inclined sections need no world-space special case.
    """

    mesh: MeshSpec
    velocity_field_type: Literal["constant", "pivot"]
    direction: tuple[float, float, float]
    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float | None = None
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


def belt_direction(side: str) -> float:
    """Return ``1`` for clockwise motion and ``-1`` for counter-clockwise motion."""
    if side == "Left":
        return 1.0
    if side == "Right":
        return -1.0
    raise ValueError(f"Unknown conveyor side: {side!r}.")


def _racetrack_centerline(center_y: float) -> tuple[tuple[float, float, float, float], ...]:
    """Sample a clockwise racetrack centerline with an outward normal at every point."""
    left_x = BELT_CENTER_X - BELT_HALF_STRAIGHT
    right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
    radius = BELT_TURN_RADIUS
    points: list[tuple[float, float, float, float]] = [
        (left_x, center_y + radius, 0.0, 1.0),
        (right_x, center_y + radius, 0.0, 1.0),
    ]

    for index in range(1, TURN_SEGMENT_COUNT + 1):
        angle = 0.5 * math.pi - index * math.pi / TURN_SEGMENT_COUNT
        normal_x = math.cos(angle)
        normal_y = math.sin(angle)
        points.append((right_x + radius * normal_x, center_y + radius * normal_y, normal_x, normal_y))

    points.append((left_x, center_y - radius, 0.0, -1.0))
    for index in range(1, TURN_SEGMENT_COUNT):
        angle = -0.5 * math.pi - index * math.pi / TURN_SEGMENT_COUNT
        normal_x = math.cos(angle)
        normal_y = math.sin(angle)
        points.append((left_x + radius * normal_x, center_y + radius * normal_y, normal_x, normal_y))
    return tuple(points)


def _racetrack_prism_mesh(
    name: str,
    center_y: float,
    lateral_offset: float,
    width: float,
    z_min: float,
    z_max: float,
) -> MeshSpec:
    """Build one closed prism following a racetrack centerline."""
    centerline = _racetrack_centerline(center_y)
    half_width = 0.5 * width
    outer_offset = lateral_offset + half_width
    inner_offset = lateral_offset - half_width

    outer_top = [(x + nx * outer_offset, y + ny * outer_offset, z_max) for x, y, nx, ny in centerline]
    inner_top = [(x + nx * inner_offset, y + ny * inner_offset, z_max) for x, y, nx, ny in centerline]
    outer_bottom = [(x + nx * outer_offset, y + ny * outer_offset, z_min) for x, y, nx, ny in centerline]
    inner_bottom = [(x + nx * inner_offset, y + ny * inner_offset, z_min) for x, y, nx, ny in centerline]
    points = tuple(inner_top + outer_top + inner_bottom + outer_bottom)

    count = len(centerline)
    outer_top_offset = count
    inner_bottom_offset = 2 * count
    outer_bottom_offset = 3 * count
    indices: list[int] = []
    for index in range(count):
        next_index = (index + 1) % count
        inner_top_i = index
        inner_top_j = next_index
        outer_top_i = outer_top_offset + index
        outer_top_j = outer_top_offset + next_index
        inner_bottom_i = inner_bottom_offset + index
        inner_bottom_j = inner_bottom_offset + next_index
        outer_bottom_i = outer_bottom_offset + index
        outer_bottom_j = outer_bottom_offset + next_index

        # Top, bottom, outer wall, and inner wall; two triangles per surface.
        indices.extend((inner_top_i, outer_top_i, outer_top_j, inner_top_i, outer_top_j, inner_top_j))
        indices.extend(
            (
                inner_bottom_i,
                inner_bottom_j,
                outer_bottom_j,
                inner_bottom_i,
                outer_bottom_j,
                outer_bottom_i,
            )
        )
        indices.extend(
            (
                outer_bottom_i,
                outer_bottom_j,
                outer_top_j,
                outer_bottom_i,
                outer_top_j,
                outer_top_i,
            )
        )
        indices.extend(
            (
                inner_bottom_i,
                inner_top_i,
                inner_top_j,
                inner_bottom_i,
                inner_top_j,
                inner_bottom_j,
            )
        )

    # The centerline is sampled clockwise so its tangent matches the positive
    # conveyor direction. The face pattern above is written for a
    # counter-clockwise ring, so reverse every triangle to keep its collision
    # normal outward (most importantly, the belt top must point +Z).
    for triangle_start in range(0, len(indices), 3):
        indices[triangle_start + 1], indices[triangle_start + 2] = (
            indices[triangle_start + 2],
            indices[triangle_start + 1],
        )

    return MeshSpec(
        name=name,
        vertices=points,
        faces=tuple(tuple(indices[offset : offset + 3]) for offset in range(0, len(indices), 3)),
    )


def belt_mesh_spec(side: str) -> MeshSpec:
    """Build the seamless, watertight visual mesh for one conveyor belt."""
    center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
    return _racetrack_prism_mesh(
        name=f"Conveyor{side}BeltVisual",
        center_y=center_y,
        lateral_offset=0.0,
        width=BELT_WIDTH,
        z_min=BELT_TOP_Z - BELT_THICKNESS,
        z_max=BELT_TOP_Z,
    )


def _straight_collision_mesh(name: str, center_y: float) -> MeshSpec:
    """Build one horizontal straight collision surface with +Z face winding."""
    half_width = 0.5 * BELT_WIDTH + BELT_COLLISION_OVERHANG
    x_min = BELT_CENTER_X - BELT_HALF_STRAIGHT - BELT_COLLISION_SEAM_OVERLAP
    x_max = BELT_CENTER_X + BELT_HALF_STRAIGHT + BELT_COLLISION_SEAM_OVERLAP
    vertices = (
        (x_min, center_y - half_width, BELT_TOP_Z),
        (x_max, center_y - half_width, BELT_TOP_Z),
        (x_max, center_y + half_width, BELT_TOP_Z),
        (x_min, center_y + half_width, BELT_TOP_Z),
    )
    return MeshSpec(name=name, vertices=vertices, faces=((0, 1, 2), (0, 2, 3)))


def _turn_collision_mesh(name: str, pivot_x: float, center_y: float, start_angle: float) -> MeshSpec:
    """Build one horizontal annular half-turn collision surface with +Z normals."""
    half_width = 0.5 * BELT_WIDTH + BELT_COLLISION_OVERHANG
    inner_radius = BELT_TURN_RADIUS - half_width
    outer_radius = BELT_TURN_RADIUS + half_width
    angle_overlap = BELT_COLLISION_SEAM_OVERLAP / BELT_TURN_RADIUS
    angle_start = start_angle - angle_overlap
    angle_step = (math.pi + 2.0 * angle_overlap) / TURN_SEGMENT_COUNT
    angles = tuple(angle_start + index * angle_step for index in range(TURN_SEGMENT_COUNT + 1))

    inner = tuple(
        (pivot_x + inner_radius * math.cos(angle), center_y + inner_radius * math.sin(angle), BELT_TOP_Z)
        for angle in angles
    )
    outer = tuple(
        (pivot_x + outer_radius * math.cos(angle), center_y + outer_radius * math.sin(angle), BELT_TOP_Z)
        for angle in angles
    )
    outer_offset = len(inner)
    faces: list[tuple[int, int, int]] = []
    for index in range(TURN_SEGMENT_COUNT):
        next_index = index + 1
        faces.extend(
            (
                (index, outer_offset + index, outer_offset + next_index),
                (index, outer_offset + next_index, next_index),
            )
        )
    return MeshSpec(name=name, vertices=inner + outer, faces=tuple(faces))


def belt_collision_mesh_specs(side: str) -> tuple[MeshSpec, MeshSpec, MeshSpec, MeshSpec]:
    """Build straight and pivot-field collision sections for one racetrack."""
    return tuple(section.mesh for section in belt_collision_section_specs(side))


def belt_collision_section_specs(
    side: str,
) -> tuple[ConveyorSectionSpec, ConveyorSectionSpec, ConveyorSectionSpec, ConveyorSectionSpec]:
    """Build collision meshes and their matching conveyor velocity fields."""
    center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
    left_x = BELT_CENTER_X - BELT_HALF_STRAIGHT
    right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
    direction = belt_direction(side)
    return (
        ConveyorSectionSpec(
            mesh=_straight_collision_mesh(f"Conveyor{side}TopStraightCollision", center_y + BELT_TURN_RADIUS),
            velocity_field_type="constant",
            direction=(direction, 0.0, 0.0),
        ),
        ConveyorSectionSpec(
            mesh=_straight_collision_mesh(f"Conveyor{side}BottomStraightCollision", center_y - BELT_TURN_RADIUS),
            velocity_field_type="constant",
            direction=(-direction, 0.0, 0.0),
        ),
        ConveyorSectionSpec(
            mesh=_turn_collision_mesh(f"Conveyor{side}RightTurnCollision", right_x, center_y, -0.5 * math.pi),
            velocity_field_type="pivot",
            direction=(0.0, 0.0, -direction),
            pivot_point=(right_x, center_y, 0.0),
            radius=BELT_TURN_RADIUS,
        ),
        ConveyorSectionSpec(
            mesh=_turn_collision_mesh(f"Conveyor{side}LeftTurnCollision", left_x, center_y, 0.5 * math.pi),
            velocity_field_type="pivot",
            direction=(0.0, 0.0, -direction),
            pivot_point=(left_x, center_y, 0.0),
            radius=BELT_TURN_RADIUS,
        ),
    )


def guard_mesh_specs(side: str) -> tuple[MeshSpec, MeshSpec]:
    """Build seamless inner and outer guardrail meshes for one racetrack."""
    center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
    rail_offset = 0.5 * (BELT_WIDTH + GUARD_THICKNESS)
    z_min = BELT_TOP_Z - GUARD_BASE_OVERLAP
    z_max = BELT_TOP_Z + GUARD_HEIGHT
    return (
        _racetrack_prism_mesh(
            name=f"Guard{side}Inner",
            center_y=center_y,
            lateral_offset=-rail_offset,
            width=GUARD_THICKNESS,
            z_min=z_min,
            z_max=z_max,
        ),
        _racetrack_prism_mesh(
            name=f"Guard{side}Outer",
            center_y=center_y,
            lateral_offset=rail_offset,
            width=GUARD_THICKNESS,
            z_min=z_min,
            z_max=z_max,
        ),
    )
