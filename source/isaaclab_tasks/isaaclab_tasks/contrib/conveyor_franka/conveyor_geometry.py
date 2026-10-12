# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared racetrack geometry and velocity-field descriptions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec

BELT_COLOR = (0.09, 0.09, 0.09)
"""Dark-rubber color used by Newton's conveyor example."""

GUARD_COLOR = (0.66, 0.69, 0.74)
"""Brushed-metal color used by Newton's conveyor example."""

PARCEL_COLOR = (0.72, 0.55, 0.35)
CUBE_COLORS = (
    (0.15, 0.35, 0.90),
    (0.90, 0.20, 0.15),
    (0.15, 0.75, 0.25),
    PARCEL_COLOR,
)
"""Stable sRGB colors for the four numbered transfer cubes."""
"""Cardboard color used by Newton's conveyor example."""

BELT_CENTER_X = 0.58
BELT_CENTER_Y = 0.51
BELT_TURN_RADIUS = 0.24
BELT_HALF_STRAIGHT = 0.44
BELT_WIDTH = 0.15
BELT_THICKNESS = 0.04
BELT_TOP_Z = 0.04
TURN_SEGMENT_COUNT = 96

# Four deployment slots place one cube on each straight run of the two
# racetracks.  The x coordinates are mirrored about the racetrack center, so
# the inner/outer pair on a belt is separated by exactly half a lap.
BELT_INNER_STRAIGHT_Y = BELT_CENTER_Y - BELT_TURN_RADIUS
BELT_OUTER_STRAIGHT_Y = BELT_CENTER_Y + BELT_TURN_RADIUS
CUBE_INNER_SLOT_X = BELT_CENTER_X - BELT_HALF_STRAIGHT / 3.0
CUBE_OUTER_SLOT_X = BELT_CENTER_X + BELT_HALF_STRAIGHT / 3.0

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
class CuboidSpec:
    """Native cuboid and semantic name for one static racetrack component."""

    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class ConveyorSectionSpec:
    """Task geometry paired with its backend-neutral conveyor description."""

    geometry: MeshSpec | CuboidSpec
    belt: SurfaceVelocitySpec


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


def _prism_faces(count: int, *, closed: bool) -> tuple[tuple[int, int, int], ...]:
    """Triangulate top, bottom, and side walls for inner/outer vertex rings.

    Vertices are grouped as inner top, outer top, inner bottom, outer bottom.
    Counter-clockwise centerlines produce outward faces; open arcs also receive end caps.
    """
    quads = []
    for i in range(count if closed else count - 1):
        j = (i + 1) % count
        quads.extend(
            (
                (i, count + i, count + j, j),
                (2 * count + i, 2 * count + j, 3 * count + j, 3 * count + i),
                (3 * count + i, 3 * count + j, count + j, count + i),
                (2 * count + i, i, j, 2 * count + j),
            )
        )
    if not closed:
        end = count - 1
        quads.extend(((0, 2 * count, 3 * count, count), (end, count + end, 3 * count + end, 2 * count + end)))
    return tuple(triangle for a, b, c, d in quads for triangle in ((a, b, c), (a, c, d)))


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

    # Racetrack centerlines are clockwise; reverse the shared counter-clockwise winding.
    faces = tuple((a, c, b) for a, b, c in _prism_faces(len(centerline), closed=True))
    return MeshSpec(name=name, vertices=points, faces=faces)


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


def _straight_collision_cuboid(name: str, center_y: float) -> CuboidSpec:
    """Build one solid native cuboid for a straight conveyor section."""
    half_width = 0.5 * BELT_WIDTH + BELT_COLLISION_OVERHANG
    x_min = BELT_CENTER_X - BELT_HALF_STRAIGHT - BELT_COLLISION_SEAM_OVERLAP
    x_max = BELT_CENTER_X + BELT_HALF_STRAIGHT + BELT_COLLISION_SEAM_OVERLAP
    return CuboidSpec(
        name=name,
        size=(x_max - x_min, 2.0 * half_width, BELT_THICKNESS),
        position=(0.5 * (x_min + x_max), center_y, BELT_TOP_Z - 0.5 * BELT_THICKNESS),
    )


def _turn_collision_mesh(
    name: str, pivot_x: float, center_y: float, start_angle: float, angle_span: float = math.pi
) -> MeshSpec:
    """Build a closed annular turn with a +Z top surface and an angular span [rad]."""
    half_width = 0.5 * BELT_WIDTH + BELT_COLLISION_OVERHANG
    inner_radius = BELT_TURN_RADIUS - half_width
    outer_radius = BELT_TURN_RADIUS + half_width
    angle_overlap = BELT_COLLISION_SEAM_OVERLAP / BELT_TURN_RADIUS
    angle_start = start_angle - angle_overlap
    segment_count = round(TURN_SEGMENT_COUNT * angle_span / math.pi)
    angle_step = (angle_span + 2.0 * angle_overlap) / segment_count
    angles = tuple(angle_start + index * angle_step for index in range(segment_count + 1))

    inner_top = tuple(
        (pivot_x + inner_radius * math.cos(angle), center_y + inner_radius * math.sin(angle), BELT_TOP_Z)
        for angle in angles
    )
    outer_top = tuple(
        (pivot_x + outer_radius * math.cos(angle), center_y + outer_radius * math.sin(angle), BELT_TOP_Z)
        for angle in angles
    )
    bottom_z = BELT_TOP_Z - BELT_THICKNESS
    inner_bottom = tuple((x, y, bottom_z) for x, y, _ in inner_top)
    outer_bottom = tuple((x, y, bottom_z) for x, y, _ in outer_top)

    return MeshSpec(
        name=name,
        vertices=inner_top + outer_top + inner_bottom + outer_bottom,
        faces=_prism_faces(len(inner_top), closed=False),
    )


def belt_collision_geometry_specs(side: str) -> tuple[CuboidSpec | MeshSpec, ...]:
    """Build native straight and closed-mesh turn collision geometry for one racetrack."""
    center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
    left_x = BELT_CENTER_X - BELT_HALF_STRAIGHT
    right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
    return (
        _straight_collision_cuboid(f"Conveyor{side}TopStraightCollision", center_y + BELT_TURN_RADIUS),
        _straight_collision_cuboid(f"Conveyor{side}BottomStraightCollision", center_y - BELT_TURN_RADIUS),
        _turn_collision_mesh(f"Conveyor{side}RightTurnCollision", right_x, center_y, -0.5 * math.pi),
        _turn_collision_mesh(f"Conveyor{side}LeftTurnCollision", left_x, center_y, 0.5 * math.pi),
    )


def belt_collision_section_specs(
    side: str,
    *,
    velocity: float = 0.0,
    friction_coefficient: float = 0.7,
    contact_threshold: float = 0.997,
    enabled: bool = True,
) -> tuple[ConveyorSectionSpec, ConveyorSectionSpec, ConveyorSectionSpec, ConveyorSectionSpec]:
    """Build collision geometry and authored conveyor intent for one racetrack."""
    center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
    left_x = BELT_CENTER_X - BELT_HALF_STRAIGHT
    right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
    direction = belt_direction(side)
    top_straight, bottom_straight, right_turn, left_turn = belt_collision_geometry_specs(side)

    def belt(
        geometry: MeshSpec | CuboidSpec,
        travel_direction: tuple[float, float, float],
        *,
        curved: bool = False,
        pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float | None = None,
    ) -> ConveyorSectionSpec:
        return ConveyorSectionSpec(
            geometry=geometry,
            belt=SurfaceVelocitySpec(
                prim_path=f"{{ENV_REGEX_NS}}/{geometry.name}",
                velocity=velocity,
                enabled=enabled,
                direction=travel_direction,
                curved=curved,
                pivot_point=pivot_point,
                radius=radius,
                contact_threshold=contact_threshold,
                friction_coefficient=friction_coefficient,
            ),
        )

    return (
        belt(top_straight, (direction, 0.0, 0.0)),
        belt(bottom_straight, (-direction, 0.0, 0.0)),
        belt(
            right_turn,
            (0.0, 0.0, -direction),
            curved=True,
            pivot_point=(right_x, center_y, 0.0),
            radius=BELT_TURN_RADIUS,
        ),
        belt(
            left_turn,
            (0.0, 0.0, -direction),
            curved=True,
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
