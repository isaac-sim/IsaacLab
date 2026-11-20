# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared racetrack geometry and velocity-field descriptions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab.controllers import ConveyorSurfaceCfg

PARCEL_COLOR = (0.72, 0.55, 0.35)
"""Cardboard color used by Newton's conveyor example."""

CUBE_COLORS = (
    (0.15, 0.35, 0.90),
    (0.90, 0.20, 0.15),
    (0.15, 0.75, 0.25),
    PARCEL_COLOR,
)
"""Stable sRGB colors for the four numbered transfer cubes."""

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
    belt: ConveyorSurfaceCfg


def belt_direction(side: str) -> float:
    """Return ``1`` for clockwise motion and ``-1`` for counter-clockwise motion."""
    if side == "Left":
        return 1.0
    if side == "Right":
        return -1.0
    raise ValueError(f"Unknown conveyor side: {side!r}.")


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


def racetrack_surface_specs(side: str, **kwargs: float) -> tuple[ConveyorSurfaceCfg, ...]:
    """Configure traction on the authored fixed racetrack sections."""
    sign = belt_direction(side)
    prefix = f"{{ENV_REGEX_NS}}/Racetracks/Conveyor{side}"
    straights = tuple(
        ConveyorSurfaceCfg(prim_path=f"{prefix}{name}StraightCollision", direction=(travel_sign, 0.0, 0.0), **kwargs)
        for name, travel_sign in (("Top", sign), ("Bottom", -sign))
    )
    turns = tuple(
        ConveyorSurfaceCfg(
            prim_path=f"{prefix}{name}TurnCollision",
            direction=(0.0, 0.0, -sign),
            curved=True,
            pivot_point=(BELT_CENTER_X + offset, sign * BELT_CENTER_Y, 0.0),
            radius=BELT_TURN_RADIUS,
            **kwargs,
        )
        for name, offset in (("Right", BELT_HALF_STRAIGHT), ("Left", -BELT_HALF_STRAIGHT))
    )
    return straights + turns


def working_straight_section(side: str, **kwargs: float) -> ConveyorSectionSpec:
    """Retain the original work-facing straight in the extended warehouse route."""
    sign = belt_direction(side)
    run = "Bottom" if sign > 0 else "Top"
    name = f"Conveyor{side}{run}StraightCollision"
    return ConveyorSectionSpec(
        _straight_collision_cuboid(name, sign * BELT_INNER_STRAIGHT_Y),
        ConveyorSurfaceCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", direction=(-1.0, 0.0, 0.0), **kwargs),
    )
