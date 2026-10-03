# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical surfaces for the USD-authored elevated conveyor network."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from isaaclab.physics import SurfaceVelocitySpec

from .conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_COLLISION_OVERHANG,
    BELT_COLLISION_SEAM_OVERLAP,
    BELT_HALF_STRAIGHT,
    BELT_THICKNESS,
    GUARD_BASE_OVERLAP,
    GUARD_HEIGHT,
    GUARD_THICKNESS,
    ConveyorSectionSpec,
    MeshSpec,
    _turn_collision_mesh,
    belt_collision_section_specs,
    belt_direction,
)

_ROUTE_ASSET = Path(__file__).parent / "assets" / "conveyor_routes.usda"


@lru_cache(maxsize=1)
def _route_layer():
    # Import Usd to register file formats in kitless mode, without composing asset dependencies.
    from pxr import Sdf, Usd  # noqa: F401

    return Sdf.Layer.FindOrOpen(str(_ROUTE_ASSET))


@dataclass(frozen=True)
class _RouteSegment:
    name: str
    points: tuple[tuple[float, float, float], ...]
    direction: tuple[float, float, float]
    width: float
    pivot: tuple[float, float, float] | None
    radius: float | None
    friction: float | None
    velocity: float | None
    path: str


@lru_cache(maxsize=2)
def _route_segments(side: str) -> tuple[_RouteSegment, ...]:
    """Read the same centerlines that author the visible conveyor modules."""
    from pxr import Sdf

    belt_direction(side)
    layer = _route_layer()
    root = layer.GetPrimAtPath(Sdf.Path(f"/ConveyorRoutes/{side}"))
    segments = []
    for name in root.attributes["conveyor:order"].default:
        prim = layer.GetPrimAtPath(root.path.AppendChild(name))
        points = layer.GetPrimAtPath(prim.path.AppendChild("Centerline")).attributes["points"].default
        curved = "conveyor:pivot" in prim.attributes
        segments.append(
            _RouteSegment(
                name=name,
                velocity=prim.attributes["conveyor:velocity"].default
                if "conveyor:velocity" in prim.attributes
                else None,
                path=prim.attributes["conveyor:path"].default if "conveyor:path" in prim.attributes else "main",
                points=tuple(tuple(point) for point in points),
                direction=tuple(prim.attributes["conveyor:direction"].default),
                width=prim.attributes["conveyor:width"].default,
                pivot=tuple(prim.attributes["conveyor:pivot"].default) if curved else None,
                radius=prim.attributes["conveyor:radius"].default if curved else None,
                friction=prim.attributes["conveyor:friction"].default
                if "conveyor:friction" in prim.attributes
                else None,
            )
        )
    return tuple(segments)


def warehouse_parcel_positions() -> tuple[tuple[float, float, float], ...]:
    """Return physical parcel infeed spawn positions in workspace coordinates [m]."""
    root = _route_layer().GetPrimAtPath("/ConveyorRoutes")
    return tuple(tuple(position) for position in root.attributes["conveyor:parcelSpawnPositions"].default)


def _normals(segment: _RouteSegment) -> np.ndarray:
    if segment.pivot is None:
        tangent = np.asarray(segment.direction)
        normal = np.array((-tangent[1], tangent[0], 0.0))
        return np.tile(normal / np.linalg.norm(normal), (len(segment.points), 1))
    radial = np.asarray(segment.points) - segment.pivot
    radial[:, 2] = 0
    return -math.copysign(1, segment.direction[2]) * radial / np.linalg.norm(radial, axis=1, keepdims=True)


def _prism(
    name: str,
    points: np.ndarray,
    normals: np.ndarray,
    offset: float | np.ndarray,
    width: float,
    bottom: float,
    top: float,
    *,
    closed: bool = False,
) -> MeshSpec:
    """Sweep a closed solid along a three-dimensional centerline [m]."""
    offset = np.broadcast_to(offset, (len(points),))[:, None]
    left = points + normals * (offset - width / 2)
    right = points + normals * (offset + width / 2)
    vertices = np.concatenate((left + (0, 0, top), right + (0, 0, top), left + (0, 0, bottom), right + (0, 0, bottom)))
    count = len(points)
    quads = []
    for i in range(count if closed else count - 1):
        j = (i + 1) % count
        quads.extend(
            (
                (i, j, count + j, count + i),
                (2 * count + i, 3 * count + i, 3 * count + j, 2 * count + j),
                (i, 2 * count + i, 2 * count + j, j),
                (count + i, count + j, 3 * count + j, 3 * count + i),
            )
        )
    if not closed:
        quads.extend(((0, count, 3 * count, 2 * count), (count - 1, 3 * count - 1, 4 * count - 1, 2 * count - 1)))
    faces = tuple(triangle for a, b, c, d in quads for triangle in ((a, b, c), (a, c, d)))
    return MeshSpec(name, tuple(tuple(vertex) for vertex in vertices), faces)


def warehouse_belt_sections(side: str, **kwargs: float | bool) -> tuple[ConveyorSectionSpec, ...]:
    """Build linked belt surfaces from USD while preserving the original manipulation geometry."""
    sign = belt_direction(side)
    sections = []
    for segment in _route_segments(side):
        name = f"Conveyor{side}{segment.name}Collision"
        pivot = segment.pivot
        surface_kwargs = dict(kwargs)
        if segment.velocity is not None:
            surface_kwargs["velocity"] = segment.velocity
        if segment.friction is not None:
            surface_kwargs["friction_coefficient"] = segment.friction
        if segment.name == "Working":
            original = belt_collision_section_specs(side, **kwargs)[1 if sign > 0 else 0]
            sections.append(original)
            continue
        if segment.name in {"PickupBend", "PlacementBend"}:
            pickup = segment.name == "PickupBend"
            name = f"Conveyor{side}{'Left' if pickup else 'Right'}InnerTurnCollision"
            pivot = (BELT_CENTER_X + (-BELT_HALF_STRAIGHT if pickup else BELT_HALF_STRAIGHT), sign * BELT_CENTER_Y, 0.0)
            geometry = _turn_collision_mesh(
                name, pivot[0], BELT_CENTER_Y, math.pi if pickup else -math.pi / 2, math.pi / 2
            )
            if sign < 0:
                geometry = replace(
                    geometry,
                    vertices=tuple((x, -y, z) for x, y, z in geometry.vertices),
                    faces=tuple((a, c, b) for a, b, c in geometry.faces),
                )
        else:
            points = np.array(segment.points)
            # Overlap adjacent panels without exposing a vertical collision seam.
            for index, neighbor, direction in ((0, 1, -1), (-1, -2, 1)):
                tangent = points[neighbor] - points[index] if index == 0 else points[index] - points[neighbor]
                points[index] += direction * BELT_COLLISION_SEAM_OVERLAP * tangent / np.linalg.norm(tangent)
            geometry = _prism(
                name, points, _normals(segment), 0, segment.width + 2 * BELT_COLLISION_OVERHANG, -BELT_THICKNESS, 0
            )
        sections.append(
            ConveyorSectionSpec(
                geometry,
                SurfaceVelocitySpec(
                    prim_path=f"{{ENV_REGEX_NS}}/{geometry.name}",
                    direction=segment.direction,
                    surface_normal=(0.0, 0.0, 1.0)
                    if segment.pivot is not None
                    else tuple(np.cross(segment.direction, _normals(segment)[0])),
                    curved=segment.pivot is not None,
                    pivot_point=pivot or (0, 0, 0),
                    radius=segment.radius,
                    **surface_kwargs,
                ),
            )
        )
    return tuple(sections)


def warehouse_guard_meshes(side: str) -> tuple[MeshSpec, ...]:
    """Build continuous guides for each closed circulation route and open supply belt."""
    segments = _route_segments(side)
    guards = []
    for path in dict.fromkeys(segment.path for segment in segments):
        points, normals, offsets = [], [], []
        route = [segment for segment in segments if segment.path == path]
        for index, segment in enumerate(route):
            stop = None if path != "main" and index == len(route) - 1 else -1
            for point, normal in zip(segment.points[:stop], _normals(segment)[:stop]):
                points.append(point)
                normals.append(normal)
                offsets.append((segment.width + GUARD_THICKNESS) / 2)
        for boundary, sign in (("Inner", -1), ("Outer", 1)):
            guards.append(
                _prism(
                    f"Guard{side}{path.title()}{boundary}",
                    np.asarray(points),
                    np.asarray(normals),
                    sign * np.asarray(offsets),
                    GUARD_THICKNESS,
                    -GUARD_BASE_OVERLAP,
                    GUARD_HEIGHT,
                    closed=path == "main",
                )
            )
    return tuple(guards)
