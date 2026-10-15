# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The lab table, the open punnet, the glass or porcelain receiving bowl and the reject dish; dimensions in metres."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path

from ..assets.asset_paths import punnet_asset_path

# (x, y, outer radius, inner radius, base thickness, total height), task frame.
BOWL = (0.48, 0.16, 0.060, 0.057, 0.004, 0.030)
REJECT_DISH = (0.40, -0.085, 0.028, 0.026, 0.004, 0.018)
PUNNET_CENTER = (0.48, 0.0)
"""Center (x, y) [m] of the punnet on the table, in the task frame."""
PUNNET_FLOOR = 0.004
"""Height [m] of the punnet's inner floor, on which the berries rest."""


@dataclass(frozen=True)
class PunnetSpec:
    """A punnet's visual asset and the simple collision geometry that follows its cavity.

    The inner wall is a rounded rectangle that tapers toward the floor: its half extents and corner radius grow by
    :attr:`flare` per unit height above the floor. Its fine relief (ribs, slots) is visual only.
    """

    asset: str
    """File name of the visual USDZ package, whose ``/Punnet`` default prim is centered on the tabletop."""
    floor_outline: tuple[float, float, float]
    """Half width, half depth and corner radius [m] of the inner wall at the floor."""
    flare: float
    """Growth of the inner wall's half extents and corner radius per unit height [m/m]."""
    wall: float
    """Thickness [m] of the collision wall, outside the inner wall."""
    height: float
    """Height [m] of the collision wall's top, just below the rim."""

    def outline(self, z: float) -> tuple[float, float, float]:
        """Return the inner wall's half width, half depth and corner radius [m] at height ``z`` [m]."""
        grow = self.flare * (z - PUNNET_FLOOR)
        return tuple(value + grow for value in self.floor_outline)


# Thermoformed clear PET sheet, 0.35 mm thick: its collision wall is thicker so that thin contact stays robust.
PUNNET = PunnetSpec("clear_pet_container.usdz", (0.0431, 0.0556, 0.0086), 0.0035 / 0.021, 0.001, 0.027)
"""The clear plastic punnet."""
# The spawner replaces (rather than composes with) the asset root pose. Retain
# its authored 55 cm translation and quarter turn. The table's visible top lies
# 1.42 mm above its collision box, whose authored top is at -3 mm: place the
# visible top 0.1 mm below z=0, where the tableware stands, so that no glass or
# plastic face is buried in or coincident with the tabletop.
TABLE_VISIBLE_TOP = 0.00142
TABLE_POSITION = (0.55, 0.0, 0.003 - TABLE_VISIBLE_TOP - 0.0001)
TABLE_ROTATION = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
BOWL_MATERIALS = ("glass", "porcelain")
"""Materials of the receiving bowl."""
PORCELAIN_COLOR = (0.93, 0.92, 0.89)
"""Color [linear RGB] of the porcelain bowl's glaze."""


def tableware_solids(punnet: PunnetSpec) -> np.ndarray:
    """Return static contact solids [m], including open mouths (never convex lids).

    Each row is (cx, cy, zmid, a, b, halfheight). Positive a/b describe annular
    cylinders (inner/outer radii). Negative a selects a rounded rectangle with
    half width abs(a), half depth abs(b); negative b selects its hollow wall, whose
    outline at its bottom is given, and which flares upward by ``punnet.flare``.
    The punnet's outer corner radius and wall thickness go with it: see
    :func:`punnet_wall_contact`.
    """
    solids = []
    for x, y, outer, inner, base, height in (BOWL, REJECT_DISH):
        solids += [(x, y, base / 2, 0, outer, base / 2), (x, y, (height + base) / 2, inner, outer, (height - base) / 2)]
    x, y = PUNNET_CENTER
    hx, hy, _ = (value + punnet.wall for value in punnet.floor_outline)
    middle, half = (punnet.height + PUNNET_FLOOR) / 2, (punnet.height - PUNNET_FLOOR) / 2
    solids += [(x, y, PUNNET_FLOOR / 2, -hx, hy, PUNNET_FLOOR / 2), (x, y, middle, -hx, -hy, half)]
    return np.asarray(solids, np.float32)


def punnet_wall_contact(punnet: PunnetSpec) -> tuple[float, float, float]:
    """Return the outer corner radius [m] at the floor, thickness [m] and flare of the punnet's analytic wall."""
    return punnet.floor_outline[2] + punnet.wall, punnet.wall, punnet.flare


def spawn_tableware(prim_path: str, cfg: TablewareCfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """Spawn fixed collision proxies in the task frame; RTX authors their visuals."""
    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    for index, (x, y, outer, inner, base, height) in enumerate((BOWL, REJECT_DISH)):
        disk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Base{index}")
        disk.CreateRadiusAttr(outer)
        disk.CreateHeightAttr(base)
        disk.CreateAxisAttr("Z")
        disk.AddTranslateOp().Set(Gf.Vec3d(x, y, base / 2))
        UsdPhysics.CollisionAPI.Apply(disk.GetPrim())
        disk.CreateVisibilityAttr("invisible")
        # Individual convex wall segments keep the mouth open in MuJoCo/Newton.
        # Never give the whole bowl to a convex-hull importer.
        segments = 48
        radius = (inner + outer) / 2
        for j in range(segments):
            angle = 2 * math.pi * j / segments
            wall = UsdGeom.Cube.Define(stage, f"{prim_path}/Wall{index}_{j}")
            wall.CreateSizeAttr(1)
            wall.AddTranslateOp().Set(
                Gf.Vec3d(x + radius * math.cos(angle), y + radius * math.sin(angle), (base + height) / 2)
            )
            wall.AddRotateZOp().Set(math.degrees(angle))
            wall.AddScaleOp().Set(Gf.Vec3f(outer - inner, 2 * outer * math.tan(math.pi / segments), height - base))
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
            wall.CreateVisibilityAttr("invisible")
    # A floor slab and thin box wall segments, tilted outward, follow the punnet's tapered cavity.
    punnet = PUNNET
    x, y = PUNNET_CENTER
    hx, hy, corner = punnet.floor_outline
    floor = UsdGeom.Cube.Define(stage, f"{prim_path}/PunnetFloor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set(Gf.Vec3d(x, y, PUNNET_FLOOR / 2))
    floor.AddScaleOp().Set(Gf.Vec3f(2 * (hx + punnet.wall), 2 * (hy + punnet.wall), PUNNET_FLOOR))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateVisibilityAttr("invisible")
    rise = punnet.height - PUNNET_FLOOR
    bottom = _rounded_outline(*(value + punnet.wall / 2 for value in punnet.floor_outline))
    top = _rounded_outline(*(value + punnet.wall / 2 for value in punnet.outline(punnet.height)))
    tilt = math.degrees(math.atan(punnet.flare))
    for index in range(len(bottom)):
        following = (index + 1) % len(bottom)
        delta = (bottom[following] - bottom[index] + top[following] - top[index]) / 2
        middle = (bottom[index] + bottom[following] + top[index] + top[following]) / 4
        wall = UsdGeom.Cube.Define(stage, f"{prim_path}/PunnetWall{index}")
        wall.CreateSizeAttr(1)
        wall.AddTranslateOp().Set(Gf.Vec3d(x + middle[0], y + middle[1], (PUNNET_FLOOR + punnet.height) / 2))
        wall.AddRotateZOp().Set(math.degrees(math.atan2(delta[1], delta[0])))
        # The outline runs counterclockwise: its outward normal is the segment's -y, toward which the top leans.
        wall.AddRotateXOp().Set(tilt)
        wall.AddScaleOp().Set(
            Gf.Vec3f(float(np.linalg.norm(delta)) + 0.0001, punnet.wall, rise / math.cos(math.radians(tilt)))
        )
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        wall.CreateVisibilityAttr("invisible")
    return root.GetPrim()


@configclass
class TablewareCfg(sim_utils.SpawnerCfg):
    """Configuration of the tableware's static collision proxies."""

    func: callable = spawn_tableware


def _rounded_outline(hx: float, hy: float, radius: float) -> np.ndarray:
    """Counterclockwise rounded rectangle perimeter [m]."""
    points = []
    for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for angle in np.radians(np.linspace(start, start + 90, 9)):
            points.append(
                (sx * (hx - radius) + radius * math.cos(angle), sy * (hy - radius) + radius * math.sin(angle))
            )
    return np.asarray(points)


def _no_shadow(geometry: UsdGeom.Gprim) -> None:
    """Keep ``geometry`` from casting shadows, which RTX would make opaque for glass."""
    UsdGeom.PrimvarsAPI(geometry).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)


def _glass_material(
    stage: Usd.Stage, name: str, color: tuple[float, float, float], ior: float, frosting: float, thin_walled: bool
) -> UsdShade.Material:
    """Clear glass or plastic; a thin-walled sheet reflects without bending what is behind it."""
    material = UsdShade.Material.Define(stage, f"/World/Tableware/{name}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(ior)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(frosting)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(thin_walled)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _lathe(
    stage: Usd.Stage,
    path: str,
    profile: list[tuple[float, float]],
    center: tuple[float, float],
    smooth_corners: tuple[int, ...] = (),
) -> UsdGeom.Mesh:
    """Revolve a closed (radius, height) outline [m], preserving an open mouth.

    The normals are smooth around the axis, so that the polygonal revolution renders, and refracts, as a surface of
    revolution rather than as facets. Along the outline they are sharp, except at the ``smooth_corners`` indices of
    ``profile``, where they average the two adjoining sides.
    """
    segments = 128
    points, rings = [], []
    for r, z in profile:
        # Weld each axis pole and use triangle fans instead of collapsed quads.
        ring_angles = [0.0] if r == 0 else np.linspace(0, 2 * math.pi, segments, endpoint=False)
        rings.append(list(range(len(points), len(points) + len(ring_angles))))
        points.extend((center[0] + r * math.cos(a), center[1] + r * math.sin(a), z) for a in ring_angles)

    def side_normal(i):
        """Return the outward (radial, axial) unit normal of the outline side from point ``i``, or None on the axis."""
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % len(profile)]
        if r0 == r1 == 0:
            return None
        normal = np.array([z1 - z0, r0 - r1])
        return normal / np.linalg.norm(normal)

    def corner_normal(i, side):
        """Return the (radial, axial) normal of side ``side`` at its outline point ``i``."""
        own = side_normal(side)
        other = side_normal(i - 1 if side == i else i)
        if i % len(profile) not in smooth_corners or other is None:
            return own
        blended = own + other
        return blended / np.linalg.norm(blended)

    def normal_3d(normal, angle):
        return (normal[0] * math.cos(angle), normal[0] * math.sin(angle), normal[1])

    faces, counts, normals = [], [], []
    for i, ring in enumerate(rings):
        following = rings[(i + 1) % len(rings)]
        if len(ring) == len(following) == 1:
            continue  # Revolving an axis segment creates no surface.
        start, end = corner_normal(i, i), corner_normal((i + 1) % len(profile), i)
        for j in range(segments):
            k = (j + 1) % segments
            # An axis pole takes the direction of its fan triangle's middle.
            a_j, a_k, middle = (2 * math.pi * t / segments for t in (j, k, j + 0.5))
            if len(ring) == 1:
                face = [ring[0], following[k], following[j]]
                corners = [(start, middle), (end, a_k), (end, a_j)]
            elif len(following) == 1:
                face = [ring[j], ring[k], following[0]]
                corners = [(start, a_j), (start, a_k), (end, middle)]
            else:
                face = [ring[j], ring[k], following[k], following[j]]
                corners = [(start, a_j), (start, a_k), (end, a_k), (end, a_j)]
            faces.extend(face)
            counts.append(len(face))
            normals.extend(normal_3d(normal, angle) for normal, angle in corners)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def _porcelain_material(stage: Usd.Stage, path: str) -> UsdShade.Material:
    """Glossy glazed white porcelain."""
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*PORCELAIN_COLOR))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
    surface.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.02)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def add_tableware_visuals(stage: Usd.Stage, bowl_material: str = "porcelain") -> None:
    """Reference the punnet's visual asset and author the bowl's and reject dish's open render meshes.

    Args:
        stage: Stage to author the meshes in.
        bowl_material: Material of the receiving bowl, one of :data:`BOWL_MATERIALS`.
    """
    if bowl_material not in BOWL_MATERIALS:
        raise ValueError(f"Unknown bowl material {bowl_material!r}; expected one of {BOWL_MATERIALS}")
    UsdGeom.Xform.Define(stage, "/World/Tableware")
    # The package is centered on the tabletop, with its floor's support height in its geometry.
    package = UsdGeom.Xform.Define(stage, "/World/Tableware/Punnet")
    package.GetPrim().GetReferences().AddReference(retrieve_file_path(punnet_asset_path(PUNNET.asset)))
    package.AddTranslateOp().Set(Gf.Vec3d(*PUNNET_CENTER, 0.0))
    for name, spec in (("Reject", REJECT_DISH), ("Bowl", BOWL)):
        x, y, outer, inner, base, height = spec
        bevel = min(0.001, (outer - inner) / 3)
        # Closed solid cross-section: underside, outside, lip, inside, inner floor.
        profile = [
            (0, 0),
            (outer - bevel, 0),
            (outer, bevel),
            (outer, height - bevel),
            (outer - bevel, height),
            (inner + bevel, height),
            (inner, height - bevel),
            (inner, base + bevel),
            (inner - bevel, base),
            (0, base),
        ]
        if name == "Bowl":
            # A shallow recessed underside avoids coincident glass/table faces.
            # Only the outer foot ring touches the tabletop.
            profile[:1] = [(0, 0.0007), (outer - 0.004, 0.0007), (outer - 0.002, 0)]
        # Round the bowl's foot ring into its recess, and its 1 mm lip top into the lip bevels.
        smooth_corners = (2, 6, 7) if name == "Bowl" else ()
        mesh = _lathe(stage, f"/World/Tableware/{name}", profile, (x, y), smooth_corners)
        if name == "Bowl" and bowl_material == "porcelain":
            material = _porcelain_material(stage, "/World/Tableware/BowlMaterial")
        elif name == "Bowl":
            # RTX shadow rays do not refract: solid glass would cast an opaque
            # shadow and leave the tabletop seen through its floor black.
            _no_shadow(mesh)
            material = _glass_material(stage, "BowlMaterial", (0.98, 1.0, 0.995), 1.47, 0.03, False)
        else:
            material = UsdShade.Material.Define(stage, f"/World/Tableware/{name}Material")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.50, 0.53, 0.56))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.85)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
