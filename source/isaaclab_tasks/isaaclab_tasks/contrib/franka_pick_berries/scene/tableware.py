# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open plastic punnet, glass receiving bowl and reject dish; dimensions in metres."""

import math

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

# (x, y, outer radius, inner radius, base thickness, total height), task frame.
PLATE = (0.48, 0.0, 0.065, 0.060, 0.004, 0.009)
BOWL = (0.48, 0.16, 0.060, 0.057, 0.004, 0.045)
# Legacy PLATE center/floor remain the spawn-layout reference for CLI compatibility.
# Punnet: (x, y, half width, half depth, corner radius, wall, floor, height).
PUNNET = (0.48, 0.0, 0.055, 0.070, 0.010, 0.001, 0.004, 0.028)
REJECT = (0.40, -0.085, 0.028, 0.026, 0.004, 0.018)
# The spawner replaces (rather than composes with) the asset root pose. Retain
# its authored 55 cm translation and quarter turn; raise its -3 mm top to z=0.
TABLE_POSITION = (0.55, 0.0, 0.003)
TABLE_ROTATION = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


def tableware_solids() -> np.ndarray:
    """Return static contact solids [m], including open mouths (never convex lids).

    Each row is (cx, cy, zmid, a, b, halfheight). Positive a/b describe annular
    cylinders (inner/outer radii). Negative a selects a rounded rectangle with
    half width abs(a), half depth abs(b); negative b selects its hollow wall.
    The punnet corner radius and wall thickness are shared with its render mesh.
    """
    solids = []
    for x, y, outer, inner, base, height in (BOWL, REJECT):
        solids += [(x, y, base / 2, 0, outer, base / 2), (x, y, (height + base) / 2, inner, outer, (height - base) / 2)]
    x, y, hx, hy, _, _, base, height = PUNNET
    solids += [(x, y, base / 2, -hx, hy, base / 2), (x, y, (height + base) / 2, -hx, -hy, (height - base) / 2)]
    return np.asarray(solids, np.float32)


def spawn_tableware(prim_path: str, cfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """Spawn fixed collision proxies in the task frame; RTX authors their visuals."""
    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    for index, (x, y, outer, inner, base, height) in enumerate((BOWL, REJECT)):
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
    # Thin box wall segments approximate the same rounded perimeter for robot contact.
    x, y, hx, hy, corner, thickness, base, height = PUNNET
    floor = UsdGeom.Cube.Define(stage, f"{prim_path}/PunnetFloor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set(Gf.Vec3d(x, y, base / 2))
    floor.AddScaleOp().Set(Gf.Vec3f(2 * hx, 2 * hy, base))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateVisibilityAttr("invisible")
    outline = _rounded_outline(hx - thickness / 2, hy - thickness / 2, corner - thickness / 2)
    for index, (a, b) in enumerate(zip(outline, np.roll(outline, -1, axis=0))):
        delta = b - a
        middle = (a + b) / 2
        wall = UsdGeom.Cube.Define(stage, f"{prim_path}/PunnetWall{index}")
        wall.CreateSizeAttr(1)
        wall.AddTranslateOp().Set(Gf.Vec3d(x + middle[0], y + middle[1], (base + height) / 2))
        wall.AddRotateZOp().Set(math.degrees(math.atan2(delta[1], delta[0])))
        wall.AddScaleOp().Set(Gf.Vec3f(float(np.linalg.norm(delta)) + 0.0001, thickness, height - base))
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        wall.CreateVisibilityAttr("invisible")
    return root.GetPrim()


def _rounded_outline(hx: float, hy: float, radius: float) -> np.ndarray:
    """Counterclockwise rounded rectangle perimeter [m]."""
    points = []
    for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for angle in np.radians(np.linspace(start, start + 90, 9)):
            points.append(
                (sx * (hx - radius) + radius * math.cos(angle), sy * (hy - radius) + radius * math.sin(angle))
            )
    return np.asarray(points)


def _punnet_mesh(stage: Usd.Stage) -> UsdGeom.Mesh:
    """Closed thin shell with an open mouth, rounded corners and rolled flange."""
    x, y, hx, hy, radius, wall, base, height = PUNNET
    # Traverse underside -> outer wall -> flange -> inner wall -> inner floor.
    profile = [
        (hx, hy, radius, 0.0016),
        (hx, hy, radius, base),
        (hx, hy, radius, height - 0.002),
        (hx + 0.003, hy + 0.003, radius + 0.003, height - 0.001),
        (hx + 0.003, hy + 0.003, radius + 0.003, height),
        (hx - wall, hy - wall, radius - wall, height),
        (hx - wall, hy - wall, radius - wall, base),
    ]
    points, faces, counts = [], [], []
    for width, depth, corner, z in profile:
        outline = _rounded_outline(width, depth, corner)
        points.extend((float(px + x), float(py + y), z) for px, py in outline)
    n = len(outline)
    for ring in range(len(profile) - 1):
        for j in range(n):
            k = (j + 1) % n
            faces.extend((ring * n + j, ring * n + k, (ring + 1) * n + k, (ring + 1) * n + j))
            counts.append(4)
    # Bottom and inner floor are separate surfaces: never cap the mouth.
    for ring, z, reverse in ((0, 0.0016, True), (len(profile) - 1, base, False)):
        pole = len(points)
        points.append((x, y, z))
        for j in range(n):
            a, b = ring * n + j, ring * n + (j + 1) % n
            faces.extend((pole, b, a) if reverse else (pole, a, b))
            counts.append(3)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tableware/Punnet")
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def _punnet_material(stage: Usd.Stage, name: str, opacity: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"/World/Tableware/{name}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
    # Thin clear packaging approximation avoids dark multi-bounce solid-glass
    # reflections on the table. The receiving bowl retains its refractive MDL.
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR_Opacity.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR_Opacity", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.98, 1.0))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.15)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _add_punnet(stage: Usd.Stage) -> None:
    mesh = _punnet_mesh(stage)
    material = _punnet_material(stage, "PunnetMaterial", 0.02)
    accent = _punnet_material(stage, "PunnetMoldedEdges", 0.14)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    x, y, hx, hy, corner, _, base, height = PUNNET
    # Thicker rolled/molded details catch light while the thin panels remain clear.
    n = len(_rounded_outline(hx, hy, corner))
    lip = UsdGeom.Subset.Define(stage, "/World/Tableware/Punnet/RolledRim")
    lip.CreateElementTypeAttr("face")
    lip.CreateFamilyNameAttr("materialBind")
    lip.CreateIndicesAttr(list(range(2 * n, 5 * n)))
    UsdShade.MaterialBindingAPI.Apply(lip.GetPrim()).Bind(accent)
    # Molded ribs are appearance-only, outside the smooth collision envelope.
    for axis, half, other in ((0, hx, hy), (1, hy, hx)):
        for side in (-1, 1):
            for index, along in enumerate(np.arange(-other + corner + 0.005, other - corner, 0.009)):
                rib = UsdGeom.Capsule.Define(stage, f"/World/Tableware/PunnetRib{axis}_{int(side > 0)}_{index}")
                rib.CreateRadiusAttr(0.0008)
                rib.CreateHeightAttr(height - base - 0.005)
                rib.CreateAxisAttr("Z")
                xy = [x, y]
                xy[axis] += side * half
                xy[1 - axis] += float(along)
                rib.AddTranslateOp().Set(Gf.Vec3d(*xy, (height + base) / 2))
                UsdShade.MaterialBindingAPI.Apply(rib.GetPrim()).Bind(accent)


def _lathe(
    stage: Usd.Stage, path: str, profile: list[tuple[float, float]], center: tuple[float, float]
) -> UsdGeom.Mesh:
    """Revolve a closed (radius, height) outline [m], preserving an open mouth."""
    segments = 128
    points, rings = [], []
    for r, z in profile:
        # Weld each axis pole and use triangle fans instead of collapsed quads.
        angles = [0.0] if r == 0 else np.linspace(0, 2 * math.pi, segments, endpoint=False)
        rings.append(list(range(len(points), len(points) + len(angles))))
        points.extend((center[0] + r * math.cos(a), center[1] + r * math.sin(a), z) for a in angles)
    faces, counts = [], []
    for i, ring in enumerate(rings):
        following = rings[(i + 1) % len(rings)]
        if len(ring) == len(following) == 1:
            continue  # Revolving an axis segment creates no surface.
        for j in range(segments):
            k = (j + 1) % segments
            if len(ring) == 1:
                face = [ring[0], following[k], following[j]]
            elif len(following) == 1:
                face = [ring[j], ring[k], following[0]]
            else:
                face = [ring[j], ring[k], following[k], following[j]]
            faces.extend(face)
            counts.append(len(face))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def add_tableware_visuals(stage: Usd.Stage) -> None:
    """Author open container render meshes; no external asset sidecars."""
    UsdGeom.Xform.Define(stage, "/World/Tableware")
    _add_punnet(stage)
    for name, spec in (("Reject", REJECT), ("Bowl", BOWL)):
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
        mesh = _lathe(stage, f"/World/Tableware/{name}", profile, (x, y))
        material = UsdShade.Material.Define(stage, f"/World/Tableware/{name}Material")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
        if name == "Bowl":
            shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.98, 1.0, 0.995))
            shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.47)
            shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.03)
            shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        else:
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.50, 0.53, 0.56))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.85)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
