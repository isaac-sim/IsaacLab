# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, reproducible coffee props with hollow MPM collision geometry.

The meshes are authored in meters with Z up. The bowl and cup origins are on
the table. The spoon origin is the inner bottom of its bowl, its handle points
along -X, and +Z points out of the bowl. Generated USD and texture files live in
the user cache; no downloaded assets or generated binaries are required.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from filelock import FileLock

if TYPE_CHECKING:
    import trimesh

    from pxr import Usd, UsdGeom, UsdShade

TABLE_TOP_Z = 0.0
BOWL_POSITION = (0.52, -0.12, TABLE_TOP_Z)
CUP_POSITION = (0.60, 0.04, TABLE_TOP_Z)
BOWL_INNER_RADIUS = 0.065
BOWL_HEIGHT = 0.040
BOWL_BOTTOM_THICKNESS = 0.004
# The centered oval contains the coffee pile, which keeps its circular seed
# region.
BOWL_SCALE = (1.23, 1.15, 1.0)
BOWL_CAVITY_OFFSET = (0.0, 0.0, 0.0)
CUP_INNER_RADIUS = 0.039
CUP_HEIGHT = 0.085
CUP_BOTTOM_THICKNESS = 0.004
SPOON_BOWL_LENGTH = 0.050
SPOON_BOWL_WIDTH = 0.032
SPOON_BOWL_DEPTH = 0.010
SPOON_BOWL_THICKNESS = 0.0018
SPOON_HANDLE_LENGTH = 0.095
SPOON_HANDLE_HEIGHT = 0.010
COFFEE_HEAP_APEX_HEIGHT = 0.080

# Ordered from the flat cavity bottom to its upper rim. This profile also owns
# particle seeding clearance, so changes to the bowl do not invalidate its fill.
_BOWL_INNER_PROFILE = (
    (0.038, BOWL_BOTTOM_THICKNESS),
    (0.043, 0.006),
    (0.049, 0.012),
    (0.056, 0.022),
    (0.062, 0.032),
    (BOWL_INNER_RADIUS, BOWL_HEIGHT - 0.001),
    (0.066, BOWL_HEIGHT),
)


def coffee_cavity_radius(height: float | np.ndarray) -> float | np.ndarray:
    """Return the circular seed radius through the cavity and an above-rim heap.

    The circular profile fits the centered oval cavity below its rim.
    Above the rim it tapers to zero at ``COFFEE_HEAP_APEX_HEIGHT`` without adding
    collision walls. Seeders should subtract their particle radius and collider
    margin from this radius. Heights outside the profile return zero.
    """
    profile = np.asarray((*_BOWL_INNER_PROFILE, (0.0, COFFEE_HEAP_APEX_HEIGHT)))
    radius = np.interp(height, profile[:, 1], profile[:, 0], left=0.0, right=0.0)
    return float(radius) if np.isscalar(height) else radius


def bowl_mesh() -> trimesh.Trimesh:
    """Return an oval ceramic shell with an open cavity and rounded lip."""
    profile = [
        (0.0, BOWL_BOTTOM_THICKNESS),
        *_BOWL_INNER_PROFILE,
        (0.068, BOWL_HEIGHT),
        (0.069, BOWL_HEIGHT - 0.001),
        (0.067, 0.030),
        (0.061, 0.019),
        (0.052, 0.008),
        (0.044, 0.0),
        (0.0, 0.0),
    ]
    mesh = _lathe(profile)
    mesh.apply_scale(BOWL_SCALE)
    mesh.apply_translation(BOWL_CAVITY_OFFSET)
    return mesh


def cup_mesh() -> trimesh.Trimesh:
    """Return a watertight cup wall and bottom, without capping its mouth."""
    return _lathe(
        [
            (0.0, CUP_BOTTOM_THICKNESS),
            (CUP_INNER_RADIUS - 0.003, CUP_BOTTOM_THICKNESS),
            (CUP_INNER_RADIUS - 0.001, 0.006),
            (CUP_INNER_RADIUS, 0.010),
            (CUP_INNER_RADIUS, CUP_HEIGHT - 0.001),
            (CUP_INNER_RADIUS + 0.0007, CUP_HEIGHT),
            (CUP_INNER_RADIUS + 0.0020, CUP_HEIGHT),
            (CUP_INNER_RADIUS + 0.0027, CUP_HEIGHT - 0.001),
            (CUP_INNER_RADIUS + 0.0027, 0.004),
            (CUP_INNER_RADIUS + 0.001, 0.0),
            (0.0, 0.0),
        ]
    )


def cup_handle_mesh() -> trimesh.Trimesh:
    """Return a capped C-shaped handle embedded in the cup at both ends.

    Sweep a round tube around half an ellipse in the XZ plane. The ends lie
    inside the ceramic wall rather than touching it at a single tangent point.
    The handle stays outside the opening and is visual decoration only.
    """
    import trimesh

    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 64)
    radial_angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    centers = np.column_stack(
        (CUP_INNER_RADIUS + 0.0018 + 0.031 * np.cos(angles), np.zeros_like(angles), 0.044 + 0.024 * np.sin(angles))
    )
    normals = np.column_stack((0.024 * np.cos(angles), np.zeros_like(angles), 0.031 * np.sin(angles)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rings = centers[:, None, :] + 0.0038 * (
        normals[:, None, :] * np.cos(radial_angles)[None, :, None]
        + np.array((0.0, 1.0, 0.0)) * np.sin(radial_angles)[None, :, None]
    )
    sides = len(radial_angles)
    vertices = np.concatenate((rings.reshape(-1, 3), centers[[0, -1]]))
    faces = []
    for ring in range(len(angles) - 1):
        for side in range(sides):
            a = ring * sides + side
            b = ring * sides + (side + 1) % sides
            faces.extend(((a, b, b + sides), (a, b + sides, a + sides)))
    start, end = len(vertices) - 2, len(vertices) - 1
    last_ring = (len(angles) - 1) * sides
    for side in range(sides):
        following = (side + 1) % sides
        faces.extend(((start, following, side), (end, last_ring + side, last_ring + following)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def spoon_mesh() -> trimesh.Trimesh:
    """Return the closed, concave spoon bowl; the handle is a separate mesh.

    The shallow bowl retains its original 1.8 mm vertical shell. Convexifying
    this mesh would close the cavity and prevent material from being carried.
    """
    profile = [(0.0, 0.0)]
    radii = np.linspace(0.1, 1.0, 14)
    profile.extend((float(r), SPOON_BOWL_DEPTH * float(r) ** 2) for r in radii)
    profile.extend((float(r), SPOON_BOWL_DEPTH * float(r) ** 2 - SPOON_BOWL_THICKNESS) for r in radii[::-1])
    profile.append((0.0, -SPOON_BOWL_THICKNESS))
    mesh = _lathe(profile)
    mesh.apply_scale((SPOON_BOWL_LENGTH / 2.0, SPOON_BOWL_WIDTH / 2.0, 1.0))
    return mesh


def spoon_handle_mesh() -> trimesh.Trimesh:
    """Return the original straight, flattened handle."""
    import trimesh

    radius = 0.0052
    rear_x = -0.025 - (SPOON_HANDLE_LENGTH - 2.0 * radius) + 0.003
    front_x = -0.022
    handle = trimesh.creation.capsule(radius=radius, height=front_x - rear_x, count=(12, 16))
    handle.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (0.0, 1.0, 0.0)))
    handle.apply_scale((1.0, 1.0, 0.26))
    handle.apply_translation(((front_x + rear_x) / 2.0, 0.0, SPOON_HANDLE_HEIGHT))
    return handle


def build_coffee_assets(cache_dir: str | Path | None = None) -> dict[str, str]:
    """Build cached bowl, cup, spoon and tabletop USD files and return their paths.

    USD roots for the bowl, cup and table are static; the spoon is a kinematic
    rigid body. All collision meshes retain exact triangles. The cup has an
    opaque sage exterior and an ivory interior.

    Args:
        cache_dir: Parent cache directory. Defaults to the Isaac Lab user cache.

    Returns:
        Absolute USD paths indexed by ``bowl``, ``cup``, ``spoon`` and ``table``.
    """
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "isaaclab"
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    directory = Path(cache_dir).expanduser().resolve() / "rizon_sharpa_coffee" / digest
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: str(directory / f"{name}.usda") for name in ("bowl", "cup", "spoon", "table")}
    with FileLock(str(directory / ".build.lock")):
        if all(Path(path).is_file() for path in paths.values()):
            return paths
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            temporary = Path(temporary)
            _write_bowl(temporary / "bowl.usda")
            _write_cup(temporary / "cup.usda")
            _write_spoon(temporary / "spoon.usda")
            _write_table(temporary / "table.usda")
            for generated in temporary.iterdir():
                generated.replace(directory / generated.name)
    return paths


def _lathe(profile: list[tuple[float, float]]) -> trimesh.Trimesh:
    """Revolve a cross-section, merging pole vertices and orienting faces outward."""
    import trimesh

    mesh = trimesh.creation.revolve(np.asarray(profile), sections=96)
    mesh.fix_normals()
    return mesh


def _stage(path: Path, *, rigid: bool = False) -> Usd.Stage:
    """Create a Z-up stage with a default prim and a shared contact material."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if rigid:
        physics = UsdPhysics.RigidBodyAPI.Apply(root)
        physics.CreateRigidBodyEnabledAttr(True)
        physics.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.025)
    material = UsdShade.Material.Define(stage, "/Asset/Looks/Contact")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.5)
    physics.CreateDynamicFrictionAttr(0.4)
    physics.CreateRestitutionAttr(0.0)
    return stage


def _material(
    stage: Usd.Stage,
    name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.35,
    metallic: float = 0.0,
    opacity: float = 1.0,
    texture: str | None = None,
) -> UsdShade.Material:
    """Author a portable preview material, optionally with a UV-mapped texture."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, f"/Asset/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.47)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if texture is not None:
        reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Coordinates")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        sampler = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        sampler.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        sampler.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
    return material


def _mesh(
    stage: Usd.Stage,
    name: str,
    geometry: trimesh.Trimesh,
    material: UsdShade.Material,
    *,
    collision: bool = True,
    uv: np.ndarray | None = None,
    smooth: bool = True,
) -> UsdGeom.Mesh:
    """Author visual geometry and, when requested, a separate exact collider.

    Newton hides colliders when a body has visual-only shapes. Separate leaves
    preserve these surfaces alongside decoration and allow material subsets to
    follow the visual-mesh importer without changing physical shape counts.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
    mesh.CreatePointsAttr(geometry.vertices.astype(np.float32).tolist())
    mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    # Smooth box corners tilt the tabletop's shading normals and cause black
    # ray-traced patches beneath the robot. Keep each planar face's true normal.
    normals = geometry.vertex_normals if smooth else np.repeat(geometry.face_normals, 3, axis=0)
    mesh.CreateNormalsAttr(normals.astype(np.float32).tolist())
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex if smooth else UsdGeom.Tokens.faceVarying)
    mesh.CreateExtentAttr(geometry.bounds.astype(np.float32).tolist())
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collision:
        collider = UsdGeom.Mesh.Define(stage, f"/Asset/{name}Collision")
        collider.CreatePointsAttr(mesh.GetPointsAttr().Get())
        collider.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
        collider.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
        collider.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("none")
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            UsdShade.Material.Get(stage, "/Asset/Looks/Contact"), materialPurpose="physics"
        )
    if uv is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            uv.astype(np.float32).tolist()
        )
    return mesh


def _vessel_mesh(
    stage: Usd.Stage,
    name: str,
    geometry: trimesh.Trimesh,
    interior: UsdShade.Material,
    exterior: UsdShade.Material,
    *,
    center: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Author an open vessel with distinct inner and outer glazes."""
    from pxr import UsdShade

    mesh = _mesh(stage, name, geometry, interior)
    # The outer walls face radially away from the center; the inner walls face in.
    radial_centers = geometry.triangles_center[:, :2] - np.asarray(center)
    radial_dot = np.sum(geometry.face_normals[:, :2] * radial_centers, axis=1)
    faces = np.flatnonzero((radial_dot > 0.0) | (geometry.face_normals[:, 2] < -0.8))
    subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset("Exterior", faces.tolist())
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(exterior)


def _write_bowl(path: Path) -> None:
    """Create a glazed ivory bowl with a muted teal exterior."""
    stage = _stage(path)
    ivory = _material(stage, "IvoryGlaze", (0.78, 0.70, 0.53), roughness=0.20)
    teal = _material(stage, "TealGlaze", (0.026, 0.115, 0.125), roughness=0.22)
    _vessel_mesh(stage, "Bowl", bowl_mesh(), ivory, teal, center=BOWL_CAVITY_OFFSET[:2])
    stage.GetRootLayer().Save()


def _write_cup(path: Path) -> None:
    """Create an opaque sage ceramic cup with an ivory interior and loop handle."""
    stage = _stage(path)
    ivory = _material(stage, "IvoryGlaze", (0.82, 0.77, 0.67), roughness=0.28)
    sage = _material(stage, "SageGlaze", (0.12, 0.23, 0.19), roughness=0.32)
    _vessel_mesh(stage, "Cup", cup_mesh(), ivory, sage)
    _mesh(stage, "Handle", cup_handle_mesh(), sage, collision=False)
    stage.GetRootLayer().Save()


def _write_spoon(path: Path) -> None:
    """Create the shallow polished steel spoon and its straight flattened handle."""
    stage = _stage(path, rigid=True)
    steel = _material(stage, "PolishedSteel", (0.72, 0.76, 0.78), roughness=0.17, metallic=1.0)
    _mesh(stage, "Bowl", spoon_mesh(), steel)
    # Separate closed meshes preserve the concave cavity at their overlaps.
    _mesh(stage, "Handle", spoon_handle_mesh(), steel)
    stage.GetRootLayer().Save()


def _write_table(path: Path) -> None:
    """Create a walnut studio tabletop, linen runner and understated brass trim."""
    import trimesh

    from pxr import UsdGeom

    stage = _stage(path)
    _write_walnut_texture(path.parent / "walnut.png")
    walnut = _material(stage, "Walnut", (0.19, 0.082, 0.033), roughness=0.57, texture="walnut.png")
    brass = _material(stage, "Brass", (0.56, 0.37, 0.13), roughness=0.27, metallic=0.85)
    linen = _material(stage, "NaturalLinen", (0.47, 0.43, 0.35), roughness=0.95)
    charcoal = _material(stage, "Charcoal", (0.035, 0.040, 0.037), roughness=0.55)
    tabletop = trimesh.creation.box((1.20, 0.90, 0.044))
    tabletop.apply_translation((0.40, 0.0, -0.022))
    uv = np.column_stack(((tabletop.vertices[:, 0] + 0.20) / 1.20, (tabletop.vertices[:, 1] + 0.45) / 0.90))
    _mesh(stage, "WalnutTop", tabletop, walnut, uv=uv, smooth=False)
    for index, y in enumerate((-0.447, 0.447)):
        rail = trimesh.creation.box((1.18, 0.002, 0.002))
        rail.apply_translation((0.40, y, -0.004))
        _mesh(stage, f"BrassTrim{index}", rail, brass, collision=False, smooth=False)
    runner = trimesh.creation.box((0.34, 0.68, 0.0007))
    runner.apply_translation((0.55, 0.01, -0.00005))
    _mesh(stage, "LinenRunner", runner, linen, collision=False, smooth=False)
    # Fine stitched borders give the fabric scale without adding collision work.
    thread = _material(stage, "LinenStitch", (0.40, 0.34, 0.24), roughness=1.0)
    for index, x in enumerate((0.386, 0.714)):
        seam = trimesh.creation.box((0.0006, 0.67, 0.0002))
        seam.apply_translation((x, 0.01, 0.0004))
        _mesh(stage, f"RunnerSeam{index}", seam, thread, collision=False, smooth=False)
    for index, x in enumerate((-0.05, 0.85)):
        leg = trimesh.creation.box((0.060, 0.62, 0.70))
        leg.apply_translation((x, 0.0, -0.394))
        _mesh(stage, f"Pedestal{index}", leg, charcoal, collision=False, smooth=False)
    floor_material = _material(stage, "StudioFloor", (0.018, 0.023, 0.026), roughness=0.95)
    floor = trimesh.creation.box((30.0, 30.0, 0.006))
    floor.apply_translation((0.40, 0.0, -0.749))
    _mesh(stage, "StudioFloor", floor, floor_material, smooth=False)
    # Keep the visible top while giving falling grains a resolved collision slab.
    # The 6 mm visual shell can be crossed in one step by fast escaped grains.
    floor.vertices[floor.vertices[:, 2] < -0.749, 2] = -0.796
    floor_collider = UsdGeom.Mesh.Get(stage, "/Asset/StudioFloorCollision")
    floor_collider.GetPointsAttr().Set(floor.vertices.astype(np.float32).tolist())
    stage.GetRootLayer().Save()


def _write_walnut_texture(path: Path) -> None:
    """Synthesize a deterministic wood grain texture without external assets."""
    from PIL import Image

    rng = np.random.default_rng(19)
    y, x = np.mgrid[0:768, 0:1024].astype(np.float32)
    x /= 1024.0
    y /= 768.0
    flow = y + 0.017 * np.sin(6.0 * x + 9.0 * y) + 0.008 * np.sin(17.0 * x - 6.0 * y)
    grain = 0.015 * np.sin(230.0 * flow) + 0.006 * np.sin(740.0 * flow + 2.0 * x)
    grain += 0.004 * np.sin(1800.0 * flow) + 0.024 * np.sin(15.0 * flow)
    grain += rng.normal(0.0, 0.004, x.shape)
    colors = np.array((0.30, 0.185, 0.11), dtype=np.float32)[None, None, :] + grain[:, :, None]
    Image.fromarray(np.uint8(255.0 * np.clip(colors, 0.0, 1.0))).save(path)
