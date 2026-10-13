# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural geometry and materials for the G1 terrain courses."""

import itertools
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

GRANULAR_TOP = 2.5 * math.tan(math.radians(10.0))
PLATFORM_HEIGHTS = (
    (0.51, 0.56, 0.50),
    (0.56, 0.51, 0.58),
    (0.50, 0.56, 0.53),
    (0.55, 0.50, 0.58),
    (0.50, 0.57, 0.51),
    (0.54, 0.51, 0.56),
)


def visual_material(stage, path, color, roughness=0.9, texture=None):
    """Create a matte material, optionally with a repeating texture."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    if texture is not None:
        uv = UsdShade.Shader.Define(stage, f"{path}/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        for axis in ("S", "T"):
            tex.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return material


def _physics_material(stage, path, friction):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    return material


def _mesh(stage, name, vertices, faces, material=None, physics=None, uv_scale=1.0):
    mesh = UsdGeom.Mesh.Define(stage, f"/Course/{name}")
    vertices = np.asarray(vertices, dtype=np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(vertices[:, :2] / uv_scale)
        )
    if physics is not None:
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(physics, materialPurpose="physics")
        mesh.GetPrim().CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1)
    return mesh


def _textures(directory):
    """Generate deterministic stone and plywood surface textures."""
    size = 1024
    rng = np.random.default_rng(20261009)
    gravel = Image.new("RGB", (size, size), (105, 102, 94))
    draw = ImageDraw.Draw(gravel)
    stone_colors = [(67, 69, 68), (88, 86, 80), (112, 108, 98), (132, 127, 115), (153, 148, 135), (95, 99, 101)]
    for _ in range(10000):
        x, y = rng.integers(-12, size + 12, 2)
        rx, ry = int(rng.integers(2, 9)), int(rng.integers(2, 7))
        color = stone_colors[int(rng.integers(len(stone_colors)))]
        shadow = tuple(max(0, c - 24) for c in color)
        light = tuple(min(255, c + 18) for c in color)
        draw.ellipse((x - rx + 1, y - ry + 2, x + rx + 2, y + ry + 3), fill=shadow)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=color)
        draw.arc((x - rx, y - ry, x + rx, y + ry), 195, 330, fill=light, width=1)
    gravel.filter(ImageFilter.GaussianBlur(0.35)).save(directory / "gravel.png")

    rng = np.random.default_rng(42)
    pixels = np.clip(np.array([190, 153, 99]) + rng.normal(0, 3, (512, 512, 1)), 0, 255).astype(np.uint8)
    wood = Image.fromarray(pixels)
    draw = ImageDraw.Draw(wood)
    for _ in range(6500):
        center = rng.uniform(0, 512, 2)
        angle = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]])
        length, width = rng.uniform(5, 28), rng.uniform(0.5, 3)
        corners = [
            tuple(center + du * u * length + dv * v * width)
            for du, dv in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
        ]
        color = tuple(np.clip(np.array([190, 153, 99]) + rng.normal(0, 13), 0, 255).astype(int))
        draw.polygon(corners, fill=color)
    wood.save(directory / "wood.png")


def _platforms_and_ramps(stage, directory, extended_exit):
    colors = {
        "white": (0.73, 0.75, 0.75),
        "steel": (0.26, 0.28, 0.29),
        "black": (0.055, 0.06, 0.065),
        "wood": (0.62, 0.43, 0.20),
        "wall": (0.68, 0.67, 0.64),
        "floor": (0.39, 0.39, 0.37),
        "gravel": (0.32, 0.32, 0.30),
    }
    materials = {
        name: visual_material(
            stage,
            f"/Course/Looks/{name}",
            color,
            texture=directory / f"{name}.png" if name in ("wood", "gravel") else None,
        )
        for name, color in colors.items()
    }
    physics = _physics_material(stage, "/Course/Physics", 1.0)

    def box(name, center, size, material):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(center)
        _mesh(stage, name, mesh.vertices, mesh.faces, materials[material], physics, 0.6)

    xmin = -2.9 if extended_exit else -1.4
    box("floor", (1.8, 1, -0.05), (18, 14, 0.1), "floor")
    box("wall", (xmin - 0.05, 1.5, 2.3), (0.10, 16, 4.6), "wall")
    box("wood_base", (1.8, 0, 0.42), (3.6, 1.6, 0.06), "black")
    for i in range(6):
        for j in range(2):
            x0, y0 = i * 0.6, -0.8 + j * 0.8
            x1, y1 = x0 + 0.6, y0 + 0.8
            sx = math.tan(math.radians(10)) * (1 if (i + j) % 3 else -1)
            sy = math.tan(math.radians(10)) * (1 if j else -1) * (0.35 if i in (2, 3) else 0.0)
            top = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=float)
            top[:, 2] = 0.45 + sx * (top[:, 0] - (x0 + x1) / 2) + sy * (top[:, 1] - (y0 + y1) / 2) + 0.053
            vertices = np.vstack([top, top.copy()])
            vertices[4:, 2] = 0.42
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 6, 5],
                    [4, 7, 6],
                    [0, 4, 5],
                    [0, 5, 1],
                    [1, 5, 6],
                    [1, 6, 2],
                    [2, 6, 7],
                    [2, 7, 3],
                    [3, 7, 4],
                    [3, 4, 0],
                ]
            )
            _mesh(stage, f"wood_{i}_{j}", vertices, faces, materials["wood"], physics, 0.6)
    for i, row in enumerate(PLATFORM_HEIGHTS):
        for j, height in enumerate(row):
            x, y = (i + 0.5) * 0.6, 1.05 + (j + 0.5) * 0.5
            box(f"white_{i}_{j}", (x, y, height - 0.015), (0.59, 0.49, 0.03), "white")
            for k, (dx, dy) in enumerate(((-0.23, -0.18), (0.23, -0.18), (-0.23, 0.18), (0.23, 0.18))):
                box(f"leg_{i}_{j}_{k}", (x + dx, y + dy, (height - 0.03) / 2), (0.045, 0.045, height - 0.03), "steel")
    # Both lanes contribute to the on-board depth image.
    box("gravel_lane", (-0.35, 1.25, 0.465), (0.7, 4.1, 0.03), "gravel")
    box("left_platform", (xmin / 2, 1.25, 0.425), (-xmin, 4.1, 0.05), "black")
    box("right_platform", (4.3, 1.25, 0.425), (1.4, 4.1, 0.05), "black")
    for i in range(4):
        box(
            f"background_step_{i}",
            (0, 3.15 + i * 0.3, 0.45 + (i + 1) * 0.075 / 2),
            (0.85, 0.3, (i + 1) * 0.075),
            "black",
        )


def _container_mesh():
    """Triangulate a closed tray without overlapping internal contact surfaces."""
    bottom = GRANULAR_TOP - 0.03
    axes = (
        np.array([5.4, 5.5, 11.5, 11.6]),
        np.array([-0.7, -0.6, 0.6, 0.7]),
        np.array([bottom - 0.1, bottom, GRANULAR_TOP]),
    )
    filled = {(i, j, k) for i, j, k in itertools.product(range(3), range(3), range(2)) if k == 0 or i != 1 or j != 1}
    vertices, faces, ids = [], [], {}
    for cell in sorted(filled):
        for axis, side in itertools.product(range(3), (-1, 1)):
            neighbor = list(cell)
            neighbor[axis] += side
            if tuple(neighbor) in filled:
                continue
            other = [a for a in range(3) if a != axis]
            quad = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                index = list(cell)
                index[axis] += int(side > 0)
                index[other[0]] += u
                index[other[1]] += v
                index = tuple(index)
                if index not in ids:
                    ids[index] = len(vertices)
                    vertices.append([axes[a][index[a]] for a in range(3)])
                quad.append(ids[index])
            a, b, c = np.asarray(vertices)[quad[:3]]
            if np.cross(b - a, c - a)[axis] * side < 0:
                quad.reverse()
            faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _granular(stage, directory, is_physx):
    for name, start, end in (("approach", -1.5, 5.5), ("end", 11.5, 16.5)):
        x = np.round(np.arange(start, end + 0.025, 0.05), 6)
        y = np.round(np.arange(-4.0, 4.025, 0.05), 6)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.clip((xx - 1.0) / 2.5, 0, 1) * GRANULAR_TOP
        vertices = np.stack((xx, yy, zz), -1).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(len(x) - 1), np.arange(len(y) - 1), indexing="ij")
        a = (i * len(y) + j).ravel()
        b, c, d = a + len(y), a + len(y) + 1, a + 1
        faces = np.concatenate((np.stack((a, b, c), -1), np.stack((a, c, d), -1)))
        material = visual_material(
            stage, f"/Course/Looks/{name}", (0.055, 0.065, 0.08), 0.96, directory / "asphalt.png"
        )
        physics = _physics_material(stage, f"/Course/{name}_material", 1.0)
        mesh = _mesh(stage, name, vertices, faces, material, physics)
        mesh.CreateDisplayColorAttr([(0.22, 0.23, 0.25)])
    vertices, faces = _container_mesh()
    tray = _mesh(stage, "granular_container", vertices, faces)
    tray.CreateDisplayColorAttr([(0.18, 0.19, 0.20)])
    if is_physx:
        from .granular import create_pbd_beds  # noqa: PLC0415

        tray.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
        create_pbd_beds(stage)


def create_terrain(directory: Path, terrain_name: str, is_physx: bool = False) -> Path:
    """Write a course and its procedural textures into the run's asset directory."""
    directory.mkdir(parents=True, exist_ok=True)
    _textures(directory)
    path = directory / "course.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Course").GetPrim())
    if terrain_name == "granular":
        _granular(stage, directory, is_physx)
    else:
        _platforms_and_ramps(stage, directory, extended_exit=terrain_name == "platforms")
    stage.GetRootLayer().Save()
    return path
