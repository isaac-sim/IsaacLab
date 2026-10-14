# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed platform, ramp, and granular courses reconstructed from real-world traversals."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.terrains import TerrainImporterCfg

    from .fixed_course_env_cfg import FixedCourseCfg


def _apply_texture(stage: Usd.Stage, material_path: str, texture: Path) -> None:
    """Connect a repeating texture to a configured preview surface."""
    from pxr import Sdf, UsdShade

    uv = UsdShade.Shader.Define(stage, f"{material_path}/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    for axis in ("S", "T"):
        tex.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Shader"))
    shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")


def _mesh(path, vertices, faces, cfg: FixedCourseCfg, material=None, physics=None, uv_scale=1.0):
    from pxr import Sdf, UsdGeom, Vt

    mesh_cfg = sim_utils.MeshFileCfg(
        mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=vertices, faces=faces),
        collision_props=cfg.collision_props if physics else [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        mesh_collision_props=cfg.mesh_collision_props if physics else None,
    )
    mesh_cfg.func(path, mesh_cfg)
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/mesh"))
    mesh.CreateSubdivisionSchemeAttr("none")
    if material is not None:
        sim_utils.bind_visual_material(mesh.GetPath(), material)
        vertices = np.asarray(vertices, dtype=np.float32)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(vertices[:, :2] / uv_scale)
        )
    if physics is not None:
        sim_utils.bind_physics_material(mesh.GetPath(), physics)
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


def _platforms_and_ramps(cfg: FixedCourseCfg):
    from isaaclab.utils.mesh import convert_faces_to_triangles  # noqa: PLC0415

    parts = []

    def box(name, center, size, material):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(center)
        parts.append((name, material, mesh))

    xmin = -2.9 if cfg.layout == "platforms" else -1.4
    box("floor", (1.8, 1, -0.05), (18, 14, 0.1), "floor")
    box("wall", (xmin - 0.05, 1.5, 2.3), (0.10, 16, 4.6), "wall")
    box("wood_base", (1.8, 0, 0.42), (3.6, 1.6, 0.06), "black")
    slope = math.tan(math.radians(cfg.ramp_angle_deg))
    # Keep the face order fixed for depth rasterization and contact generation.
    ramp_faces = ((0, 1, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    ramp_triangles = convert_faces_to_triangles(
        np.concatenate(ramp_faces), np.array([len(face) for face in ramp_faces])
    )
    for i in range(6):
        for j in range(2):
            x0, y0 = i * 0.6, -0.8 + j * 0.8
            x1, y1 = x0 + 0.6, y0 + 0.8
            sx = slope * (1 if (i + j) % 3 else -1)
            sy = slope * (1 if j else -1) * (0.35 if i in (2, 3) else 0.0)
            top = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=float)
            top[:, 2] = 0.45 + sx * (top[:, 0] - (x0 + x1) / 2) + sy * (top[:, 1] - (y0 + y1) / 2) + 0.053
            vertices = np.vstack([top, top.copy()])
            vertices[4:, 2] = 0.42
            parts.append((f"wood_{i}_{j}", "wood", trimesh.Trimesh(vertices, ramp_triangles, process=False)))
    for i, row in enumerate(cfg.platform_heights):
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
    return parts


def separate_ramp_base(stage: Usd.Stage) -> None:
    """Inset the visible ramp base by 1 mm after collision and depth geometry are imported."""
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    base = next(prim for prim in stage.Traverse() if prim.GetName() == "wood_base")
    wood = next(prim for prim in stage.Traverse() if prim.GetName() == "wood_0_0")
    base = base.GetChild("mesh")
    wood = wood.GetChild("mesh")
    source = UsdGeom.Mesh(base)
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32).copy()
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    points[:, :2] = center[:2] + (points[:, :2] - center[:2]) * (1.0 - 0.002 / (upper - lower)[:2])
    visual = UsdGeom.Mesh.Define(stage, base.GetPath().GetParentPath().AppendChild("wood_base_visual"))
    visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    visual.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    visual.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    visual.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        UsdGeom.PrimvarsAPI(source).GetPrimvar("st").Get()
    )
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(
        UsdShade.MaterialBindingAPI(base).ComputeBoundMaterial()[0]
    )
    faces = np.asarray(source.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    top_faces = np.flatnonzero(np.all(np.isclose(points[faces, 2], upper[2]), axis=1))
    subset = UsdGeom.Subset.CreateGeomSubset(
        visual, "wood_top", UsdGeom.Tokens.face, top_faces.tolist(), "materialBind"
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
        UsdShade.MaterialBindingAPI(wood).ComputeBoundMaterial()[0]
    )
    UsdGeom.Imageable(base).MakeInvisible()


def _container_mesh(top: float):
    """Triangulate a closed tray without overlapping internal contact surfaces."""
    bottom = top - 0.03
    axes = (
        np.array([5.4, 5.5, 11.5, 11.6]),
        np.array([-0.7, -0.6, 0.6, 0.7]),
        np.array([bottom - 0.1, bottom, top]),
    )
    cells = [(i, j, k) for i, j, k in itertools.product(range(3), range(3), range(2)) if k == 0 or i != 1 or j != 1]
    mesh = trimesh.voxel.ops.multibox(cells)
    occupied = set(cells)
    neighbors = np.rint(mesh.triangles_center + 0.5 * mesh.face_normals).astype(int)
    mesh.update_faces([tuple(neighbor) not in occupied for neighbor in neighbors])
    mesh.remove_unreferenced_vertices()
    # Map the exposed unit-cell corners to the tray's nonuniform wall and floor intervals.
    corners = np.rint(mesh.vertices + 0.5).astype(int)
    vertices = np.column_stack([axis[corners[:, i]] for i, axis in enumerate(axes)])
    return vertices.astype(np.float32), mesh.faces


def _granular(cfg: FixedCourseCfg):
    parts = []
    for name, start, end in (("approach", -1.5, 5.5), ("end", 11.5, 16.5)):
        x = np.round(np.arange(start, end + 0.025, 0.05), 6)
        y = np.round(np.arange(-4.0, 4.025, 0.05), 6)
        heights = np.broadcast_to((np.clip((x - 1.0) / 2.5, 0, 1) * cfg.granular_top)[:, None], (len(x), len(y)))
        vertices, faces = convert_height_field_to_mesh(heights, horizontal_scale=0.05, vertical_scale=1.0)
        vertices[:, :2] = np.round(vertices[:, :2] + (start, -4.0), 6)
        parts.append((name, name, trimesh.Trimesh(vertices, faces, process=False)))
    vertices, faces = _container_mesh(cfg.granular_top)
    parts.append(("granular_container", None, trimesh.Trimesh(vertices, faces, process=False)))
    return parts


def fixed_course_terrain(difficulty: float, cfg: FixedCourseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one fixed course tile; its measured layout is independent of difficulty."""
    parts = _granular(cfg) if cfg.layout == "granular" else _platforms_and_ramps(cfg)
    mesh = trimesh.util.concatenate([part for _, _, part in parts])
    # Preserve part ranges through generation for separate materials and colliders at import.
    mesh.metadata["course_parts"] = [
        (name, material, len(part.vertices), len(part.faces)) for name, material, part in parts
    ]
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    mesh.apply_translation(origin)
    return [mesh], origin


class FixedCourseImporter(TerrainImporter):
    """Import a generated course while retaining its part materials and collision settings."""

    def __init__(self, cfg: TerrainImporterCfg):
        generator = cfg.terrain_generator
        if generator.num_rows != 1 or generator.num_cols != 1 or generator.use_cache or generator.border_width:
            raise ValueError("Fixed courses require one uncached terrain tile without a border.")
        self._texture_directory = TemporaryDirectory(prefix="g1-course-textures-")
        super().__init__(cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        from isaaclab.sim.spawners.materials import spawn_physics_material  # noqa: PLC0415

        cfg = self.cfg.terrain_generator.sub_terrains["course"]
        directory = Path(self._texture_directory.name)
        _textures(directory)
        stage = sim_utils.get_current_stage()
        root = f"{self.cfg.prim_path}/{name}"
        materials = {}
        for _, material, _, _ in mesh.metadata["course_parts"]:
            if material is None or material in materials:
                continue
            path = f"{root}/Looks/{material}"
            material_cfg = cfg.visual_materials[material]
            material_cfg.func(path, material_cfg)
            if material in cfg.textures:
                _apply_texture(stage, path, directory / cfg.textures[material])
            materials[material] = path
        physics = f"{root}/Physics"
        spawn_physics_material(physics, cfg.physics_material)
        vertex_start, face_start = 0, 0
        for part, material, vertex_count, face_count in mesh.metadata["course_parts"]:
            vertices = mesh.vertices[vertex_start : vertex_start + vertex_count]
            faces = mesh.faces[face_start : face_start + face_count] - vertex_start
            imported = _mesh(
                f"{root}/{part}",
                vertices,
                faces,
                cfg,
                material=materials.get(material),
                physics=physics if material is not None else None,
                uv_scale=1.0 if cfg.layout == "granular" else 0.6,
            )
            if cfg.layout == "granular":
                imported.CreateDisplayColorAttr([(0.22, 0.23, 0.25) if material else (0.18, 0.19, 0.20)])
            vertex_start += vertex_count
            face_start += face_count
        self.terrain_prim_paths.append(root)
