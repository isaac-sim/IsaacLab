"""Build the self-contained, solid-color USD assets used by the fixed G1 demos."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils.mesh import convert_faces_to_triangles

PLATFORM_HEIGHTS = (
    (0.51, 0.56, 0.50),
    (0.56, 0.51, 0.58),
    (0.50, 0.56, 0.53),
    (0.55, 0.50, 0.58),
    (0.50, 0.57, 0.51),
    (0.54, 0.51, 0.56),
)
GRANULAR_TOP = 2.5 * math.tan(math.radians(10.0))
COLORS = {
    "white": (0.73, 0.75, 0.75),
    "steel": (0.26, 0.28, 0.29),
    "black": (0.055, 0.06, 0.065),
    "wood": (0.5136162501, 0.3179454321, 0.1250055410),
    "wall": (0.68, 0.67, 0.64),
    "floor": (0.39, 0.39, 0.37),
    "gravel": (0.1454259782, 0.1379929152, 0.1172773380),
    "approach": (0.055, 0.065, 0.08),
    "end": (0.055, 0.065, 0.08),
    "tray": (0.18, 0.19, 0.20),
}


def platforms_and_ramps(layout: str) -> list[tuple[str, str, trimesh.Trimesh]]:
    """Reproduce the measured platforms, wedges, supports, and surrounding geometry."""
    parts = []

    def box(name, center, size, material):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(center)
        parts.append((name, material, mesh))

    xmin = -2.9 if layout == "platforms" else -1.4
    box("floor", (1.8, 1, -0.05), (18, 14, 0.1), "floor")
    box("wall", (xmin - 0.05, 1.5, 2.3), (0.10, 16, 4.6), "wall")
    box("wood_base", (1.8, 0, 0.42), (3.6, 1.6, 0.06), "black")
    slope = math.tan(math.radians(10.0))
    ramp_faces = ((0, 1, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    triangles = convert_faces_to_triangles(np.concatenate(ramp_faces), np.array([len(face) for face in ramp_faces]))
    for column in range(6):
        for row in range(2):
            left, bottom = column * 0.6, -0.8 + row * 0.8
            right, top = left + 0.6, bottom + 0.8
            slope_x = slope * (1 if (column + row) % 3 else -1)
            slope_y = slope * (1 if row else -1) * (0.35 if column in (2, 3) else 0.0)
            surface = np.array([[left, bottom, 0], [right, bottom, 0], [right, top, 0], [left, top, 0]], dtype=float)
            surface[:, 2] = (
                0.45
                + slope_x * (surface[:, 0] - (left + right) / 2)
                + slope_y * (surface[:, 1] - (bottom + top) / 2)
                + 0.053
            )
            vertices = np.vstack([surface, surface.copy()])
            vertices[4:, 2] = 0.42
            parts.append((f"wood_{column}_{row}", "wood", trimesh.Trimesh(vertices, triangles, process=False)))
    for column, heights in enumerate(PLATFORM_HEIGHTS):
        for row, height in enumerate(heights):
            center_x, center_y = (column + 0.5) * 0.6, 1.05 + (row + 0.5) * 0.5
            box(f"white_{column}_{row}", (center_x, center_y, height - 0.015), (0.59, 0.49, 0.03), "white")
            for leg, (offset_x, offset_y) in enumerate(((-0.23, -0.18), (0.23, -0.18), (-0.23, 0.18), (0.23, 0.18))):
                box(
                    f"leg_{column}_{row}_{leg}",
                    (center_x + offset_x, center_y + offset_y, (height - 0.03) / 2),
                    (0.045, 0.045, height - 0.03),
                    "steel",
                )
    box("gravel_lane", (-0.35, 1.25, 0.465), (0.7, 4.1, 0.03), "gravel")
    box("left_platform", (xmin / 2, 1.25, 0.425), (-xmin, 4.1, 0.05), "black")
    box("right_platform", (4.3, 1.25, 0.425), (1.4, 4.1, 0.05), "black")
    for step in range(4):
        box(
            f"background_step_{step}",
            (0, 3.15 + step * 0.3, 0.45 + (step + 1) * 0.075 / 2),
            (0.85, 0.3, (step + 1) * 0.075),
            "black",
        )
    return parts


def granular_course() -> list[tuple[str, str, trimesh.Trimesh]]:
    """Create the approach, exit, and tray; particle simulation remains in the task."""
    parts = []
    for name, start, end in (("approach", -1.5, 5.5), ("end", 11.5, 16.5)):
        columns = np.round(np.arange(start, end + 0.025, 0.05), 6)
        rows = np.round(np.arange(-4.0, 4.025, 0.05), 6)
        heights = np.broadcast_to(
            (np.clip((columns - 1.0) / 2.5, 0, 1) * GRANULAR_TOP)[:, None], (len(columns), len(rows))
        )
        vertices, faces = convert_height_field_to_mesh(heights, horizontal_scale=0.05, vertical_scale=1.0)
        vertices[:, :2] = np.round(vertices[:, :2] + (start, -4.0), 6)
        parts.append((name, name, trimesh.Trimesh(vertices, faces, process=False)))
    bottom = GRANULAR_TOP - 0.03
    axes = (
        np.array([5.4, 5.5, 11.5, 11.6]),
        np.array([-0.7, -0.6, 0.6, 0.7]),
        np.array([bottom - 0.1, bottom, GRANULAR_TOP]),
    )
    cells = [
        cell for cell in itertools.product(range(3), range(3), range(2)) if cell[2] == 0 or cell[0] != 1 or cell[1] != 1
    ]
    mesh = trimesh.voxel.ops.multibox(cells)
    occupied = set(cells)
    neighbors = np.rint(mesh.triangles_center + 0.5 * mesh.face_normals).astype(int)
    mesh.update_faces([tuple(neighbor) not in occupied for neighbor in neighbors])
    mesh.remove_unreferenced_vertices()
    corners = np.rint(mesh.vertices + 0.5).astype(int)
    mesh.vertices = np.column_stack([axis[corners[:, dimension]] for dimension, axis in enumerate(axes)]).astype(
        np.float32
    )
    parts.append(("granular_container", "tray", mesh))
    return parts


def add_ramp_appearance_variants(stage: Usd.Stage) -> None:
    """Keep original sensor geometry until the task selects its spectator appearance."""
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Course/wood_base/mesh"))
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32).copy()
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    points[:, :2] = center[:2] + (points[:, :2] - center[:2]) * (1.0 - 0.002 / (upper - lower)[:2])
    visual = UsdGeom.Mesh.Define(stage, "/Course/wood_base/wood_base_visual")
    visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    visual.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    visual.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    visual.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Course/Looks/black"))
    )
    faces = np.asarray(source.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    top_faces = np.flatnonzero(np.all(np.isclose(points[faces, 2], upper[2]), axis=1))
    subset = UsdGeom.Subset.CreateGeomSubset(
        visual, "wood_top", UsdGeom.Tokens.face, top_faces.tolist(), "materialBind"
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Course/Looks/wood"))
    )
    variants = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("appearance")
    for name in ("simulation", "visual"):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            visual.GetPrim().SetActive(name == "visual")
            source.CreateVisibilityAttr(UsdGeom.Tokens.invisible if name == "visual" else UsdGeom.Tokens.inherited)
    variants.SetVariantSelection("simulation")


def export_course(output: Path, layout: str) -> Path:
    """Write a standalone USD with geometry, solid materials, and static contact properties."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{layout}.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Course").GetPrim())
    physics = UsdShade.Material.Define(stage, "/Course/Physics")
    physics_api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    physics_api.CreateStaticFrictionAttr(1.0)
    physics_api.CreateDynamicFrictionAttr(1.0)
    physics_api.CreateRestitutionAttr(0.0)
    parts = granular_course() if layout == "granular" else platforms_and_ramps(layout)
    for material_name in dict.fromkeys(material for _, material, _ in parts):
        material = UsdShade.Material.Define(stage, f"/Course/Looks/{material_name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[material_name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.96 if layout == "granular" else 0.9)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for name, material_name, geometry in parts:
        UsdGeom.Xform.Define(stage, f"/Course/{name}")
        mesh = UsdGeom.Mesh.Define(stage, f"/Course/{name}/mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(geometry.vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(geometry.faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding.Bind(UsdShade.Material(stage.GetPrimAtPath(f"/Course/Looks/{material_name}")))
        if material_name != "tray":
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
            mesh.GetPrim().CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int, custom=False).Set(1)
            binding.Bind(physics, materialPurpose="physics")
        if layout == "granular":
            mesh.CreateDisplayColorAttr([(0.18, 0.19, 0.20) if material_name == "tray" else (0.22, 0.23, 0.25)])
    if layout != "granular":
        add_ramp_appearance_variants(stage)
    stage.GetRootLayer().Save()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for layout in ("platforms", "ramps", "granular"):
        print(export_course(args.output, layout))


if __name__ == "__main__":
    main()
