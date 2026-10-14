# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack MicroDuck's warehouse visuals and prepare its authored collisions for MJWarp."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import trimesh
from bundle import write_binary_files
from PIL import Image
from scipy.spatial import cKDTree

import omni.client
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

SOURCE = (
    "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Environments/MicroDuckWarehouse/microduck_warehouse_static.usdc"
)
SOURCE_SHA256 = "4ac3e17232179a72086a3ff7c5af4d4546694f15a4931a0d21478a4a523da419"
ROOT = "/World/envs/env_0/Factory"
BOUNDS = (-5.85, 5.85, -8.85, 3.65)
MAX_FREE_PROPS = 12


def prepare_warehouse_physics(stage_path: Path, output: Path) -> dict:
    """Keep fixtures fixed and make nearby floor props free bodies with nominal density.

    The source is static and has no authored masses. Props use uniform 250 kg/m³
    density and convex collision geometry; these are illustrative physical settings.
    MJWarp contacts use convex meshes. Disconnected rack and pallet components are
    kept separate so a whole shelf row does not become one solid collision box.
    """
    stage = Usd.Stage.Open(str(stage_path))
    output.mkdir(parents=True, exist_ok=True)
    fixtures = Usd.Stage.CreateNew(str(output / "fixtures.usdc"))
    fixture_root = UsdGeom.Xform.Define(fixtures, "/Warehouse")
    fixtures.SetDefaultPrim(fixture_root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(fixtures, 1.0)
    UsdGeom.SetStageUpAxis(fixtures, UsdGeom.Tokens.z)
    cache, geometries, props = UsdGeom.XformCache(), {}, []
    shape_count = 0

    def geometry(prim: Usd.Prim) -> list[trimesh.Trimesh]:
        if prim.IsA(UsdGeom.Cube):
            return [trimesh.creation.box(extents=[UsdGeom.Cube(prim).GetSizeAttr().Get()] * 3)]
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Unsupported warehouse collider: {prim.GetPath()}: {prim.GetTypeName()}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = mesh.GetFaceVertexCountsAttr().Get()
        key = hashlib.sha256(points.tobytes() + indices.tobytes() + np.asarray(counts).tobytes()).hexdigest()
        if key not in geometries:
            faces, start = [], 0
            for count in counts:
                faces.extend((indices[start], indices[start + i], indices[start + i + 1]) for i in range(1, count - 1))
                start += count
            original = trimesh.Trimesh(points, faces, process=True)
            components = (
                original.split(only_watertight=False)
                if any(n in str(prim.GetPath()) for n in ("Rack", "Pallet"))
                else [original]
            )
            hulls = []
            for component in components:
                if (
                    len(component.vertices) < 4
                    or np.linalg.matrix_rank(component.vertices - component.vertices.mean(axis=0), tol=1e-4) < 3
                ):
                    hull = component
                else:
                    hull = component.convex_hull
                    if len(hull.faces) > 128:
                        reduced = hull.simplify_quadric_decimation(face_count=128).convex_hull
                        if np.all(
                            np.abs(reduced.bounds - hull.bounds) <= np.maximum(hull.extents * 0.02, 1e-6)
                        ) and np.isclose(reduced.volume, hull.volume, rtol=0.02):
                            hull = reduced
                hulls.append(hull)
            geometries[key] = hulls
        return geometries[key]

    def add_shape(target: Usd.Stage, path: str, hull: trimesh.Trimesh) -> None:
        # Apply thickness in world metres, after the source asset's nested scales.
        points = hull.vertices
        _, singular, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        if singular[-1] < singular[0] * 1e-5:
            coordinates = points @ axes.T
            bounds = np.array([coordinates.min(axis=0), coordinates.max(axis=0)])
            hull = trimesh.creation.box(extents=np.maximum(bounds[1] - bounds[0], 0.005))
            transform = np.eye(4)
            transform[:3, :3] = axes.T
            transform[:3, 3] = bounds.mean(axis=0) @ axes
            hull.apply_transform(transform)
        mesh = UsdGeom.Mesh.Define(target, path)
        mesh.CreatePointsAttr(hull.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(hull.faces))
        mesh.CreateFaceVertexIndicesAttr(hull.faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    # The task's existing plane remains the contact floor; decorations have no physics.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT), Usd.TraverseInstanceProxies()):
        if (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ):
            continue
        path = str(prim.GetPath())
        if path.startswith(ROOT + "/Floor/") or path.startswith(ROOT + "/Clutter/"):
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        for source in geometry(prim):
            hull = source.copy()
            hull.apply_transform(matrix.T)
            add_shape(fixtures, f"/Warehouse/Shape_{shape_count}", hull)
            shape_count += 1
    # Retain the four containment walls, including the warehouse's open ends.
    for index, collider in enumerate(warehouse_boundaries()):
        cube = UsdGeom.Cube.Define(fixtures, f"/Warehouse/Boundary_{index}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*collider["position"]))
        cube.AddScaleOp().Set(Gf.Vec3f(*collider["size"]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    items = list(stage.GetPrimAtPath(ROOT + "/Clutter").GetChildren())
    bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    free = {
        item.GetPath()
        for item in sorted(
            items,
            key=lambda p: np.linalg.norm(np.asarray(bbox.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint())[:2]),
        )[:MAX_FREE_PROPS]
    }
    for item in items:
        hulls = []
        for prim in Usd.PrimRange(item, Usd.TraverseInstanceProxies()):
            if (
                not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            ):
                continue
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            for source in geometry(prim):
                hull = source.copy()
                hull.apply_transform(matrix.T)
                hulls.append(hull)
        if not hulls:
            raise ValueError(f"Floor prop has no enabled collision geometry: {item.GetPath()}")
        if item.GetPath() not in free:
            for hull in hulls:
                add_shape(fixtures, f"/Warehouse/Shape_{shape_count}", hull)
                shape_count += 1
            continue
        body = trimesh.util.concatenate(hulls)
        body.density = 250.0
        center = body.center_mass
        if not np.isfinite(center).all() or body.mass <= 0 or np.min(np.linalg.eigvalsh(body.moment_inertia)) <= 0:
            raise ValueError(f"Invalid floor prop inertia: {item.GetPath()}")
        target = Usd.Stage.CreateNew(str(output / f"{item.GetName()}.usdc"))
        root = UsdGeom.Xform.Define(target, "/Object").GetPrim()
        target.SetDefaultPrim(root)
        UsdGeom.SetStageMetersPerUnit(target, 1.0)
        UsdGeom.SetStageUpAxis(target, UsdGeom.Tokens.z)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(float(body.mass))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        values, axes = np.linalg.eigh(body.moment_inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(Gf.Matrix3d(axes.T.tolist()).ExtractRotation().GetQuat()))
        for index, hull in enumerate(hulls):
            hull.apply_translation(-center)
            add_shape(target, f"/Object/Shape_{index}", hull)
        target.GetRootLayer().Save()
        props.append(
            {
                "name": item.GetName(),
                "sourcePath": str(item.GetPath()),
                "position": center.tolist(),
                "mass": float(body.mass),
            }
        )
    fixtures.GetRootLayer().Save()
    print(f"Warehouse physics: {shape_count + 4} fixed colliders, {len(props)} free props", flush=True)
    return {"fixtures": shape_count + 4, "props": props, "density": 250.0}


def resolve_warehouse(output: Path) -> Path:
    """Cache the pinned scene and resolve its USD references without modifying server assets."""
    output.mkdir(parents=True, exist_ok=True)
    public = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/"
    paths, pending, provenance = {}, [SOURCE], []

    def path(url: str) -> Path:
        return output / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".usdc")

    async def download(url: str) -> None:
        result, _, content = await asyncio.wait_for(omni.client.read_file_async(url), 30)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"Warehouse source unavailable: {url}: {result}")
        raw = bytes(content)
        sha = hashlib.sha256(raw).hexdigest()
        if url == SOURCE and sha != SOURCE_SHA256:
            raise ValueError("Warehouse scene changed; review its layout and collision proxies before exporting")
        temporary = path(url).with_suffix(".input.usdc")
        temporary.write_bytes(raw)
        layer = Sdf.Layer.FindOrOpen(str(temporary))
        if url == SOURCE:
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.Remove("/World/envs/env_0/Robot"))
            if not layer.Apply(edit):
                raise ValueError("Warehouse robot path changed")

        def remap(value: str) -> str:
            if not value:
                return value
            if value.startswith("../../../"):
                resolved = public + value.removeprefix("../../../")
            elif "://" in value:
                resolved = value
            elif url.startswith("omniverse://"):
                resolved = url.rsplit("/", 1)[0] + "/" + value.removeprefix("./")
            else:
                resolved = urljoin(url, value)
            if resolved.lower().endswith((".usd", ".usda", ".usdc")):
                if resolved not in paths:
                    paths[resolved] = path(resolved)
                    pending.append(resolved)
                return str(path(resolved))
            return resolved

        UsdUtils.ModifyAssetPaths(layer, remap)
        layer.Export(str(path(url)))
        provenance.append({"url": url, "sha256": sha})
        temporary.unlink()

    async def collect() -> None:
        paths[SOURCE] = path(SOURCE)
        while pending:
            batch = pending[:6]
            del pending[:6]
            await asyncio.gather(*(download(url) for url in batch))
            if len(paths) > 400:
                raise ValueError("Warehouse dependency layout changed")

    asyncio.run(collect())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return path(SOURCE)


def _material(prim: Usd.Prim) -> tuple[dict, str | None, str]:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    result = {"color": [0.65, 0.67, 0.69], "roughness": 0.7}
    texture, uv_name = None, "st"
    if not material:
        return result, texture, uv_name
    for shader_prim in Usd.PrimRange(material.GetPrim()):
        if shader_prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(shader_prim)
        if not any(
            shader.GetInput(name)
            for name in ("diffuseColor", "ColorAlbedo", "diffuse_color_constant", "AlbedoTexture", "diffuse_texture")
        ):
            continue
        for name in ("diffuseColor", "ColorAlbedo", "diffuse_color_constant"):
            value = shader.GetInput(name)
            if value and value.Get() is not None:
                result["color"] = list(value.Get())[:3]
        value = shader.GetInput("roughness")
        if value and value.Get() is not None:
            result["roughness"] = float(value.Get())
        value = shader.GetInput("opacity")
        if value and value.Get() is not None:
            result["opacity"] = float(value.Get())
        diffuse = shader.GetInput("diffuseColor")
        connected = diffuse.GetConnectedSource() if diffuse else None
        image_shader = UsdShade.Shader(connected[0]) if connected else shader
        for name in ("AlbedoTexture", "diffuse_texture", "file"):
            value = image_shader.GetInput(name)
            if value and value.Get() is not None and hasattr(value.Get(), "path"):
                texture = value.Get().path
                # The collected scene's base-material references use the Isaac prefix;
                # the public copy is under NVIDIA and contains the same texture bytes.
                texture = texture.replace("/6.2/Isaac/Materials/", "/6.2/NVIDIA/Materials/")
                coordinates = image_shader.GetInput("st")
                reader = coordinates.GetConnectedSource() if coordinates else None
                if reader:
                    variable = UsdShade.Shader(reader[0]).GetInput("varname")
                    if variable and variable.Get():
                        uv_name = str(variable.Get())
                break
    if texture:
        result["color"] = [1.0, 1.0, 1.0]
    return result, texture, uv_name


def _geometry(prim: Usd.Prim, uv_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    triangles, start = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        triangles.extend((start, start + i, start + i + 1) for i in range(1, count - 1))
        start += count
    corners = np.asarray(triangles)
    faces = indices[corners]
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
    if uv and uv.HasValue():
        values = np.asarray(uv.ComputeFlattened(), dtype=np.float64)
        coordinates = values[corners] if uv.GetInterpolation() == "faceVarying" else values[faces]
    else:
        coordinates = np.zeros((*faces.shape, 2))
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        faces = faces[:, ::-1]
        coordinates = coordinates[:, ::-1]
    original = trimesh.Trimesh(points, faces, process=False)
    valid = original.nondegenerate_faces()
    faces, coordinates = faces[valid], coordinates[valid]
    original = trimesh.Trimesh(points, faces, process=False)
    original.merge_vertices()
    geometry = original
    if len(original.faces) > 800:
        reduced = original.simplify_quadric_decimation(face_count=800)
        # These are decorative meshes; physics uses separate convex colliders.
        # Preserve silhouette and volume rather than tiny scanned interior holes.
        if np.all(np.abs(reduced.bounds - original.bounds) <= np.maximum(original.extents * 0.02, 1e-6)) and (
            not original.is_watertight or np.isclose(reduced.volume, original.volume, rtol=0.02, atol=0.0)
        ):
            geometry = reduced
            # Transfer each reduced face through one source UV patch, preserving texture seams.
            source = cKDTree(points[faces].mean(axis=1)).query(geometry.triangles_center)[1]
            barycentric = trimesh.triangles.points_to_barycentric(
                np.repeat(points[faces[source]], 3, axis=0), geometry.triangles.reshape(-1, 3)
            ).reshape(-1, 3, 3)
            coordinates = np.einsum("fvi,fij->fvj", barycentric, coordinates[source])
    vertices = geometry.triangles.astype("<f4").reshape(-1, 3)
    coordinates = coordinates.astype("<f4").reshape(-1, 2)
    if not np.isfinite(vertices).all() or not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite warehouse geometry: {prim.GetPath()}")
    return vertices, np.arange(len(vertices), dtype="<u4"), coordinates


def pack_warehouse(stage_path: Path, output: Path, props: list[dict] | None = None) -> dict:
    """Use the resolved Factory subset; the captured robot and RTX cameras are excluded."""
    stage = Usd.Stage.Open(str(stage_path))
    root = stage.GetPrimAtPath(ROOT)
    if not root:
        raise ValueError("Warehouse Factory root changed")
    output.mkdir(parents=True, exist_ok=True)
    cache, prototypes, geometries, textures = UsdGeom.XformCache(), {}, {}, {}
    data, meshes = bytearray(), []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if (
            mesh.ComputeVisibility() == "invisible"
            or mesh.ComputePurpose() in ("guide", "proxy")
            or "collision" in prim.GetName().lower()
            or not mesh.GetPointsAttr().Get()
        ):
            continue
        material, texture, uv_name = _material(prim)
        # Identical shelf contents share one reduced geometry and one instanced draw.
        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
        uv_identity = np.asarray(primvar.ComputeFlattened()).tobytes() if primvar and primvar.HasValue() else b""
        identity = hashlib.sha256(
            np.asarray(mesh.GetPointsAttr().Get()).tobytes()
            + np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).tobytes()
            + np.asarray(mesh.GetFaceVertexCountsAttr().Get()).tobytes()
            + uv_identity
            + str((uv_name, material, texture, mesh.GetOrientationAttr().Get())).encode()
        ).hexdigest()
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float32)
        body = -1
        for prop in props or ():
            if str(prim.GetPath()).startswith(prop["sourcePath"] + "/"):
                body = prop["body"]
                transform[3, :3] -= np.asarray(prop["position"], dtype=np.float32)
                break
        instance_key = (identity, body)
        if instance_key in prototypes:
            prototypes[instance_key]["instances"].append(transform.reshape(-1).tolist())
            continue
        if identity in geometries:
            record = {**geometries[identity], "body": body, "instances": [transform.reshape(-1).tolist()]}
            meshes.append(record)
            prototypes[instance_key] = record
            continue
        vertices, indices, uv = _geometry(prim, uv_name)
        record = {
            "body": body,
            "environment": True,
            "name": prim.GetName(),
            "instances": [transform.reshape(-1).tolist()],
            **material,
            "flatShading": False,
            "vertexOffset": len(data),
            "vertexCount": len(vertices),
        }
        data.extend(vertices.tobytes())
        record.update(indexOffset=len(data), indexCount=len(indices), indexType="uint32")
        data.extend(indices.tobytes())
        record["uvOffset"] = len(data)
        data.extend(uv.tobytes())
        if texture:
            if texture not in textures:
                result, _, content = omni.client.read_file(texture)
                if result != omni.client.Result.OK:
                    raise RuntimeError(f"Warehouse texture unavailable: {texture}: {result}")
                image = Image.open(io.BytesIO(bytes(content)))
                image.thumbnail((512, 512))
                path = output / (hashlib.sha256(texture.encode()).hexdigest()[:16] + ".webp")
                image.save(path, format="WEBP", quality=85)
                textures[texture] = path.name
            record["texture"] = textures[texture]
        meshes.append(record)
        prototypes[instance_key] = record
        geometries[identity] = record
    files = write_binary_files(gzip.compress(bytes(data), mtime=0), output, "warehouse.bin.gz")
    for stale in output.glob("warehouse.bin*"):
        if stale.name not in files:
            stale.unlink()
    description = {"file": "warehouse.bin.gz", "files": files, "byteLength": len(data), "meshes": meshes}
    (output / "warehouse.json").write_text(json.dumps(description, indent=2) + "\n")
    print(
        f"Warehouse: {len(geometries)} geometries, {sum(len(m['instances']) for m in meshes)} instances",
        flush=True,
    )
    return description


def warehouse_boundaries() -> list[dict]:
    """Keep the robot and free props inside the walkable room, including its open ends."""
    x0, x1, y0, y1 = BOUNDS
    height, width = 1.0, 0.3
    return [
        {"position": [x0 - width / 2, (y0 + y1) / 2, height / 2], "size": [width, y1 - y0 + 2 * width, height]},
        {"position": [x1 + width / 2, (y0 + y1) / 2, height / 2], "size": [width, y1 - y0 + 2 * width, height]},
        {"position": [(x0 + x1) / 2, y0 - width / 2, height / 2], "size": [x1 - x0, width, height]},
        {"position": [(x0 + x1) / 2, y1 + width / 2, height / 2], "size": [x1 - x0, width, height]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_warehouse(args.output / "source")
    pack_warehouse(stage, args.output)
    print(f"Resolved warehouse: {stage}", flush=True)


if __name__ == "__main__":
    main()
